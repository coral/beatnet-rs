//! Synchronous ONNX inference with explicitly retained recurrent state.
use crate::{Error, features::FEATURES, validate_probabilities};
use ort::{
    session::{IoBinding, Session},
    value::{Outlet, Tensor, TensorElementType, ValueType},
};
use std::path::Path;

/// A synchronous backend owned by the processing worker.
/// Each call advances exactly one 20 ms frame. Return finite probabilities in
/// [beat, downbeat, non-beat] order, normalized to one within 1e-4. Implementations
/// must retain recurrent state across calls and clear it in `reset`.
pub trait Inference: Send {
    fn infer(&mut self, features: &[f32; FEATURES]) -> Result<[f32; 3], Error>;
    fn reset(&mut self);
}
#[derive(Clone, Copy, Debug, Default)]
pub enum Model {
    #[default]
    One,
    Two,
    Three,
}
impl Model {
    pub fn bytes(self) -> &'static [u8] {
        match self {
            Self::One => include_bytes!("../models/beatnet-1.onnx"),
            Self::Two => include_bytes!("../models/beatnet-2.onnx"),
            Self::Three => include_bytes!("../models/beatnet-3.onnx"),
        }
    }
}
/// CPU ONNX session with fixed input/output buffers and explicit recurrent state.
pub struct OnnxModel {
    session: Session,
    binding: IoBinding,
    features: Tensor<f32>,
    hidden: Tensor<f32>,
    cell: Tensor<f32>,
}
impl OnnxModel {
    pub fn bundled(model: Model) -> Result<Self, Error> {
        Self::from_bytes(model.bytes())
    }
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::from_bytes(&std::fs::read(path)?)
    }
    /// Validate the fixed float32 interface, warm inference, and clear state.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        ort::init().with_telemetry(false).commit();
        let session = Session::builder()?
            .with_no_environment_execution_providers()
            .map_err(ort::Error::from)?
            .with_intra_threads(1)
            .map_err(ort::Error::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::from)?
            .with_parallel_execution(false)
            .map_err(ort::Error::from)?
            .commit_from_memory(bytes)?;
        validate_interface(
            session.inputs(),
            &[
                ("features", &[1, 1, FEATURES as i64]),
                ("hidden", &[2, 1, 150]),
                ("cell", &[2, 1, 150]),
            ],
        )?;
        validate_interface(
            session.outputs(),
            &[
                ("probabilities", &[3]),
                ("hidden_out", &[2, 1, 150]),
                ("cell_out", &[2, 1, 150]),
            ],
        )?;
        let mut binding = session.create_binding()?;
        let features = Tensor::from_array(([1usize, 1, FEATURES], vec![0f32; FEATURES]))?;
        let hidden = Tensor::from_array(([2usize, 1, 150], vec![0f32; 300]))?;
        let cell = Tensor::from_array(([2usize, 1, 150], vec![0f32; 300]))?;
        binding.bind_output(
            "probabilities",
            Tensor::from_array(([3usize], vec![0f32; 3]))?,
        )?;
        binding.bind_output(
            "hidden_out",
            Tensor::from_array(([2usize, 1, 150], vec![0f32; 300]))?,
        )?;
        binding.bind_output(
            "cell_out",
            Tensor::from_array(([2usize, 1, 150], vec![0f32; 300]))?,
        )?;
        let mut model = Self {
            session,
            binding,
            features,
            hidden,
            cell,
        };
        for _ in 0..3 {
            model.infer(&[0.; FEATURES])?;
        }
        model.reset();
        Ok(model)
    }
    /// Read-only state for numerical validation and diagnostics.
    pub fn recurrent_state(&self) -> (&[f32], &[f32]) {
        (
            self.hidden.try_extract_tensor::<f32>().unwrap().1,
            self.cell.try_extract_tensor::<f32>().unwrap().1,
        )
    }
}
impl Inference for OnnxModel {
    fn infer(&mut self, features: &[f32; FEATURES]) -> Result<[f32; 3], Error> {
        if features.iter().any(|value| !value.is_finite()) {
            return Err(Error::ModelInput);
        }
        self.features
            .try_extract_tensor_mut::<f32>()?
            .1
            .copy_from_slice(features);
        // Rebind after mutation: required by the ort API, even on CPU.
        self.binding.bind_input("features", &self.features)?;
        self.binding.bind_input("hidden", &self.hidden)?;
        self.binding.bind_input("cell", &self.cell)?;
        let outputs = self.session.run_binding(&self.binding)?;
        let p = outputs["probabilities"].try_extract_tensor::<f32>()?.1;
        if p.len() != 3 {
            return Err(Error::ModelOutput);
        }
        let probabilities = [p[0], p[1], p[2]];
        validate_probabilities(&probabilities)?;
        let h = outputs["hidden_out"].try_extract_tensor::<f32>()?.1;
        let c = outputs["cell_out"].try_extract_tensor::<f32>()?.1;
        if h.len() != 300 || c.len() != 300 || h.iter().chain(c).any(|value| !value.is_finite()) {
            return Err(Error::ModelOutput);
        }
        self.hidden
            .try_extract_tensor_mut::<f32>()?
            .1
            .copy_from_slice(h);
        self.cell
            .try_extract_tensor_mut::<f32>()?
            .1
            .copy_from_slice(c);
        Ok(probabilities)
    }
    fn reset(&mut self) {
        self.hidden
            .try_extract_tensor_mut::<f32>()
            .expect("owned float32 state tensor")
            .1
            .fill(0.);
        self.cell
            .try_extract_tensor_mut::<f32>()
            .expect("owned float32 state tensor")
            .1
            .fill(0.);
    }
}

// Reject incompatible external exports before binding buffers or running warmup.
fn validate_interface(
    outlets: &[Outlet],
    expected: &[(&'static str, &[i64])],
) -> Result<(), Error> {
    if outlets.len() != expected.len() {
        return Err(Error::ModelInterface(
            "expected three inputs and three outputs",
        ));
    }
    for &(name, dimensions) in expected {
        let valid = outlets.iter().any(|outlet| outlet.name() == name && matches!(outlet.dtype(),
            ValueType::Tensor { ty: TensorElementType::Float32, shape, .. } if &shape[..] == dimensions));
        if !valid {
            return Err(Error::ModelInterface(name));
        }
    }
    Ok(())
}
