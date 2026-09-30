//! Streaming BeatNet beat/downbeat tracking without Python or an async runtime.
//!
//! Construct and process on a worker thread. Only [`transport::AudioProducer`]
//! belongs in an audio callback. Frame/event times refer to the source audio,
//! not wall-clock delivery time; the frontend needs 706 samples of lookahead.
#![forbid(unsafe_code)]

#[cfg(feature = "live")]
pub mod capture;
pub mod features;
pub mod model;
pub mod tracker;
pub mod transport;

use features::{FeatureExtractor, SAMPLE_RATE};
pub use model::{Inference, Model, OnnxModel};
mod resample;
use resample::RateConverter;
use tracker::ParticleFilter;
pub use tracker::TrackerConfig;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(&'static str),
    #[error("PCM contains non-finite samples")]
    InvalidPcm,
    #[error("invalid model probabilities or recurrent output")]
    ModelOutput,
    #[error("processor must be reset after a backend failure")]
    NeedsReset,
    #[error("timestamp must be finite, nonnegative, and increase between frames")]
    InvalidTime,
    #[error("model requires finite feature values")]
    ModelInput,
    #[error("incompatible ONNX interface: {0}")]
    ModelInterface(&'static str),
    #[cfg(feature = "live")]
    #[error(transparent)]
    Capture(#[from] cpal::Error),
    #[error("stream has been finished; reset before processing more audio")]
    Finished,
    #[error("incomplete interleaved frame at end of stream")]
    IncompleteFrame,
    #[error(transparent)]
    Onnx(#[from] ort::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    ResamplerConstruction(#[from] rubato::ResamplerConstructionError),
    #[error(transparent)]
    Resample(#[from] rubato::ResampleError),
}
/// Source audio format, pretrained model, and tracker settings.
#[derive(Clone, Debug)]
pub struct BeatNetConfig {
    /// Input samples per second (8,000 through 192,000).
    pub sample_rate: u32,
    /// Interleaved channel count (1 through 64).
    pub channels: u16,
    pub model: Model,
    pub tracker: TrackerConfig,
}
impl Default for BeatNetConfig {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            channels: 1,
            model: Model::One,
            tracker: TrackerConfig::default(),
        }
    }
}
impl BeatNetConfig {
    /// Validate settings without allocating DSP buffers or loading a model.
    pub fn validate(&self) -> Result<(), Error> {
        validate_audio_format(self.sample_rate, self.channels)?;
        self.tracker.validate()
    }
}

pub(crate) fn validate_audio_format(rate: u32, channels: u16) -> Result<(), Error> {
    if !(8_000..=192_000).contains(&rate) || !(1..=64).contains(&channels) {
        return Err(Error::Config(
            "sample rate must be 8000..=192000; channels 1..=64",
        ));
    }
    Ok(())
}

pub(crate) fn validate_probabilities(probabilities: &[f32; 3]) -> Result<(), Error> {
    if probabilities
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || (probabilities.iter().sum::<f32>() - 1.0).abs() > 1e-4
    {
        return Err(Error::ModelOutput);
    }
    Ok(())
}

/// An accepted beat on the source audio timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeatEvent {
    /// Seconds from the beginning of the source audio.
    pub time: f64,
    pub bpm: f32,
    pub downbeat: bool,
    /// Circular concentration of beat particles, not a calibrated probability.
    pub confidence: f32,
}
/// Posterior estimates for a single 20 ms feature frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingState {
    pub time: f64,
    pub bpm: Option<f32>,
    /// Phase within a beat, in [0, 1); absent before the first accepted beat.
    pub phase: Option<f32>,
    pub meter: Option<u8>,
    pub confidence: f32,
    /// Class probabilities: beat, downbeat, non-beat.
    pub probabilities: [f32; 3],
}
/// State and optional beat event produced together for one feature frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameOutput {
    pub state: TrackingState,
    pub event: Option<BeatEvent>,
}

struct Engine<M> {
    features: FeatureExtractor,
    model: M,
    tracker: ParticleFilter,
    origin: f64,
    frames: u64,
    frame_limit: Option<u64>,
}
impl<M: Inference> Engine<M> {
    fn push(&mut self, sample: f32, emit: &mut impl FnMut(FrameOutput)) -> Result<(), Error> {
        if self.frame_limit.is_some_and(|limit| self.frames >= limit) {
            return Ok(());
        }
        if let Some((position, features)) = self.features.push(sample) {
            let probabilities = self.model.infer(&features)?;
            let (state, event) = self.tracker.process(
                probabilities,
                self.origin + position as f64 / SAMPLE_RATE as f64,
            )?;
            self.frames += 1;
            emit(FrameOutput { state, event });
        }
        Ok(())
    }
}
/// Synchronous PCM processor. `process` accepts arbitrary interleaved chunk sizes,
/// including chunks ending partway through a multichannel frame.
pub struct BeatNet<M = OnnxModel> {
    config: BeatNetConfig,
    engine: Engine<M>,
    converter: Option<RateConverter>,
    channel: usize,
    sum: f64,
    input_frames: u64,
    status: Status,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Active,
    Failed,
    Finished,
}

impl BeatNet<OnnxModel> {
    /// Load and warm the selected model, then allocate processing buffers.
    pub fn new(config: BeatNetConfig) -> Result<Self, Error> {
        config.validate()?;
        let model = OnnxModel::bundled(config.model)?;
        Self::with_model(config, model)
    }
}
impl<M: Inference> BeatNet<M> {
    /// Use a caller-supplied synchronous backend; its recurrent state is reset.
    pub fn with_model(config: BeatNetConfig, mut model: M) -> Result<Self, Error> {
        config.validate()?;
        let tracker = ParticleFilter::new(config.tracker.clone())?;
        let converter = if config.sample_rate == SAMPLE_RATE {
            None
        } else {
            Some(RateConverter::new(config.sample_rate)?)
        };
        model.reset();
        Ok(Self {
            config,
            engine: Engine {
                features: FeatureExtractor::new(),
                model,
                tracker,
                origin: 0.,
                frames: 0,
                frame_limit: None,
            },
            converter,
            channel: 0,
            sum: 0.,
            input_frames: 0,
            status: Status::Active,
        })
    }
    /// Output callback runs on the calling thread; it should not block.
    /// Invalid PCM is rejected before consuming any of the supplied chunk.
    /// A backend error or unwinding callback puts this processor in `NeedsReset`
    /// state. Reset before resuming; already emitted outputs are not rolled back.
    pub fn process(&mut self, pcm: &[f32], mut emit: impl FnMut(FrameOutput)) -> Result<(), Error> {
        match self.status {
            Status::Finished => return Err(Error::Finished),
            Status::Failed => return Err(Error::NeedsReset),
            Status::Active => {}
        }
        if pcm.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidPcm);
        }
        // A backend or user callback can unwind after advancing only part of a chunk.
        // Stay failed unless the entire operation returns successfully.
        self.status = Status::Failed;
        let channels = usize::from(self.config.channels);
        if channels == 1 {
            self.input_frames += pcm.len() as u64;
            self.process_mono(pcm, &mut emit)?;
        } else {
            let mut mono = [0.; features::HOP];
            for block in pcm.chunks(mono.len() * channels) {
                let mut frames = 0;
                for &sample in block {
                    self.sum += f64::from(sample);
                    self.channel += 1;
                    if self.channel == channels {
                        mono[frames] = (self.sum / channels as f64) as f32;
                        frames += 1;
                        self.sum = 0.;
                        self.channel = 0;
                    }
                }
                self.input_frames += frames as u64;
                self.process_mono(&mono[..frames], &mut emit)?;
            }
        }
        self.status = Status::Active;
        Ok(())
    }
    fn process_mono(
        &mut self,
        samples: &[f32],
        emit: &mut impl FnMut(FrameOutput),
    ) -> Result<(), Error> {
        if let Some(converter) = &mut self.converter {
            converter.process(samples, &mut self.engine, emit)
        } else {
            for &sample in samples {
                self.engine.push(sample, emit)?;
            }
            Ok(())
        }
    }
    /// Drain resampler and centered-window lookahead with zero padding. Emits only
    /// frames whose centers lie before the end of the original audio. Idempotent.
    pub fn finish(&mut self, mut emit: impl FnMut(FrameOutput)) -> Result<(), Error> {
        match self.status {
            Status::Finished => return Ok(()),
            Status::Failed => return Err(Error::NeedsReset),
            Status::Active => {}
        }
        if self.channel != 0 {
            return Err(Error::IncompleteFrame);
        }
        // Count frame centers with integers, including at large source offsets.
        let frames = (u128::from(self.input_frames) * u128::from(features::FRAME_RATE))
            .div_ceil(u128::from(self.config.sample_rate)) as u64;
        self.status = Status::Failed;
        self.engine.frame_limit = Some(frames);
        while self.engine.frames < frames {
            self.process_mono(&[0.0], &mut emit)?;
        }
        self.status = Status::Finished;
        Ok(())
    }
    /// Reset a new stream at time zero.
    pub fn reset(&mut self) {
        self.discontinuity(0);
    }
    /// Reset after missing PCM. `source_frame` is the next frame's absolute
    /// position at the configured input rate (one frame contains all channels).
    pub fn discontinuity(&mut self, source_frame: u64) {
        self.engine.origin = source_frame as f64 / self.config.sample_rate as f64;
        self.engine.features.reset();
        self.engine.model.reset();
        self.engine
            .tracker
            .reset_at(self.engine.origin)
            .expect("sample-derived time is finite and nonnegative");
        self.engine.frames = 0;
        self.engine.frame_limit = None;
        if let Some(converter) = &mut self.converter {
            converter.reset();
        }
        self.channel = 0;
        self.sum = 0.;
        self.input_frames = 0;
        self.status = Status::Active;
    }
    /// The immutable settings used to construct this processor.
    pub fn config(&self) -> &BeatNetConfig {
        &self.config
    }
}
