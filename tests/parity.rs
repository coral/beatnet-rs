use beatnet_rs::{
    features::{FEATURES, FeatureExtractor},
    model::{Inference, Model, OnnxModel},
};
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
#[test]
fn frontend_matches_madmom() {
    for name in ["silence", "impulse", "tone", "noise", "rhythm"] {
        let pcm = floats(&std::fs::read(format!("tests/fixtures/{name}.pcm")).unwrap());
        let expected = floats(&std::fs::read(format!("tests/fixtures/{name}.features")).unwrap());
        let mut frontend = FeatureExtractor::new();
        let mut actual = Vec::new();
        for sample in pcm.iter().copied().chain(std::iter::repeat_n(0., 706)) {
            if let Some((position, features)) = frontend.push(sample)
                && position < pcm.len() as u64
            {
                actual.extend(features);
            }
        }
        assert_eq!(actual.len(), expected.len(), "{name}");
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 3e-6, "{name}: max difference {max}");
    }
}
#[test]
fn models_match_pytorch_and_reset() {
    let features = floats(include_bytes!("fixtures/rhythm.features"));
    for (i, model) in [Model::One, Model::Two, Model::Three]
        .into_iter()
        .enumerate()
    {
        let expected =
            floats(&std::fs::read(format!("tests/fixtures/model-{}.reference", i + 1)).unwrap());
        let mut backend = OnnxModel::bundled(model).unwrap();
        for (frame, row) in features
            .as_chunks::<FEATURES>()
            .0
            .iter()
            .zip(expected.as_chunks::<603>().0)
        {
            let p = backend.infer(frame).unwrap();
            let (h, c) = backend.recurrent_state();
            for (&actual, &reference) in p.iter().chain(h).chain(c).zip(row) {
                assert!(
                    (actual - reference).abs() < 5e-5 + reference.abs() * 5e-4,
                    "model {}, {actual} vs {reference}",
                    i + 1
                );
            }
        }
        backend.reset();
        assert!(
            backend
                .recurrent_state()
                .0
                .iter()
                .chain(backend.recurrent_state().1)
                .all(|&x| x == 0.)
        );
        let p = backend
            .infer(features[..FEATURES].try_into().unwrap())
            .unwrap();
        for (a, b) in p.iter().zip(&expected[..3]) {
            assert!((a - b).abs() < 5e-5);
        }
    }
}

#[test]
fn pcm_to_probabilities_matches_reference() {
    use beatnet_rs::{BeatNet, BeatNetConfig};
    let pcm = floats(include_bytes!("fixtures/rhythm.pcm"));
    let reference = floats(include_bytes!("fixtures/model-1.reference"));
    let mut net = BeatNet::new(BeatNetConfig::default()).unwrap();
    let mut output = Vec::new();
    for block in pcm.chunks(137) {
        net.process(block, |f| output.push(f)).unwrap();
    }
    net.finish(|f| output.push(f)).unwrap();
    assert_eq!(output.len(), reference.len() / 603);
    for (actual, expected) in output.iter().zip(reference.as_chunks::<603>().0) {
        for (a, b) in actual.state.probabilities.iter().zip(expected) {
            assert!((a - b).abs() < 5e-5);
        }
    }
    net.reset();
    let mut second = Vec::new();
    net.process(&pcm, |f| second.push(f)).unwrap();
    net.finish(|f| second.push(f)).unwrap();
    assert_eq!(output, second);
}

#[test]
fn incompatible_model_and_nonfinite_features_are_rejected() {
    use beatnet_rs::Error;
    let mut renamed = Model::One.bytes().to_vec();
    for offset in 0..renamed.len() - 8 {
        if &renamed[offset..offset + 8] == b"features" {
            renamed[offset..offset + 8].copy_from_slice(b"wrong___");
        }
    }
    assert!(matches!(
        OnnxModel::from_bytes(&renamed),
        Err(Error::ModelInterface("features"))
    ));
    let mut model = OnnxModel::bundled(Model::One).unwrap();
    assert!(matches!(
        model.infer(&[f32::NAN; FEATURES]),
        Err(Error::ModelInput)
    ));
    let (hidden, cell) = model.recurrent_state();
    assert!(hidden.iter().chain(cell).all(|&v| v == 0.0));
}
