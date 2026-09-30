use beatnet_rs::{
    BeatNet, BeatNetConfig, Error, FrameOutput, Inference,
    features::FEATURES,
    tracker::{ParticleFilter, TrackerConfig},
    transport,
};
#[derive(Default)]
struct TestModel;
impl Inference for TestModel {
    fn infer(&mut self, f: &[f32; FEATURES]) -> Result<[f32; 3], Error> {
        let p = (f.iter().sum::<f32>() / 100.).clamp(0., 1.);
        Ok([p, 0., 1. - p])
    }
    fn reset(&mut self) {}
}
fn run(rate: u32, channels: u16, chunk: usize) -> Vec<FrameOutput> {
    let mut net = BeatNet::with_model(
        BeatNetConfig {
            sample_rate: rate,
            channels,
            ..Default::default()
        },
        TestModel,
    )
    .unwrap();
    let pcm: Vec<f32> = (0..rate as usize * 2 * channels as usize)
        .map(|i| {
            let time = (i / channels as usize) as f32 / rate as f32;
            (time * std::f32::consts::TAU * 220.).sin() * 0.4
        })
        .collect();
    let mut out = Vec::new();
    for block in pcm.chunks(chunk) {
        net.process(block, |f| out.push(f)).unwrap();
    }
    net.finish(|f| out.push(f)).unwrap();
    assert_eq!(out.len(), 100);
    for (i, f) in out.iter().enumerate() {
        assert!((f.state.time - i as f64 * 0.02).abs() < 1e-10);
    }
    out
}
#[test]
fn arbitrary_chunks_and_rates() {
    for rate in [22050, 44100, 48000] {
        for channels in [1, 2] {
            let expected = run(rate, channels, 4096);
            for chunk in [1, 137, 441, 1023] {
                assert_eq!(
                    expected,
                    run(rate, channels, chunk),
                    "{rate} {channels} {chunk}"
                );
            }
        }
    }
}
#[test]
fn silence_reset_gap_and_bad_input() {
    let mut net = BeatNet::with_model(BeatNetConfig::default(), TestModel).unwrap();
    let mut output = Vec::new();
    net.process(&[0.; 44100], |f| output.push(f)).unwrap();
    assert!(
        output
            .iter()
            .all(|f| f.event.is_none() && f.state.bpm.is_none())
    );
    assert!(matches!(
        net.process(&[f32::NAN], |_| {}),
        Err(Error::InvalidPcm)
    ));
    net.discontinuity(220500);
    let mut after = Vec::new();
    net.process(&[0.; 2205], |f| after.push(f)).unwrap();
    assert_eq!(after[0].state.time, 10.);
    net.reset();
    let mut again = Vec::new();
    net.process(&[0.; 44100], |f| again.push(f)).unwrap();
    assert_eq!(again, output);
    net.finish(|_| {}).unwrap();
    net.finish(|_| panic!("idempotent")).unwrap();
    assert!(matches!(net.process(&[], |_| {}), Err(Error::Finished)));
}
#[test]
fn transport_overflow_reports_gaps_and_stays_bounded() {
    let (mut producer, mut consumer, stats) = transport::audio_queue(48000, 2).unwrap();
    let data = [0f32; 480];
    for _ in 0..100 {
        producer.push(&data, |x| x);
    }
    assert!(stats.snapshot().dropped_input_frames > 0);
    let first = consumer.pop().unwrap();
    assert!(first.source_frame() > 0);
    assert!(stats.snapshot().discarded_stale_frames > 0);
    assert!(consumer.pop().is_none());
    producer.push(&data, |x| x);
    assert_eq!(consumer.pop().unwrap().source_frame(), 24000);
}
#[test]
fn synthetic_activations_track_tempo_and_meters() {
    for meter in [2, 3, 4] {
        let mut pf = ParticleFilter::new(TrackerConfig {
            min_bpm: 90.,
            max_bpm: 160.,
            min_meter: meter,
            max_meter: meter,
            ..Default::default()
        })
        .unwrap();
        let mut events = Vec::new();
        for i in 0..3000 {
            let p = if i % 25 == 0 {
                if (i / 25) % meter as usize == 0 {
                    [0.01, 0.98, 0.01]
                } else {
                    [0.98, 0.01, 0.01]
                }
            } else {
                [0.005, 0.005, 0.99]
            };
            let (state, event) = pf.process(p, i as f64 * 0.02).unwrap();
            if i > 1500
                && let Some(event) = event
            {
                events.push(event);
            }
            if state.bpm.is_some() {
                assert_eq!(state.meter, Some(meter));
                assert!((0. ..1.).contains(&state.phase.unwrap()));
            }
        }
        assert!(events.len() > 40, "meter {meter}: {} events", events.len());
        assert!(events.iter().filter(|e| e.downbeat).count() > 5);
        let accurate = events.iter().filter(|e| (e.bpm - 120.).abs() < 6.).count();
        assert!(
            accurate as f32 / events.len() as f32 > 0.8,
            "meter {meter}: accurate {accurate}/{}; bpms {:?}",
            events.len(),
            events.iter().map(|e| e.bpm).collect::<Vec<_>>()
        );
    }
}

#[test]
fn resampler_delay_and_centered_timestamps() {
    for rate in [8000, 11025, 22050, 22051, 44100, 48000, 96000, 192000] {
        let mut net = BeatNet::with_model(
            BeatNetConfig {
                sample_rate: rate,
                ..Default::default()
            },
            TestModel,
        )
        .unwrap();
        let mut pcm = vec![0f32; rate as usize];
        pcm[rate as usize / 2] = 1.;
        let mut outputs = Vec::new();
        net.process(&pcm, |f| outputs.push(f)).unwrap();
        net.finish(|f| outputs.push(f)).unwrap();
        assert_eq!(outputs.len(), 50, "rate {rate}");
        let peak = outputs
            .iter()
            .max_by(|a, b| a.state.probabilities[0].total_cmp(&b.state.probabilities[0]))
            .unwrap();
        assert!(
            (peak.state.time - 0.5).abs() < 0.001,
            "rate {rate}, peak {}",
            peak.state.time
        );
    }
}
#[test]
fn changing_tempo_and_missing_observations() {
    let mut tracker = ParticleFilter::new(TrackerConfig {
        min_bpm: 90.,
        max_bpm: 180.,
        ..Default::default()
    })
    .unwrap();
    let mut bpms = Vec::new();
    let mut events = Vec::new();
    for i in 0..4000 {
        let period = if i < 2000 { 25 } else { 20 };
        let p = if i % period == 0 {
            [0.95, 0.02, 0.03]
        } else {
            [0.01, 0.01, 0.98]
        };
        let (state, event) = tracker.process(p, i as f64 * 0.02).unwrap();
        if i > 3000
            && let Some(bpm) = state.bpm
        {
            bpms.push(bpm);
        }
        if let Some(event) = event {
            events.push(event);
        }
    }
    let accurate = bpms.iter().filter(|&&bpm| (bpm - 150.).abs() < 8.).count();
    assert!(accurate as f64 / bpms.len() as f64 > 0.8);
    assert!(events.windows(2).all(|e| e[1].time > e[0].time));
    assert!(tracker.process([f32::NAN, 0., 1.], 81.).is_err());
}
#[test]
fn output_overflow_does_not_block() {
    let (_, _, stats) = transport::audio_queue(48000, 1).unwrap();
    let (mut producer, mut consumer) = transport::output_queue(stats.clone());
    let sample = run(22050, 1, 4096)[0];
    for _ in 0..1000 {
        producer.push(sample);
    }
    assert_eq!(stats.snapshot().dropped_outputs, 872);
    let mut count = 0;
    while consumer.pop().is_some() {
        count += 1;
    }
    assert_eq!(count, 128);
}

struct CountingModel {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    fail_on: Option<usize>,
}
impl Inference for CountingModel {
    fn infer(&mut self, _: &[f32; FEATURES]) -> Result<[f32; 3], Error> {
        let call = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if self.fail_on == Some(call) {
            self.fail_on = None;
            return Err(Error::ModelOutput);
        }
        Ok([0.0, 0.0, 1.0])
    }
    fn reset(&mut self) {
        self.calls.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[test]
fn finish_infers_exactly_the_audio_tail() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    for rate in [22050, 22051, 44100, 48000] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut net = BeatNet::with_model(
            BeatNetConfig {
                sample_rate: rate,
                ..Default::default()
            },
            CountingModel {
                calls: calls.clone(),
                fail_on: None,
            },
        )
        .unwrap();
        for length in [
            0,
            1,
            rate as usize / 50 - 1,
            rate as usize / 50,
            rate as usize / 50 + 1,
            rate as usize + 1,
        ] {
            // Large offsets must not affect the number of frames flushed.
            net.discontinuity(1_000_000_000_000);
            let expected = (length as u64 * 50).div_ceil(rate as u64) as usize;
            let mut emitted = 0;
            net.process(&vec![0.0; length], |_| emitted += 1).unwrap();
            net.finish(|_| emitted += 1).unwrap();
            net.finish(|_| panic!("finished twice")).unwrap();
            assert_eq!(emitted, expected, "{rate} Hz, {length} samples");
            assert_eq!(
                calls.load(Ordering::Relaxed),
                expected,
                "inferred beyond audio end"
            );
        }
    }
}

#[test]
fn backend_errors_require_reset_in_process_and_finish() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    for fail_during_finish in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut net = BeatNet::with_model(
            BeatNetConfig::default(),
            CountingModel {
                calls: calls.clone(),
                fail_on: Some(1),
            },
        )
        .unwrap();
        if fail_during_finish {
            net.process(&[0.0], |_| {}).unwrap();
            assert!(matches!(net.finish(|_| {}), Err(Error::ModelOutput)));
        } else {
            assert!(matches!(
                net.process(&[0.0; 2000], |_| {}),
                Err(Error::ModelOutput)
            ));
        }
        assert!(matches!(
            net.process(&[0.0; 2000], |_| {}),
            Err(Error::NeedsReset)
        ));
        assert!(matches!(net.finish(|_| {}), Err(Error::NeedsReset)));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        net.reset();
        net.process(&[0.0], |_| {}).unwrap();
        net.finish(|_| {}).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn partial_channel_and_invalid_input_are_recoverable() {
    let mut net = BeatNet::with_model(
        BeatNetConfig {
            channels: 2,
            ..Default::default()
        },
        TestModel,
    )
    .unwrap();
    net.process(&[0.5], |_| {}).unwrap();
    assert!(matches!(net.finish(|_| {}), Err(Error::IncompleteFrame)));
    assert!(matches!(
        net.process(&[0.5, f32::INFINITY], |_| {}),
        Err(Error::InvalidPcm)
    ));
    net.process(&[0.5], |_| {}).unwrap();
    let mut output = Vec::new();
    net.finish(|f| output.push(f)).unwrap();
    assert_eq!(output.len(), 1);
}

#[test]
fn invalid_config_does_not_reset_the_supplied_backend() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let calls = Arc::new(AtomicUsize::new(37));
    let result = BeatNet::with_model(
        BeatNetConfig {
            channels: 0,
            ..Default::default()
        },
        CountingModel {
            calls: calls.clone(),
            fail_on: None,
        },
    );
    assert!(matches!(result, Err(Error::Config(_))));
    assert_eq!(calls.load(Ordering::Relaxed), 37);
    assert!(matches!(
        BeatNet::new(BeatNetConfig {
            sample_rate: 0,
            ..Default::default()
        }),
        Err(Error::Config(_))
    ));
}

#[test]
fn caught_output_callback_panic_requires_reset() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for during_finish in [false, true] {
        let mut net = BeatNet::with_model(BeatNetConfig::default(), TestModel).unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            if during_finish {
                net.process(&[0.0], |_| {}).unwrap();
                let _ = net.finish(|_| panic!("caller callback"));
            } else {
                let _ = net.process(&[0.0; 2000], |_| panic!("caller callback"));
            }
        }));
        assert!(result.is_err());
        assert!(matches!(net.process(&[], |_| {}), Err(Error::NeedsReset)));
        assert!(matches!(net.finish(|_| {}), Err(Error::NeedsReset)));
        net.reset();
        net.process(&[0.0], |_| {}).unwrap();
        net.finish(|_| {}).unwrap();
    }
}
