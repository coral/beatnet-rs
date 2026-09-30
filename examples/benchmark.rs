//! Run with --release. Measures processing time, not centered-window or device latency.
use beatnet_rs::{
    BeatNet, BeatNetConfig,
    features::FeatureExtractor,
    model::{Inference, Model, OnnxModel},
    tracker::{ParticleFilter, TrackerConfig},
    transport,
};
use std::{hint::black_box, time::Instant};
fn report(name: &str, mut times: Vec<f64>, audio_seconds: f64) {
    let total: f64 = times.iter().sum();
    times.sort_by(f64::total_cmp);
    let percentile = |p: f64| times[((times.len() - 1) as f64 * p) as usize];
    println!(
        "{name}: p50={:.3} p95={:.3} p99={:.3} max={:.3} ms; {:.1}x realtime",
        percentile(0.5),
        percentile(0.95),
        percentile(0.99),
        times.last().unwrap(),
        audio_seconds * 1000. / total
    );
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hops = std::env::args()
        .nth(1)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(3000usize);
    if hops < 100 {
        return Err("use at least 100 hops".into());
    }
    let pcm: Vec<f32> = include_bytes!("../tests/fixtures/rhythm.pcm")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut frontend = FeatureExtractor::new();
    let mut features = [0.; 272];
    let mut times = Vec::with_capacity(hops);
    for i in 0..hops {
        let start = Instant::now();
        for &v in &pcm[(i % 200) * 441..(i % 200 + 1) * 441] {
            if let Some((_, f)) = frontend.push(v) {
                features = f;
            }
        }
        times.push(start.elapsed().as_secs_f64() * 1000.);
    }
    report("frontend / hop", times, hops as f64 * 0.02);
    for model in [Model::One, Model::Two, Model::Three] {
        let mut backend = OnnxModel::bundled(model)?;
        let mut times = Vec::with_capacity(hops);
        for _ in 0..hops {
            let start = Instant::now();
            black_box(backend.infer(&features)?);
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        report(&format!("inference {model:?}"), times, hops as f64 * 0.02);
    }
    let mut tracker = ParticleFilter::new(TrackerConfig::default())?;
    let mut times = Vec::with_capacity(hops);
    for i in 0..hops {
        let start = Instant::now();
        black_box(tracker.process(
            if i % 25 == 0 {
                [0.01, 0.98, 0.01]
            } else {
                [0.01, 0.01, 0.98]
            },
            i as f64 * 0.02,
        )?);
        times.push(start.elapsed().as_secs_f64() * 1000.);
    }
    report("tracker / hop", times, hops as f64 * 0.02);
    for rate in [22050, 44100, 48000] {
        let hop = rate as usize / 50;
        let data: Vec<f32> = (0..hop).map(|i| (i as f32 * 0.1).sin() * 0.4).collect();
        let mut net = BeatNet::new(BeatNetConfig {
            sample_rate: rate,
            ..Default::default()
        })?;
        let mut times = Vec::with_capacity(hops);
        for _ in 0..hops {
            let start = Instant::now();
            net.process(&data, |f| {
                black_box(f);
            })?;
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        report(&format!("pipeline {rate}Hz"), times, hops as f64 * 0.02);
    }
    // Realtime-paced queue/worker soak (10 seconds); terminal output stays on the caller.
    let (mut producer, mut consumer, stats) = transport::audio_queue(48000, 1)?;
    let worker = std::thread::spawn(move || -> Result<(f64, f64), beatnet_rs::Error> {
        let mut net = BeatNet::new(BeatNetConfig {
            sample_rate: 48000,
            ..Default::default()
        })?;
        let mut expected = 0;
        let mut max_age = 0f64;
        let mut max_processing = 0f64;
        let deadline = Instant::now() + std::time::Duration::from_secs(11);
        while Instant::now() < deadline {
            max_age = max_age.max(consumer.queued_frames() as f64 / 48.);
            if let Some(block) = consumer.pop() {
                if block.source_frame() != expected {
                    net.discontinuity(block.source_frame());
                }
                expected = block.source_frame() + block.frames() as u64;
                let start = Instant::now();
                net.process(block.samples(), |f| {
                    black_box(f);
                })?;
                max_processing = max_processing.max(start.elapsed().as_secs_f64() * 1000.);
            } else {
                std::thread::sleep(std::time::Duration::from_micros(500));
            }
        }
        Ok((max_age, max_processing))
    });
    let block = [0.1f32; 240];
    let start = Instant::now();
    for i in 0..2000 {
        producer.push(&block, |x| x);
        let next = start + std::time::Duration::from_millis((i + 1) * 5);
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    let (age, processing) = worker.join().unwrap()?;
    println!(
        "queue soak: max backlog={age:.2} ms, max block processing={processing:.3} ms, {:?}",
        stats.snapshot()
    );
    Ok(())
}
