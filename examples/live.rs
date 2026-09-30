//! Default input device -> callback queue -> synchronous worker -> terminal.
use beatnet_rs::{BeatNet, BeatNetConfig, Model, transport};
use cpal::{
    SampleFormat,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = match std::env::args().nth(1).as_deref() {
        None | Some("1") => Model::One,
        Some("2") => Model::Two,
        Some("3") => Model::Three,
        _ => return Err("usage: live [1|2|3]".into()),
    };
    let device = cpal::default_host()
        .default_input_device()
        .ok_or("no input device")?;
    let supported = device.default_input_config()?;
    let config: cpal::StreamConfig = supported.into();
    let sample_rate = config.sample_rate;
    let channels = config.channels;
    eprintln!("Input: {sample_rate} Hz, {channels} channels. Press Ctrl-C to stop.");
    let mut net = BeatNet::new(BeatNetConfig {
        sample_rate,
        channels,
        model,
        ..Default::default()
    })?;
    let (input, mut audio, stats) = transport::audio_queue(sample_rate, channels)?;
    let (mut outputs, mut output) = transport::output_queue(stats.clone());
    let running = Arc::new(AtomicBool::new(true));
    let signal = running.clone();
    ctrlc::set_handler(move || signal.store(false, Ordering::Relaxed))?;
    let capture = match supported.sample_format() {
        SampleFormat::F32 => {
            beatnet_rs::capture::build_input_stream::<f32>(&device, config, input)?
        }
        SampleFormat::F64 => {
            beatnet_rs::capture::build_input_stream::<f64>(&device, config, input)?
        }
        SampleFormat::I8 => beatnet_rs::capture::build_input_stream::<i8>(&device, config, input)?,
        SampleFormat::I16 => {
            beatnet_rs::capture::build_input_stream::<i16>(&device, config, input)?
        }
        SampleFormat::I32 => {
            beatnet_rs::capture::build_input_stream::<i32>(&device, config, input)?
        }
        SampleFormat::I64 => {
            beatnet_rs::capture::build_input_stream::<i64>(&device, config, input)?
        }
        SampleFormat::U8 => beatnet_rs::capture::build_input_stream::<u8>(&device, config, input)?,
        SampleFormat::U16 => {
            beatnet_rs::capture::build_input_stream::<u16>(&device, config, input)?
        }
        SampleFormat::U32 => {
            beatnet_rs::capture::build_input_stream::<u32>(&device, config, input)?
        }
        SampleFormat::U64 => {
            beatnet_rs::capture::build_input_stream::<u64>(&device, config, input)?
        }
        format => return Err(format!("unsupported input format: {format}").into()),
    };
    capture.play()?;
    let worker_running = running.clone();
    let worker = std::thread::Builder::new().name("beatnet".into()).spawn(
        move || -> Result<(), beatnet_rs::Error> {
            let mut expected = 0;
            while worker_running.load(Ordering::Relaxed) {
                if let Some(block) = audio.pop() {
                    if block.source_frame() != expected {
                        net.discontinuity(block.source_frame());
                    }
                    expected = block.source_frame() + block.frames() as u64;
                    if let Err(error) = net.process(block.samples(), |frame| outputs.push(frame)) {
                        worker_running.store(false, Ordering::Relaxed);
                        return Err(error);
                    }
                } else {
                    std::thread::sleep(Duration::from_micros(500));
                }
            }
            Ok(())
        },
    )?;
    while running.load(Ordering::Relaxed) {
        while let Some(frame) = output.pop() {
            if let Some(event) = frame.event {
                println!(
                    "{:.3}s {:>8} {:6.1} BPM confidence {:.2} meter {:?}",
                    event.time,
                    if event.downbeat { "downbeat" } else { "beat" },
                    event.bpm,
                    event.confidence,
                    frame.state.meter
                );
            }
        }
        if worker.is_finished() || stats.snapshot().device_errors > 0 {
            running.store(false, Ordering::Relaxed);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(capture);
    worker.join().map_err(|_| "worker panicked")??;
    eprintln!("{:?}", stats.snapshot());
    if stats.snapshot().device_errors > 0 {
        return Err("audio device reported an error; restart capture".into());
    }
    Ok(())
}
