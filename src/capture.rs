//! Optional CPAL callback adapter. Device lifecycle stays on the application thread.
use crate::{Error, transport::AudioProducer};
use cpal::{FromSample, SizedSample, traits::DeviceTrait};

/// Build (but do not start) a CPAL input stream feeding a preallocated queue.
/// Rejects source rate/channel mismatches before opening the device. Error
/// callbacks increment the queue's own diagnostics without logging or blocking.
pub fn build_input_stream<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut input: AudioProducer,
) -> Result<cpal::Stream, Error>
where
    T: SizedSample + Copy,
    f32: FromSample<T>,
{
    input.validate_format(config.sample_rate, config.channels)?;
    let stats = input.diagnostics().clone();
    Ok(device.build_input_stream(
        config,
        move |samples: &[T], _: &_| input.push(samples, f32::from_sample_),
        move |_| stats.device_error(),
        None,
    )?)
}
