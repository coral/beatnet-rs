//! Bounded callback-to-worker transport. Payloads are `Copy` and have no destructors.
use crate::{Error, FrameOutput, validate_audio_format};
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// Maximum interleaved samples in one transport block.
pub const BLOCK_SAMPLES: usize = 1024;
const BLOCK_DURATION_MS: usize = 5;
const QUEUE_DURATION_MS: usize = 250;
const MAX_BACKLOG_MS: usize = 100;
const OUTPUT_CAPACITY: usize = 128;
#[derive(Clone, Copy, Debug)]
pub struct AudioBlock {
    source_frame: u64,
    frames: usize,
    channels: u16,
    samples: [f32; BLOCK_SAMPLES],
}
impl AudioBlock {
    /// Absolute input frame position; a frame contains one sample per channel.
    pub fn source_frame(&self) -> u64 {
        self.source_frame
    }
    /// Number of complete multichannel frames in this block.
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn channels(&self) -> u16 {
        self.channels
    }
    /// Only the initialized interleaved PCM; excludes unused buffer capacity.
    pub fn samples(&self) -> &[f32] {
        &self.samples[..self.frames * self.channels as usize]
    }
}

#[derive(Default)]
pub struct Diagnostics {
    dropped_input: AtomicU64,
    dropped_output: AtomicU64,
    discarded_stale: AtomicU64,
    device_errors: AtomicU64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub dropped_input_frames: u64,
    pub dropped_outputs: u64,
    pub discarded_stale_frames: u64,
    pub device_errors: u64,
}
impl Diagnostics {
    pub fn snapshot(&self) -> Counters {
        Counters {
            dropped_input_frames: self.dropped_input.load(Ordering::Relaxed),
            dropped_outputs: self.dropped_output.load(Ordering::Relaxed),
            discarded_stale_frames: self.discarded_stale.load(Ordering::Relaxed),
            device_errors: self.device_errors.load(Ordering::Relaxed),
        }
    }
    pub fn device_error(&self) {
        self.device_errors.fetch_add(1, Ordering::Relaxed);
    }
}
pub struct AudioProducer {
    producer: Producer<AudioBlock>,
    block: AudioBlock,
    used: usize,
    source: u64,
    frames_per_block: usize,
    sample_rate: u32,
    diagnostics: Arc<Diagnostics>,
}
pub struct AudioConsumer {
    consumer: Consumer<AudioBlock>,
    max_backlog_blocks: usize,
    frames_per_block: usize,
    diagnostics: Arc<Diagnostics>,
}
/// Allocate approximately 250 ms of PCM capacity and a 100 ms backlog limit.
pub fn audio_queue(
    rate: u32,
    channels: u16,
) -> Result<(AudioProducer, AudioConsumer, Arc<Diagnostics>), Error> {
    validate_audio_format(rate, channels)?;
    let frames_per_block =
        (rate as usize * BLOCK_DURATION_MS / 1000).clamp(1, BLOCK_SAMPLES / channels as usize);
    let capacity = (rate as usize * QUEUE_DURATION_MS).div_ceil(frames_per_block * 1000);
    let (producer, consumer) = RingBuffer::new(capacity);
    let diagnostics = Arc::new(Diagnostics::default());
    let block = AudioBlock {
        source_frame: 0,
        frames: frames_per_block,
        channels,
        samples: [0.; BLOCK_SAMPLES],
    };
    Ok((
        AudioProducer {
            producer,
            block,
            used: 0,
            source: 0,
            frames_per_block,
            sample_rate: rate,
            diagnostics: diagnostics.clone(),
        },
        AudioConsumer {
            consumer,
            frames_per_block,
            max_backlog_blocks: (rate as usize * MAX_BACKLOG_MS).div_ceil(frames_per_block * 1000),
            diagnostics: diagnostics.clone(),
        },
        diagnostics,
    ))
}
impl AudioProducer {
    /// Source format used to size this queue.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn channels(&self) -> u16 {
        self.block.channels
    }
    #[cfg(feature = "live")]
    pub(crate) fn diagnostics(&self) -> &Arc<Diagnostics> {
        &self.diagnostics
    }
    #[cfg(feature = "live")]
    pub(crate) fn validate_format(&self, rate: u32, channels: u16) -> Result<(), Error> {
        if rate != self.sample_rate || channels != self.block.channels {
            return Err(Error::Config("capture format must match the audio queue"));
        }
        Ok(())
    }

    /// Suitable for an audio callback: no allocation, locks, waiting, or logging.
    /// `convert` should be a cheap sample-format conversion.
    pub fn push<T: Copy>(&mut self, input: &[T], mut convert: impl FnMut(T) -> f32) {
        for &sample in input {
            self.block.samples[self.used] = convert(sample);
            self.used += 1;
            if self.used == self.frames_per_block * self.block.channels as usize {
                self.block.source_frame = self.source;
                if self.producer.push(self.block).is_err() {
                    self.diagnostics
                        .dropped_input
                        .fetch_add(self.frames_per_block as u64, Ordering::Relaxed);
                }
                self.source += self.frames_per_block as u64;
                self.used = 0;
            }
        }
    }
}
impl AudioConsumer {
    /// If the worker fell behind, discard stale PCM and keep the newest block
    /// visible at entry. Source positions allow the worker to detect the gap.
    pub fn pop(&mut self) -> Option<AudioBlock> {
        let queued = self.consumer.slots();
        if queued > self.max_backlog_blocks {
            for _ in 1..queued {
                if let Ok(block) = self.consumer.pop() {
                    self.diagnostics
                        .discarded_stale
                        .fetch_add(block.frames as u64, Ordering::Relaxed);
                }
            }
        }
        self.consumer.pop().ok()
    }
    /// A concurrent snapshot of queued input frames, excluding a partial block.
    pub fn queued_frames(&self) -> usize {
        self.consumer.slots() * self.frames_per_block
    }
}
pub struct OutputProducer {
    producer: Producer<FrameOutput>,
    diagnostics: Arc<Diagnostics>,
}
pub struct OutputConsumer {
    consumer: Consumer<FrameOutput>,
}
pub fn output_queue(diagnostics: Arc<Diagnostics>) -> (OutputProducer, OutputConsumer) {
    let (producer, consumer) = RingBuffer::new(OUTPUT_CAPACITY);
    (
        OutputProducer {
            producer,
            diagnostics,
        },
        OutputConsumer { consumer },
    )
}
impl OutputProducer {
    pub fn push(&mut self, output: FrameOutput) {
        if self.producer.push(output).is_err() {
            self.diagnostics
                .dropped_output
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}
impl OutputConsumer {
    pub fn pop(&mut self) -> Option<FrameOutput> {
        self.consumer.pop().ok()
    }
}

#[cfg(all(test, feature = "live"))]
mod tests {
    use super::*;
    #[test]
    fn capture_format_must_match_queue() {
        let (producer, _, _) = audio_queue(48000, 2).unwrap();
        assert!(producer.validate_format(48000, 2).is_ok());
        assert!(matches!(
            producer.validate_format(44100, 2),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            producer.validate_format(48000, 1),
            Err(Error::Config(_))
        ));
    }
}
