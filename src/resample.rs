//! Preallocated Rubato rate conversion, owned by the synchronous processor.
use crate::{
    Engine, Error, FrameOutput, Inference,
    features::{HOP, SAMPLE_RATE},
};
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, Fft, FixedAsync, FixedSync, Resampler, SincInterpolationParameters};

pub(crate) struct RateConverter {
    resampler: Box<dyn Resampler<f32>>,
    input: Vec<f32>,
    output: Vec<f32>,
    used: usize,
    needed: usize,
    skip: usize,
}
impl RateConverter {
    pub(crate) fn new(rate: u32) -> Result<Self, Error> {
        // Fixed 441-sample output keeps frontend/inference scheduling predictable.
        let (mut a, mut b) = (rate, SAMPLE_RATE);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let quantum = SAMPLE_RATE / a;
        let resampler: Box<dyn Resampler<f32>> = if (HOP as u32).is_multiple_of(quantum) {
            Box::new(Fft::new(
                rate as usize,
                SAMPLE_RATE as usize,
                HOP,
                1,
                FixedSync::Output,
            )?)
        } else {
            // Rubato's "Async" means adjustable sample ratio, not async execution.
            // Avoid giant FFT blocks for coprime/nonstandard hardware sample rates.
            Box::new(Async::new_sinc(
                SAMPLE_RATE as f64 / rate as f64,
                1.0,
                &SincInterpolationParameters::default(),
                HOP,
                1,
                FixedAsync::Output,
            )?)
        };
        let skip = resampler.output_delay();
        Ok(Self {
            input: vec![0.; resampler.input_frames_max()],
            output: vec![0.; resampler.output_frames_max()],
            needed: resampler.input_frames_next(),
            resampler,
            used: 0,
            skip,
        })
    }
    pub(crate) fn reset(&mut self) {
        self.resampler.reset();
        self.used = 0;
        self.needed = self.resampler.input_frames_next();
        self.skip = self.resampler.output_delay();
        self.input.fill(0.);
        self.output.fill(0.);
    }
    pub(crate) fn process<M: Inference>(
        &mut self,
        mut samples: &[f32],
        engine: &mut Engine<M>,
        emit: &mut impl FnMut(FrameOutput),
    ) -> Result<(), Error> {
        while !samples.is_empty() || self.used == self.needed {
            let count = samples.len().min(self.needed - self.used);
            self.input[self.used..self.used + count].copy_from_slice(&samples[..count]);
            self.used += count;
            samples = &samples[count..];
            if self.used < self.needed {
                break;
            }
            let input = InterleavedSlice::new(&self.input, 1, self.used).expect("allocated input");
            let len = self.output.len();
            let mut output =
                InterleavedSlice::new_mut(&mut self.output, 1, len).expect("allocated output");
            let (_, produced) = self
                .resampler
                .process_into_buffer(&input, &mut output, None)?;
            self.used = 0;
            // Rubato may change the input requirement after each output block.
            self.needed = self.resampler.input_frames_next();
            let skip = self.skip.min(produced);
            self.skip -= skip;
            for &value in &self.output[skip..produced] {
                engine.push(value, emit)?;
            }
        }
        Ok(())
    }
}
