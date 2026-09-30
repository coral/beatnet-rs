//! Exact BeatNet/madmom frontend, with centered frames and bounded lookahead.
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};
use std::sync::Arc;

pub const SAMPLE_RATE: u32 = 22_050;
pub const HOP: usize = 441;
/// Number of model/tracker updates per second.
pub const FRAME_RATE: u32 = SAMPLE_RATE / HOP as u32;
pub const WINDOW: usize = 1411;
pub const FEATURES: usize = 272;
const BANDS: usize = FEATURES / 2;
const BINS: usize = WINDOW / 2; // madmom excludes the final real FFT bin, even for odd lengths.

pub struct FeatureExtractor {
    fft: Arc<dyn RealToComplex<f64>>,
    input: Vec<f64>,
    spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    window: Vec<f64>,
    filters: Vec<Vec<(usize, f32)>>,
    ring: [f32; WINDOW],
    write: usize,
    received: u64,
    next_end: u64,
    frame: u64,
    previous: [f32; BANDS],
    magnitudes: [f32; BINS],
}
impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}
impl FeatureExtractor {
    pub fn new() -> Self {
        let fft = RealFftPlanner::new().plan_fft_forward(WINDOW);
        let window = include_bytes!("../assets/window.f64")
            .chunks_exact(8)
            .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let dense: Vec<f32> = include_bytes!("../assets/filterbank.f32")
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(dense.len(), BINS * BANDS);
        let filters = (0..BANDS)
            .map(|band| {
                (0..BINS)
                    .filter_map(|bin| {
                        let w = dense[bin * BANDS + band];
                        (w != 0.).then_some((bin, w))
                    })
                    .collect()
            })
            .collect();
        Self {
            input: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window,
            filters,
            ring: [0.; WINDOW],
            write: WINDOW / 2,
            received: 0,
            next_end: (WINDOW - WINDOW / 2) as u64,
            frame: 0,
            previous: [0.; BANDS],
            magnitudes: [0.; BINS],
        }
    }
    pub fn reset(&mut self) {
        self.ring.fill(0.);
        self.write = WINDOW / 2;
        self.received = 0;
        self.next_end = (WINDOW - WINDOW / 2) as u64;
        self.frame = 0;
        self.previous.fill(0.);
    }
    /// Push one mono sample. Frame timestamps are sample positions at 22,050 Hz.
    /// The first frame (time zero) is available after 706 samples.
    pub fn push(&mut self, sample: f32) -> Option<(u64, [f32; FEATURES])> {
        self.ring[self.write] = sample;
        self.write = (self.write + 1) % WINDOW;
        self.received += 1;
        if self.received < self.next_end {
            return None;
        }
        for i in 0..WINDOW {
            self.input[i] = self.ring[(self.write + i) % WINDOW] as f64 * self.window[i];
        }
        self.fft
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
            .expect("fixed FFT buffers");
        for (out, c) in self.magnitudes.iter_mut().zip(&self.spectrum) {
            // madmom stores complex64 before computing the magnitude.
            *out = (c.re as f32).hypot(c.im as f32);
        }
        let mut output = [0.; FEATURES];
        for band in 0..BANDS {
            let magnitude: f32 = self.filters[band]
                .iter()
                .map(|&(bin, w)| self.magnitudes[bin] * w)
                .sum();
            let value = (1. + magnitude).log10();
            output[band] = value;
            output[BANDS + band] = if self.frame == 0 {
                0.
            } else {
                (value - self.previous[band]).max(0.)
            };
            self.previous[band] = value;
        }
        let time = self.frame * HOP as u64;
        self.frame += 1;
        self.next_end += HOP as u64;
        Some((time, output))
    }
}
