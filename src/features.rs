//! Exact BeatNet/madmom frontend, with centered frames and bounded lookahead.
use fearless_simd::{Level, dispatch, prelude::*};
use fearless_simd_macros::simd;
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};
use std::{ops::Range, sync::Arc};

pub const SAMPLE_RATE: u32 = 22_050;
pub const HOP: usize = 441;
/// Number of model/tracker updates per second.
pub const FRAME_RATE: u32 = SAMPLE_RATE / HOP as u32;
pub const WINDOW: usize = 1411;
pub const FEATURES: usize = 272;
const BANDS: usize = FEATURES / 2;
const BINS: usize = WINDOW / 2; // madmom excludes the final real FFT bin, even for odd lengths.

struct FilterBand {
    start_bin: usize,
    weights: Range<usize>,
}

pub struct FeatureExtractor {
    fft: Arc<dyn RealToComplex<f64>>,
    input: Vec<f64>,
    spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    window: Vec<f64>,
    filters: [FilterBand; BANDS],
    weights: Vec<f32>,
    simd: Level,
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
        let mut weights = Vec::new();
        let filters = std::array::from_fn(|band| {
            let start_bin = (0..BINS)
                .find(|&bin| dense[bin * BANDS + band] != 0.)
                .expect("nonempty filter band");
            let end_bin = (start_bin..BINS)
                .rfind(|&bin| dense[bin * BANDS + band] != 0.)
                .unwrap()
                + 1;
            let start = weights.len();
            for bin in start_bin..end_bin {
                let weight = dense[bin * BANDS + band];
                assert_ne!(weight, 0., "filter support must be contiguous");
                weights.push(weight);
            }
            FilterBand {
                start_bin,
                weights: start..weights.len(),
            }
        });
        Self {
            input: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window,
            filters,
            weights,
            simd: Level::new(),
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
        prepare_window(&self.ring, self.write, &self.window, &mut self.input);
        self.fft
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
            .expect("fixed FFT buffers");
        dispatch!(self.simd, simd => magnitudes(simd, &self.spectrum[..BINS], &mut self.magnitudes));
        let mut output = [0.; FEATURES];
        for band in 0..BANDS {
            let filter = &self.filters[band];
            let weights = &self.weights[filter.weights.clone()];
            let bins = &self.magnitudes[filter.start_bin..filter.start_bin + weights.len()];
            let magnitude: f32 = bins
                .iter()
                .zip(weights)
                .map(|(magnitude, weight)| magnitude * weight)
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

fn prepare_window(ring: &[f32; WINDOW], write: usize, window: &[f64], output: &mut [f64]) {
    let split = WINDOW - write;
    // Two contiguous loops allow autovectorization without a modulo per sample.
    for ((out, &sample), &weight) in output[..split]
        .iter_mut()
        .zip(&ring[write..])
        .zip(&window[..split])
    {
        *out = f64::from(sample) * weight;
    }
    for ((out, &sample), &weight) in output[split..]
        .iter_mut()
        .zip(&ring[..write])
        .zip(&window[split..])
    {
        *out = f64::from(sample) * weight;
    }
}

#[simd]
fn magnitudes<S: Simd>(simd: S, spectrum: &[Complex<f64>], output: &mut [f32]) {
    let lanes = S::f64s::LEN;
    let mut input = spectrum.chunks_exact(lanes);
    let mut output = output.chunks_exact_mut(lanes);
    for (bins, out) in input.by_ref().zip(output.by_ref()) {
        // Preserve madmom's complex64 rounding before the magnitude. Squaring
        // in f64 avoids overflow/underflow for the full finite f32 range.
        let re = S::f64s::from_fn(simd, |i| f64::from(bins[i].re as f32));
        let im = S::f64s::from_fn(simd, |i| f64::from(bins[i].im as f32));
        let magnitude = (re * re + im * im).sqrt();
        // Like hypot, infinity wins over NaN when either component is infinite.
        let infinite = re.abs().simd_eq(f64::INFINITY) | im.abs().simd_eq(f64::INFINITY);
        let magnitude = infinite.select(S::f64s::splat(simd, f64::INFINITY), magnitude);
        for (out, &value) in out.iter_mut().zip(magnitude.as_slice()) {
            *out = value as f32;
        }
    }
    for (out, c) in output.into_remainder().iter_mut().zip(input.remainder()) {
        *out = (c.re as f32).hypot(c.im as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn window_matches_circular_indexing_at_every_offset() {
        let frontend = FeatureExtractor::new();
        let ring = std::array::from_fn(|i| (i as f32 * 0.031).sin());
        let mut output = [0.; WINDOW];
        for write in 0..WINDOW {
            prepare_window(&ring, write, &frontend.window, &mut output);
            for (i, &value) in output.iter().enumerate() {
                assert_eq!(
                    value,
                    f64::from(ring[(write + i) % WINDOW]) * frontend.window[i]
                );
            }
        }
    }

    fn available_levels() -> Vec<Level> {
        let mut levels = vec![Level::baseline()];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let detected = Level::new();
            levels.extend(detected.as_sse4_2().map(Level::Sse4_2));
            levels.extend(detected.as_avx2().map(Level::Avx2));
            levels.extend(detected.as_avx512().map(Level::Avx512));
        }
        levels.push(Level::new());
        levels
    }

    #[test]
    fn magnitudes_match_hypot_across_backends_ranges_and_tails() {
        let values = [
            0.,
            -0.,
            1.,
            -3.,
            f64::from(f32::from_bits(1)),
            f64::from(f32::MIN_POSITIVE),
            f64::from(f32::MAX),
            f64::from(f32::MIN),
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let mut spectrum: Vec<_> = values
            .iter()
            .flat_map(|&re| values.iter().map(move |&im| Complex::new(re, im)))
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        spectrum.extend((0..4096).map(|_| {
            Complex::new(
                f64::from(f32::from_bits(rng.random())),
                f64::from(f32::from_bits(rng.random())),
            )
        }));
        for level in available_levels() {
            // Exercise empty, partial-vector, and odd-sized inputs. Offset the
            // destination to ensure vector loads/stores need no alignment.
            for len in (0..=19).chain([BINS, spectrum.len()]) {
                let mut output = vec![-1.; len + 2];
                dispatch!(level, simd => magnitudes(simd, &spectrum[..len], &mut output[1..len + 1]));
                assert_eq!(output[0], -1.);
                assert_eq!(output[len + 1], -1.);
                for (c, &actual) in spectrum[..len].iter().zip(&output[1..len + 1]) {
                    let expected = (c.re as f32).hypot(c.im as f32);
                    if expected.is_nan() {
                        assert!(actual.is_nan(), "{level:?}: {c:?}");
                    } else if expected.is_infinite() {
                        assert_eq!(actual, expected, "{level:?}: {c:?}");
                    } else {
                        assert!(actual.is_finite());
                        assert!(
                            actual.to_bits().abs_diff(expected.to_bits()) <= 1,
                            "{level:?}: {c:?}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }
}
