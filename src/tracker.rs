//! Bounded Rust port of BeatNet's particle-filter cascade.
use crate::{BeatEvent, Error, TrackingState, features::FRAME_RATE, validate_probabilities};

const SECONDS_PER_FRAME: f64 = 1.0 / FRAME_RATE as f64;
const TEMPO_SCALE: f64 = 60.0 * FRAME_RATE as f64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct TrackerConfig {
    pub particles: usize,
    pub downbeat_particles: usize,
    pub min_bpm: f64,
    pub max_bpm: f64,
    pub min_meter: u8,
    pub max_meter: u8,
    pub seed: u64,
}
impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            particles: 1500,
            downbeat_particles: 250,
            min_bpm: 55.,
            max_bpm: 215.,
            min_meter: 2,
            max_meter: 4,
            seed: 1,
        }
    }
}
impl TrackerConfig {
    /// Check tempo, meter, and particle-count bounds before allocating state.
    pub fn validate(&self) -> Result<(), Error> {
        if !self.min_bpm.is_finite()
            || !self.max_bpm.is_finite()
            || self.min_bpm < 20.
            || self.max_bpm > 400.
            || self.min_bpm > self.max_bpm
            || !(1..=100_000).contains(&self.particles)
            || !(1..=100_000).contains(&self.downbeat_particles)
            || self.min_meter < 2
            || self.max_meter > 12
            || self.min_meter > self.max_meter
        {
            return Err(Error::Config("invalid tracker bounds"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct State {
    interval: usize,
    offset: usize,
    tempo: usize,
    sin: f32,
    cos: f32,
}
struct Space {
    states: Vec<State>,
    first: Vec<usize>,
    cdf: Vec<Vec<f64>>,
}
impl Space {
    fn new(min: usize, max: usize, beat: bool) -> Self {
        let mut states = Vec::new();
        let mut first = Vec::new();
        let mut cdf = Vec::new();
        for interval in min..=max {
            first.push(states.len());
            for offset in 0..interval {
                let angle = std::f32::consts::TAU * offset as f32 / interval as f32;
                states.push(State {
                    interval,
                    offset,
                    tempo: interval - min,
                    sin: angle.sin(),
                    cos: angle.cos(),
                });
            }
            let weights: Vec<f64> = (min..=max)
                .map(|to| {
                    if beat {
                        let p = (-60. * (to as f64 / interval as f64 - 1.).abs()).exp();
                        if p <= f64::EPSILON { 0. } else { p }
                    } else if min == max {
                        1.
                    } else if to == interval {
                        0.9
                    } else {
                        0.1 / (max - min) as f64
                    }
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let mut sum = 0.;
            let mut row: Vec<f64> = weights
                .iter()
                .map(|p| {
                    sum += p / total;
                    sum
                })
                .collect();
            *row.last_mut().unwrap() = 1.;
            cdf.push(row);
        }
        Self { states, first, cdf }
    }
    fn advance(&self, id: usize, rng: &mut ChaCha8Rng) -> usize {
        let s = self.states[id];
        if s.offset + 1 < s.interval {
            id + 1
        } else {
            let draw = rng.random::<f64>();
            let next = self.cdf[s.tempo].partition_point(|&p| p <= draw);
            self.first[next.min(self.first.len() - 1)]
        }
    }
}
struct Population {
    particles: Vec<usize>,
    candidates: Vec<usize>,
    weights: Vec<f64>,
    counts: Vec<usize>,
}
impl Population {
    fn new(n: usize, space: &Space) -> Self {
        Self {
            particles: vec![0; n],
            candidates: Vec::with_capacity(n + space.first.len()),
            weights: Vec::with_capacity(n + space.first.len()),
            counts: vec![0; space.states.len()],
        }
    }
    fn reset(&mut self, space: &Space, rng: &mut ChaCha8Rng) {
        for id in &mut self.particles {
            *id = rng.random_range(0..space.states.len());
        }
        self.candidates.clear();
        self.weights.clear();
        self.counts.fill(0);
    }
    fn median(&mut self) -> usize {
        self.counts.fill(0);
        for &id in &self.particles {
            self.counts[id] += 1;
        }
        let n = self.particles.len();
        let mut total = 0;
        let mut low = None;
        for (id, &count) in self.counts.iter().enumerate() {
            total += count;
            if low.is_none() && total > (n - 1) / 2 {
                low = Some(id);
            }
            if total > n / 2 {
                return (low.unwrap() + id) / 2;
            }
        }
        0
    }
    fn mode(&mut self) -> usize {
        self.counts.fill(0);
        for &id in &self.particles {
            self.counts[id] += 1;
        }
        let mut selected = 0;
        for id in 1..self.counts.len() {
            if self.counts[id] > self.counts[selected] {
                selected = id;
            }
        }
        selected
    }
    fn motion(&mut self, space: &Space, rng: &mut ChaCha8Rng) {
        self.candidates.clear();
        // Preserve upstream ordering: interior states, then boundary transitions.
        for &id in &self.particles {
            let s = space.states[id];
            if s.offset + 1 < s.interval {
                self.candidates.push(id + 1);
            }
        }
        for &id in &self.particles {
            let s = space.states[id];
            if s.offset + 1 == s.interval {
                self.candidates.push(space.advance(id, rng));
            }
        }
    }
    fn resample(&mut self, rng: &mut ChaCha8Rng) {
        self.resample_with(|| rng.random::<f64>());
    }
    fn resample_with(&mut self, mut draw: impl FnMut() -> f64) {
        let total: f64 = self.weights.iter().sum();
        let total = if !total.is_finite() || total <= 0.0 {
            self.weights.fill(1.0);
            self.weights.len() as f64
        } else {
            total
        };
        let n = self.particles.len();
        let mut index = 0;
        let mut cumulative = self.weights[0] / total;
        for j in 0..n {
            let point = (j as f64 + draw()) / n as f64;
            while point >= cumulative && index + 1 < self.candidates.len() {
                index += 1;
                cumulative += self.weights[index] / total;
            }
            self.particles[j] = self.candidates[index];
        }
    }
}
pub struct ParticleFilter {
    config: TrackerConfig,
    rng: ChaCha8Rng,
    beat_space: Space,
    down_space: Space,
    beats: Population,
    downs: Population,
    last_beat_time: f64,
    origin: f64,
    last_frame_time: Option<f64>,
    last_downbeat: bool,
    locked: bool,
}
impl ParticleFilter {
    pub fn new(config: TrackerConfig) -> Result<Self, Error> {
        config.validate()?;
        let beat_space = Space::new(
            (TEMPO_SCALE / config.max_bpm).round_ties_even() as usize,
            (TEMPO_SCALE / config.min_bpm).round_ties_even() as usize,
            true,
        );
        let down_space = Space::new(config.min_meter as usize, config.max_meter as usize, false);
        let beats = Population::new(config.particles, &beat_space);
        let downs = Population::new(config.downbeat_particles, &down_space);
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        let mut result = Self {
            config,
            rng,
            beat_space,
            down_space,
            beats,
            downs,
            last_beat_time: 0.,
            origin: 0.0,
            last_frame_time: None,
            last_downbeat: false,
            locked: false,
        };
        result.reset();
        Ok(result)
    }
    /// Reinitialize the seeded populations at time zero.
    pub fn reset(&mut self) {
        self.reset_at(0.0).expect("zero is a valid timestamp");
    }
    /// Reset after a gap or seek. Invalid time leaves the tracker untouched.
    pub fn reset_at(&mut self, time: f64) -> Result<(), Error> {
        if !time.is_finite() || time < 0.0 {
            return Err(Error::InvalidTime);
        }
        self.origin = time;
        self.last_frame_time = None;
        self.rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        self.beats.reset(&self.beat_space, &mut self.rng);
        self.downs.reset(&self.down_space, &mut self.rng);
        self.last_beat_time = time;
        self.last_downbeat = false;
        self.locked = false;
        Ok(())
    }
    /// Advance one 20 ms frame. Probabilities are [beat, downbeat, non-beat]
    /// and must sum to one (within 1e-4). Time is on the audio timeline and must
    /// increase; call `reset_at` before a gap or seek. Errors leave state untouched.
    pub fn process(
        &mut self,
        p: [f32; 3],
        time: f64,
    ) -> Result<(TrackingState, Option<BeatEvent>), Error> {
        validate_probabilities(&p)?;
        if !time.is_finite()
            || time < self.origin
            || self
                .last_frame_time
                .is_some_and(|previous| time <= previous)
        {
            return Err(Error::InvalidTime);
        }
        self.last_frame_time = Some(time);
        let activation = p[0].max(p[1]);
        let gated = if activation < 0.4 { 0.03 } else { activation };
        // Correct with the current observation before reporting its phase/event.
        self.beats.motion(&self.beat_space, &mut self.rng);
        if gated > 0.1 {
            if gated > 0.8 {
                let start = self.rng.random_range(0..4);
                self.beats
                    .candidates
                    .extend(self.beat_space.first.iter().skip(start).step_by(6).copied());
            }
            self.beats.weights.clear();
            self.beats
                .weights
                .extend(self.beats.candidates.iter().map(|&id| {
                    let s = self.beat_space.states[id];
                    if (s.offset as f64 / s.interval as f64) < 1. / 56. {
                        gated as f64
                    } else {
                        0.03
                    }
                }));
            self.beats.resample(&mut self.rng);
        } else {
            self.beats.particles.copy_from_slice(&self.beats.candidates);
        }
        let selected = self.beats.median();
        let state = self.beat_space.states[selected];
        let mut sin = 0.;
        let mut cos = 0.;
        for &id in &self.beats.particles {
            sin += self.beat_space.states[id].sin;
            cos += self.beat_space.states[id].cos;
        }
        let confidence = (sin.hypot(cos) / self.beats.particles.len() as f32).clamp(0., 1.);
        let mut event = None;
        if gated > 0.4
            && state.offset < 4
            && time - self.last_beat_time > 0.4 * SECONDS_PER_FRAME * state.interval as f64
        {
            self.downs.motion(&self.down_space, &mut self.rng);
            if p[1] > 0.7 {
                self.downs
                    .candidates
                    .extend_from_slice(&self.down_space.first);
            }
            self.downs.weights.clear();
            self.downs
                .weights
                .extend(self.downs.candidates.iter().map(|&id| {
                    if self.down_space.states[id].offset == 0 {
                        p[1] as f64
                    } else {
                        p[0] as f64
                    }
                }));
            self.downs.resample(&mut self.rng);
            let down_id = self.downs.mode();
            let downbeat =
                self.down_space.states[down_id].offset == 0 && !self.last_downbeat && p[1] > 0.4;
            self.last_beat_time = time;
            self.last_downbeat = downbeat;
            self.locked = true;
            event = Some(BeatEvent {
                time,
                bpm: (TEMPO_SCALE / state.interval as f64) as f32,
                downbeat,
                confidence,
            });
        }
        let meter = self.down_space.states[self.downs.mode()].interval as u8;
        let tracking = TrackingState {
            time,
            bpm: self
                .locked
                .then_some((TEMPO_SCALE / state.interval as f64) as f32),
            phase: self
                .locked
                .then_some(state.offset as f32 / state.interval as f32),
            meter: self.locked.then_some(meter),
            confidence,
            probabilities: p,
        };
        Ok((tracking, event))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silence_does_not_advance_meter_particles() {
        let mut tracker = ParticleFilter::new(TrackerConfig::default()).unwrap();
        let original = tracker.downs.particles.clone();
        for frame in 0..1000 {
            let (_, event) = tracker
                .process([0.0, 0.0, 1.0], frame as f64 * SECONDS_PER_FRAME)
                .unwrap();
            assert!(event.is_none());
        }
        assert_eq!(tracker.downs.particles, original);
    }

    #[test]
    fn zero_draw_never_selects_zero_probability() {
        let space = Space::new(2, 4, false);
        let mut population = Population::new(2, &space);
        population.candidates.extend([0, 1, 2, 3]);
        population.weights.extend([0.0, 0.5, 0.0, 0.5]);
        population.resample_with(|| 0.0);
        assert_eq!(population.particles, [1, 3]);
    }

    #[test]
    fn rejected_observations_do_not_mutate_tracker() {
        let mut tracker = ParticleFilter::new(TrackerConfig::default()).unwrap();
        let mut reference = ParticleFilter::new(TrackerConfig::default()).unwrap();
        tracker.process([0.0, 0.0, 1.0], 0.0).unwrap();
        reference.process([0.0, 0.0, 1.0], 0.0).unwrap();
        for probabilities in [[0.0; 3], [0.9; 3], [f32::NAN, 0.0, 1.0]] {
            assert!(matches!(
                tracker.process(probabilities, 0.02),
                Err(Error::ModelOutput)
            ));
        }
        for time in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                tracker.process([0.0, 0.0, 1.0], time),
                Err(Error::InvalidTime)
            ));
        }
        assert!(matches!(
            tracker.reset_at(f64::NAN),
            Err(Error::InvalidTime)
        ));
        assert_eq!(
            tracker.process([0.9, 0.05, 0.05], 0.02).unwrap(),
            reference.process([0.9, 0.05, 0.05], 0.02).unwrap()
        );
        assert_eq!(tracker.beats.particles, reference.beats.particles);
    }

    #[test]
    fn reference_state_space() {
        let space = Space::new(14, 55, true);
        let expected = include_bytes!("../tests/fixtures/state_intervals.f32");
        assert_eq!(space.states.len() * 4, expected.len());
        for (s, v) in space.states.iter().zip(expected.as_chunks::<4>().0) {
            assert_eq!(s.interval as f32, f32::from_le_bytes(*v));
        }
        let expected = include_bytes!("../tests/fixtures/transitions.f64");
        for (row, bytes) in space.cdf.iter().zip(expected.as_chunks::<{ 42 * 8 }>().0) {
            let mut previous = 0.;
            for (&value, b) in row.iter().zip(bytes.as_chunks::<8>().0) {
                assert!((value - previous - f64::from_le_bytes(*b)).abs() < 1e-12);
                previous = value;
            }
        }
    }
    #[test]
    fn shared_random_resampling_reference() {
        let space = Space::new(2, 4, false);
        let mut population = Population::new(17, &space);
        population.candidates.extend(0..6);
        population.weights.extend(
            include_bytes!("../tests/fixtures/resample_weights.f64")
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b)),
        );
        let mut draws = include_bytes!("../tests/fixtures/resample_draws.f64")
            .as_chunks::<8>()
            .0
            .iter();
        population.resample_with(|| f64::from_le_bytes(*draws.next().unwrap()));
        let expected: Vec<usize> = include_bytes!("../tests/fixtures/resample_indices.u32")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b) as usize)
            .collect();
        assert_eq!(population.particles, expected);
    }
    #[test]
    fn zero_weights_and_bounded_population() {
        let mut pf = ParticleFilter::new(TrackerConfig::default()).unwrap();
        for i in 0..10_000 {
            pf.process(
                if i % 25 == 0 {
                    [0.01, 0.98, 0.01]
                } else {
                    [0., 0., 1.]
                },
                i as f64 * 0.02,
            )
            .unwrap();
        }
        assert_eq!(pf.beats.particles.len(), 1500);
        assert_eq!(pf.downs.particles.len(), 250);
        assert!(pf.beats.candidates.len() <= 1542);
        pf.downs.weights.fill(0.);
        pf.downs.resample(&mut pf.rng);
        assert!(
            pf.downs
                .particles
                .iter()
                .all(|&id| id < pf.down_space.states.len())
        );
    }
}
