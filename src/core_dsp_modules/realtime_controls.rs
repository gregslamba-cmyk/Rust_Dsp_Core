use num_traits::Float;

#[derive(Clone, Copy, Debug)]
pub struct EnvelopeFollower<T: Float> {
    sample_rate: T,
    attack_coefficient: T,
    release_coefficient: T,
    envelope: T,
}

impl<T: Float> Default for EnvelopeFollower<T> {
    fn default() -> Self {
        Self {
            sample_rate: T::zero(),
            attack_coefficient: T::zero(),
            release_coefficient: T::zero(),
            envelope: T::zero(),
        }
    }
}

impl<T: Float> EnvelopeFollower<T> {
    pub fn initialize(&mut self, sample_rate: T, attack_seconds: T, release_seconds: T) -> bool {
        if !sample_rate.is_finite() || sample_rate <= T::zero() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.set_times(attack_seconds, release_seconds)
    }

    pub fn set_times(&mut self, attack_seconds: T, release_seconds: T) -> bool {
        if !self.sample_rate.is_finite() || self.sample_rate <= T::zero() {
            return false;
        }
        self.attack_coefficient = Self::coefficient(attack_seconds, self.sample_rate);
        self.release_coefficient = Self::coefficient(release_seconds, self.sample_rate);
        true
    }

    #[inline]
    pub fn next(&mut self, level: T) -> T {
        if !level.is_finite() {
            self.reset();
            return T::zero();
        }
        let level = level.abs();
        let coefficient = if level > self.envelope {
            self.attack_coefficient
        } else {
            self.release_coefficient
        };
        self.envelope = level + coefficient * (self.envelope - level);
        self.envelope
    }

    pub fn reset(&mut self) {
        self.envelope = T::zero();
    }

    fn coefficient(time_seconds: T, sample_rate: T) -> T {
        let minimum_time = T::from(1.0e-6).unwrap();
        let time_seconds = if time_seconds.is_finite() {
            time_seconds.max(minimum_time)
        } else {
            minimum_time
        };
        (-T::one() / (time_seconds * sample_rate)).exp()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LinearSmoother<T: Float> {
    current: T,
    target: T,
    step: T,
    remaining: u32,
}

impl<T: Float> Default for LinearSmoother<T> {
    fn default() -> Self {
        Self {
            current: T::zero(),
            target: T::zero(),
            step: T::zero(),
            remaining: 0,
        }
    }
}

impl<T: Float> LinearSmoother<T> {
    pub fn reset(&mut self, value: T) {
        self.current = value;
        self.target = value;
        self.step = T::zero();
        self.remaining = 0;
    }

    pub fn set_target(&mut self, target: T, samples: u32) {
        if !target.is_finite() {
            return;
        }
        self.target = target;
        self.remaining = samples;
        if samples == 0 {
            self.current = target;
            self.step = T::zero();
        } else {
            self.step = (target - self.current) / T::from(samples).unwrap();
        }
    }

    #[inline]
    pub fn next_value(&mut self) -> T {
        if self.remaining > 0 {
            self.current = self.current + self.step;
            self.remaining -= 1;
            if self.remaining == 0 {
                self.current = self.target;
            }
        }
        self.current
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BypassCrossfade<T: Float> {
    mix: LinearSmoother<T>,
}

impl<T: Float> Default for BypassCrossfade<T> {
    fn default() -> Self {
        let mut mix = LinearSmoother::default();
        mix.reset(T::one());
        Self { mix }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LatencyDelay<T: Float, const MAX_DELAY: usize> {
    buffer: [T; MAX_DELAY],
    cursor: usize,
    delay_samples: usize,
}

impl<T: Float, const MAX_DELAY: usize> Default for LatencyDelay<T, MAX_DELAY> {
    fn default() -> Self {
        Self {
            buffer: [T::zero(); MAX_DELAY],
            cursor: 0,
            delay_samples: 0,
        }
    }
}

impl<T: Float, const MAX_DELAY: usize> LatencyDelay<T, MAX_DELAY> {
    pub fn set_delay_samples(&mut self, delay_samples: usize) -> bool {
        if delay_samples > MAX_DELAY {
            return false;
        }
        self.delay_samples = delay_samples;
        true
    }

    #[inline]
    pub fn process(&mut self, input: T) -> T {
        if self.delay_samples == 0 {
            return input;
        }

        self.buffer[self.cursor] = input;
        let read_index = (self.cursor + MAX_DELAY - self.delay_samples) % MAX_DELAY;
        let output = self.buffer[read_index];
        self.cursor = (self.cursor + 1) % MAX_DELAY;
        output
    }

    pub fn reset(&mut self) {
        self.buffer.fill(T::zero());
        self.cursor = 0;
    }
}

impl<T: Float> BypassCrossfade<T> {
    pub fn set_bypassed(&mut self, bypassed: bool, ramp_samples: u32) {
        self.mix
            .set_target(if bypassed { T::zero() } else { T::one() }, ramp_samples);
    }

    #[inline]
    pub fn process(&mut self, dry: T, wet: T) -> T {
        let wet_mix = self.mix.next_value();
        dry * (T::one() - wet_mix) + wet * wet_mix
    }
}

#[cfg(test)]
mod tests {
    use super::{BypassCrossfade, EnvelopeFollower, LatencyDelay, LinearSmoother};

    #[test]
    fn envelope_coefficients_track_sample_rate_and_support_double_precision() {
        let mut follower = EnvelopeFollower::<f64>::default();
        assert!(follower.initialize(96_000.0, 0.001, 0.2));
        assert!(follower.next(1.0) > 0.0);
    }

    #[test]
    fn smoother_reaches_target_in_exact_sample_count() {
        let mut smoother = LinearSmoother::<f32>::default();
        smoother.reset(0.0);
        smoother.set_target(1.0, 4);
        for _ in 0..3 {
            smoother.next_value();
        }
        assert_eq!(smoother.next_value(), 1.0);
    }

    #[test]
    fn bypass_uses_a_click_free_linear_transition() {
        let mut bypass = BypassCrossfade::<f32>::default();
        bypass.set_bypassed(true, 2);
        assert_eq!(bypass.process(1.0, 0.0), 0.5);
        assert_eq!(bypass.process(1.0, 0.0), 1.0);
    }

    #[test]
    fn latency_delay_applies_exact_integer_delay() {
        let mut delay = LatencyDelay::<f32, 32>::default();
        assert!(delay.set_delay_samples(3));
        assert_eq!(delay.process(1.0), 0.0);
        assert_eq!(delay.process(0.0), 0.0);
        assert_eq!(delay.process(0.0), 0.0);
        assert_eq!(delay.process(0.0), 1.0);
    }

    #[test]
    fn latency_delay_rejects_delays_beyond_its_fixed_storage() {
        let mut delay = LatencyDelay::<f32, 32>::default();
        assert!(!delay.set_delay_samples(33));
    }
}
