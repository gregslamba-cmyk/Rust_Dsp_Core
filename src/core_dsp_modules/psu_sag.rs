use num_traits::Float;

#[derive(Clone, Copy, Debug)]
pub struct PsuSag<T: Float> {
    envelope: T,
    coefficient: T,
    depth: T,
}

impl<T: Float> Default for PsuSag<T> {
    fn default() -> Self {
        Self {
            envelope: T::zero(),
            coefficient: T::zero(),
            depth: T::zero(),
        }
    }
}

impl<T: Float> PsuSag<T> {
    pub fn initialize(&mut self, sample_rate: T, recovery_seconds: T, depth: T) -> bool {
        if !sample_rate.is_finite()
            || sample_rate <= T::zero()
            || !recovery_seconds.is_finite()
            || recovery_seconds <= T::zero()
            || !depth.is_finite()
        {
            return false;
        }
        self.coefficient = (-T::one() / (sample_rate * recovery_seconds)).exp();
        self.depth = depth.max(T::zero()).min(T::one());
        self.envelope = T::zero();
        true
    }

    #[inline]
    pub fn gain(&mut self, input_level: T) -> T {
        let level = if input_level.is_finite() {
            input_level.abs()
        } else {
            T::zero()
        };
        self.envelope = level + self.coefficient * (self.envelope - level);
        let normalized_load = self.envelope / (T::one() + self.envelope);
        T::one() - self.depth * normalized_load
    }

    pub fn reset(&mut self) {
        self.envelope = T::zero();
    }
}
