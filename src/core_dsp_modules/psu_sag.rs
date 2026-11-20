use num_traits::Float;

#[derive(Clone, Copy, Debug)]
pub struct PsuSag<T: Float> {
    envelope: f64,
    coefficient: f64,
    depth: f64,
    noise_level: f64,
    ripple_enabled: bool,
    ripple_phase: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Default for PsuSag<T> {
    fn default() -> Self {
        Self {
            envelope: 0.0,
            coefficient: 0.0,
            depth: 0.0,
            noise_level: 0.0,      // OFF BY DEFAULT
            ripple_enabled: false, // OFF BY DEFAULT
            ripple_phase: 0.0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Float> PsuSag<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize(&mut self, sample_rate: T, recovery_seconds: T, depth: T) -> bool {
        let sr = sample_rate.to_f64().unwrap_or(44100.0);
        let rec = recovery_seconds.to_f64().unwrap_or(0.1);
        let dep = depth.to_f64().unwrap_or(0.0);

        if !sr.is_finite() || sr <= 0.0 || !rec.is_finite() || rec <= 0.0 || !dep.is_finite() {
            return false;
        }
        self.coefficient = (-1.0 / (sr * rec)).exp();
        self.depth = dep.clamp(0.0, 1.0);
        self.envelope = 0.0;
        self.ripple_phase = 0.0;
        true
    }

    pub fn set_noise_and_ripple(&mut self, noise_level: T, ripple_enabled: bool) {
        let nl = noise_level.to_f64().unwrap_or(0.0);
        if nl.is_finite() {
            self.noise_level = nl.max(0.0);
        }
        self.ripple_enabled = ripple_enabled;
    }

    #[inline]
    pub fn gain(&mut self, input_level: T) -> T {
        let level = if let Some(v) = input_level.to_f64() {
            if v.is_finite() { v.abs() } else { 0.0 }
        } else {
            0.0
        };
        self.envelope = level + self.coefficient * (self.envelope - level);
        let normalized_load = self.envelope / (1.0 + self.envelope);
        let mut sag = 1.0 - self.depth * normalized_load;

        if self.ripple_enabled && level > 1e-6 {
            self.ripple_phase += 0.01;
            if self.ripple_phase > std::f64::consts::TAU {
                self.ripple_phase -= std::f64::consts::TAU;
            }
            sag += self.ripple_phase.sin() * 0.001 * self.depth;
        }

        if self.noise_level > 0.0 && level > 1e-6 {
            let pseudo_noise = (self.envelope * 12345.6789).sin() * self.noise_level * 0.001;
            sag += pseudo_noise;
        }

        let final_sag = if sag.is_finite() {
            sag.clamp(0.1, 1.0)
        } else {
            1.0
        };
        T::from(final_sag).unwrap_or(T::one())
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.ripple_phase = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::PsuSag;

    #[test]
    fn test_psu_sag_zero_silence_at_rest() {
        let mut default_psu = PsuSag::<f64>::new();
        assert!(default_psu.initialize(48000.0, 0.1, 0.5));
        assert_eq!(default_psu.gain(0.0), 1.0);
    }
}
