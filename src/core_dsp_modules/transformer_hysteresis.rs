#[derive(Clone, Copy, Debug)]
pub struct TransformerHysteresis {
    flux: f64,
    lp_state: f64,
}

impl Default for TransformerHysteresis {
    fn default() -> Self {
        Self {
            flux: 0.0,
            lp_state: 0.0,
        }
    }
}

impl TransformerHysteresis {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn process(&mut self, input: f64, drive: f64, hysteresis: f64, frequency_dependency: f64) -> f64 {
        if !input.is_finite() || !drive.is_finite() || !hysteresis.is_finite() || !frequency_dependency.is_finite() {
            self.reset();
            return 0.0;
        }

        let hysteresis = hysteresis.max(0.0).min(0.999);
        let freq_dep = frequency_dependency.max(0.0).min(1.0);

        let diff = input - self.lp_state;
        self.lp_state += diff * (0.1 + freq_dep * 0.4);
        let effective_input = input - freq_dep * 0.2 * diff;

        self.flux = effective_input * drive + self.flux * hysteresis;
        let output = self.flux.tanh();
        self.flux = output * (1.0 - hysteresis);

        if output.is_finite() {
            output
        } else {
            self.reset();
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.flux = 0.0;
        self.lp_state = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::TransformerHysteresis;

    #[test]
    fn test_zero_silence() {
        let mut th = TransformerHysteresis::new();
        for _ in 0..50 {
            assert_eq!(th.process(0.0, 1.0, 0.5, 0.1), 0.0);
        }
    }

    #[test]
    fn test_stability() {
        let mut th = TransformerHysteresis::new();
        for i in 0..1000 {
            let x = (i as f64 * 0.01).sin();
            let out = th.process(x, 2.0, 0.8, 0.2);
            assert!(out.is_finite());
        }
    }
}

