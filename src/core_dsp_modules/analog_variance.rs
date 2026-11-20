#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalogVariance {
    pub left_gain: f64,
    pub right_gain: f64,
    pub left_time_scale: f64,
    pub right_time_scale: f64,
    pub crosstalk: f64,
}

impl AnalogVariance {
    pub fn from_seed(seed: u64, tolerance: f64, crosstalk_max: f64, analog_drift: f64) -> Self {
        let drift = if analog_drift.is_finite() {
            analog_drift.clamp(0.0, 1.0)
        } else {
            1.0
        };

        if drift <= 0.0 {
            return Self {
                left_gain: 1.0,
                right_gain: 1.0,
                left_time_scale: 1.0,
                right_time_scale: 1.0,
                crosstalk: 0.0,
            };
        }

        let tolerance = if tolerance.is_finite() {
            tolerance.abs().min(0.05) * drift
        } else {
            0.02 * drift
        };

        let crosstalk_max = if crosstalk_max.is_finite() {
            crosstalk_max.abs().min(0.05) * drift
        } else {
            0.0
        };

        let mut rng = seed.max(1);
        let mut factor = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        Self {
            left_gain: 1.0 + factor() * tolerance,
            right_gain: 1.0 + factor() * tolerance,
            left_time_scale: 1.0 + factor() * tolerance,
            right_time_scale: 1.0 + factor() * tolerance,
            crosstalk: factor() * crosstalk_max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AnalogVariance;

    #[test]
    fn seed_produces_repeatable_instance_variation() {
        assert_eq!(
            AnalogVariance::from_seed(123, 0.03, 0.01, 1.0),
            AnalogVariance::from_seed(123, 0.03, 0.01, 1.0)
        );
    }

    #[test]
    fn zero_drift_disables_variance() {
        let v = AnalogVariance::from_seed(123, 0.03, 0.01, 0.0);
        assert_eq!(v.left_gain, 1.0);
        assert_eq!(v.right_gain, 1.0);
        assert_eq!(v.crosstalk, 0.0);
    }
}

