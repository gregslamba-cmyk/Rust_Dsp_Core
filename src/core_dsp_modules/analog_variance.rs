#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalogVariance {
    pub left_gain: f32,
    pub right_gain: f32,
    pub left_time_scale: f32,
    pub right_time_scale: f32,
    pub crosstalk: f32,
}

impl AnalogVariance {
    pub fn from_seed(seed: u64, tolerance: f32, crosstalk_max: f32) -> Self {
        let tolerance = if tolerance.is_finite() {
            tolerance.abs().min(0.5)
        } else {
            0.0
        };
        let crosstalk_max = if crosstalk_max.is_finite() {
            crosstalk_max.abs().min(0.25)
        } else {
            0.0
        };
        let mut rng = seed.max(1);
        let mut factor = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
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
            AnalogVariance::from_seed(123, 0.03, 0.01),
            AnalogVariance::from_seed(123, 0.03, 0.01)
        );
    }
}
