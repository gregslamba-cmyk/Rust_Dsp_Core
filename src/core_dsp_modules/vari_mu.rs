use num_traits::Float;

pub const VARI_MU_ATTACK_SECONDS: f64 = 0.0002;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum TimeConstant {
    #[default]
    I,
    II,
    III,
    IV,
    V,
    VI,
}

impl TimeConstant {
    fn release_seconds<T: Float>(self) -> (T, Option<T>) {
        let (first, second) = match self {
            Self::I => (0.3, None),
            Self::II => (0.8, None),
            Self::III => (2.0, None),
            Self::IV => (5.0, None),
            Self::V => (2.0, Some(10.0)),
            Self::VI => (0.3, Some(25.0)),
        };
        (
            T::from(first).unwrap(),
            second.map(|seconds| T::from(seconds).unwrap()),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VariMuCell<T: Float> {
    sample_rate: T,
    envelope: T,
    gain_reduction_db: T,
    release_start_db: T,
    in_slow_release: bool,
    time_constant: TimeConstant,
}

impl<T: Float> Default for VariMuCell<T> {
    fn default() -> Self {
        Self {
            sample_rate: T::zero(),
            envelope: T::zero(),
            gain_reduction_db: T::zero(),
            release_start_db: T::zero(),
            in_slow_release: false,
            time_constant: TimeConstant::default(),
        }
    }
}

impl<T: Float> VariMuCell<T> {
    pub fn initialize(&mut self, sample_rate: T, time_constant: TimeConstant) -> bool {
        if !sample_rate.is_finite() || sample_rate <= T::zero() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.time_constant = time_constant;
        self.reset();
        true
    }

    pub fn set_time_constant(&mut self, time_constant: TimeConstant) {
        if self.time_constant != time_constant {
            self.time_constant = time_constant;
            self.release_start_db = T::zero();
            self.in_slow_release = false;
        }
    }

    #[inline]
    pub fn process(&mut self, detector_level: T, threshold_db: T) -> T {
        if !self.sample_rate.is_finite() || self.sample_rate <= T::zero() {
            return T::zero();
        }

        let detector_level = if detector_level.is_finite() {
            detector_level.abs()
        } else {
            T::zero()
        };
        let threshold_db = if threshold_db.is_finite() {
            threshold_db
        } else {
            T::zero()
        };
        if detector_level > self.envelope {
            let attack =
                (-T::one() / (self.sample_rate * T::from(VARI_MU_ATTACK_SECONDS).unwrap())).exp();
            self.envelope = detector_level + attack * (self.envelope - detector_level);
        } else {
            self.envelope = detector_level;
        }

        let level_db =
            T::from(20.0).unwrap() * self.envelope.max(T::from(1.0e-12).unwrap()).log10();
        let target = vari_mu_target_reduction_db(level_db, threshold_db);
        if target >= self.gain_reduction_db {
            self.release_start_db = T::zero();
            self.in_slow_release = false;
            let attack =
                (-T::one() / (self.sample_rate * T::from(VARI_MU_ATTACK_SECONDS).unwrap())).exp();
            self.gain_reduction_db = target + attack * (self.gain_reduction_db - target);
        } else {
            if self.release_start_db <= T::zero() {
                self.release_start_db = self.gain_reduction_db;
            }
            let (fast_release, slow_release) = self.time_constant.release_seconds::<T>();
            if !self.in_slow_release
                && slow_release.is_some()
                && self.gain_reduction_db <= self.release_start_db * T::from(0.5).unwrap()
            {
                self.in_slow_release = true;
            }
            let release = if self.in_slow_release {
                slow_release.unwrap_or(fast_release)
            } else {
                fast_release
            };
            let coefficient = (-T::one() / (self.sample_rate * release)).exp();
            self.gain_reduction_db = target + coefficient * (self.gain_reduction_db - target);
        }
        self.gain_reduction_db
    }

    pub fn reset(&mut self) {
        self.envelope = T::zero();
        self.gain_reduction_db = T::zero();
        self.release_start_db = T::zero();
        self.in_slow_release = false;
    }
}

pub fn vari_mu_target_reduction_db<T: Float>(input_level_db: T, threshold_db: T) -> T {
    if !input_level_db.is_finite() || !threshold_db.is_finite() {
        return T::zero();
    }

    let over_threshold = input_level_db - threshold_db;
    let knee = T::from(2.0).unwrap();
    if over_threshold <= -knee {
        return T::zero();
    }

    let over = over_threshold.max(T::zero());
    let knee_progress = ((over_threshold + knee) / (knee + knee)).min(T::one());
    let ratio = T::from(2.0).unwrap()
        + T::from(18.0).unwrap() * (T::one() - (-over / T::from(18.0).unwrap()).exp());
    let reduction = over * (T::one() - T::one() / ratio);
    if over_threshold < knee {
        reduction * knee_progress * knee_progress
    } else {
        reduction
    }
}

#[cfg(test)]
mod tests {
    use super::{TimeConstant, VARI_MU_ATTACK_SECONDS, VariMuCell, vari_mu_target_reduction_db};

    #[test]
    fn historical_fairchild_attack_nominal_is_represented() {
        assert_eq!(VARI_MU_ATTACK_SECONDS, 0.0002);
    }

    #[test]
    fn ratio_increases_with_gain_reduction() {
        let low = vari_mu_target_reduction_db(-17.0_f64, -18.0);
        let high = vari_mu_target_reduction_db(6.0_f64, -18.0);

        assert!(low > 0.0);
        assert!(high > low * 10.0);
        assert_eq!(vari_mu_target_reduction_db(-25.0_f64, -18.0), 0.0);
    }

    #[test]
    fn six_time_constants_have_distinct_release_behaviors() {
        let mut results = Vec::new();
        for time_constant in [
            TimeConstant::I,
            TimeConstant::II,
            TimeConstant::III,
            TimeConstant::IV,
            TimeConstant::V,
            TimeConstant::VI,
        ] {
            let mut cell = VariMuCell::<f32>::default();
            assert!(cell.initialize(48_000.0, time_constant));
            for _ in 0..48_000 {
                cell.process(1.0, -18.0);
            }
            for _ in 0..144_000 {
                cell.process(0.0, -18.0);
            }
            results.push(cell.process(0.0, -18.0));
        }

        assert!(results[0] < results[1]);
        assert!(results[1] < results[2]);
        assert!(results[2] < results[3]);
        assert!(results[4] > results[2]);
        assert!(results[5] > results[0]);
    }

    #[test]
    fn sample_rate_is_required_and_invalid_samples_are_sanitized() {
        let mut cell = VariMuCell::<f64>::default();
        assert!(!cell.initialize(0.0, TimeConstant::I));
        assert!(cell.initialize(48_000.0, TimeConstant::I));
        assert_eq!(cell.process(f64::NAN, f64::NAN), 0.0);
    }
}
