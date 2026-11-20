use num_traits::Float;

pub const OPTO_A2A_ATTACK_SECONDS: f64 = 0.010;
pub const OPTO_A2A_FAST_RELEASE_SECONDS: f64 = 0.060;
pub const OPTO_A2A_SLOW_RELEASE_MIN_SECONDS: f64 = 0.5;
pub const OPTO_A2A_SLOW_RELEASE_MAX_SECONDS: f64 = 5.0;
pub const OPTO_A3A_ATTACK_SECONDS: f64 = 0.0035;
pub const OPTO_A3A_FAST_RELEASE_SECONDS: f64 = 0.025;
pub const OPTO_A3A_SLOW_RELEASE_MIN_SECONDS: f64 = 0.12;
pub const OPTO_A3A_SLOW_RELEASE_MAX_SECONDS: f64 = 1.5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptoMode {
    #[default]
    Compress,
    Limit,
}

#[derive(Clone, Copy, Debug)]
pub struct OptoCell<T: Float> {
    sample_rate: T,
    reduction_db: T,
    program_memory: T,
    fast_release: bool,
    release_start_db: T,
    slow_release_seconds: T,
}

impl<T: Float> Default for OptoCell<T> {
    fn default() -> Self {
        Self {
            sample_rate: T::zero(),
            reduction_db: T::zero(),
            program_memory: T::zero(),
            fast_release: false,
            release_start_db: T::zero(),
            slow_release_seconds: T::one(),
        }
    }
}

impl<T: Float> OptoCell<T> {
    pub fn initialize(&mut self, sample_rate: T) -> bool {
        if !sample_rate.is_finite() || sample_rate <= T::zero() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.reset();
        true
    }

    #[inline]
    pub fn process(&mut self, target_reduction_db: T) -> T {
        self.process_with_character(target_reduction_db, false)
    }

    #[inline]
    pub fn process_solid_state(&mut self, target_reduction_db: T) -> T {
        self.process_with_character(target_reduction_db, true)
    }

    #[inline]
    fn process_with_character(&mut self, target_reduction_db: T, solid_state: bool) -> T {
        if !self.sample_rate.is_finite() || self.sample_rate <= T::zero() {
            return T::zero();
        }

        let target = if target_reduction_db.is_finite() {
            target_reduction_db
                .max(T::zero())
                .min(T::from(60.0).unwrap())
        } else {
            T::zero()
        };
        let activity_target = (target / T::from(24.0).unwrap()).min(T::one());
        let memory_seconds = if solid_state {
            T::from(0.5).unwrap()
        } else {
            T::from(2.0).unwrap()
        };
        let memory_coefficient = Self::coefficient(memory_seconds, self.sample_rate);
        self.program_memory =
            activity_target + memory_coefficient * (self.program_memory - activity_target);

        if target > self.reduction_db {
            self.fast_release = true;
            self.release_start_db = target;
            let attack_base = if solid_state {
                OPTO_A3A_ATTACK_SECONDS
            } else {
                OPTO_A2A_ATTACK_SECONDS
            };
            let attack_sensitivity = if solid_state { 0.15 } else { 0.25 };
            let attack_seconds = T::from(attack_base).unwrap()
                / (T::one()
                    + T::from(attack_sensitivity).unwrap()
                        * (target / T::from(20.0).unwrap()).min(T::one()));
            let coefficient = Self::coefficient(attack_seconds, self.sample_rate);
            self.reduction_db = target + coefficient * (self.reduction_db - target);
        } else if target < self.reduction_db {
            if !self.fast_release && self.release_start_db <= T::zero() {
                self.release_start_db = self.reduction_db;
                let (minimum, range) = if solid_state {
                    (
                        OPTO_A3A_SLOW_RELEASE_MIN_SECONDS,
                        OPTO_A3A_SLOW_RELEASE_MAX_SECONDS - OPTO_A3A_SLOW_RELEASE_MIN_SECONDS,
                    )
                } else {
                    (
                        OPTO_A2A_SLOW_RELEASE_MIN_SECONDS,
                        OPTO_A2A_SLOW_RELEASE_MAX_SECONDS - OPTO_A2A_SLOW_RELEASE_MIN_SECONDS,
                    )
                };
                self.slow_release_seconds =
                    T::from(minimum).unwrap() + T::from(range).unwrap() * self.program_memory;
            }

            if self.fast_release
                && self.reduction_db <= self.release_start_db * T::from(0.5).unwrap()
            {
                self.fast_release = false;
                let (minimum, range) = if solid_state {
                    (
                        OPTO_A3A_SLOW_RELEASE_MIN_SECONDS,
                        OPTO_A3A_SLOW_RELEASE_MAX_SECONDS - OPTO_A3A_SLOW_RELEASE_MIN_SECONDS,
                    )
                } else {
                    (
                        OPTO_A2A_SLOW_RELEASE_MIN_SECONDS,
                        OPTO_A2A_SLOW_RELEASE_MAX_SECONDS - OPTO_A2A_SLOW_RELEASE_MIN_SECONDS,
                    )
                };
                self.slow_release_seconds =
                    T::from(minimum).unwrap() + T::from(range).unwrap() * self.program_memory;
            }

            let release_seconds = if self.fast_release {
                let half_life = if solid_state {
                    OPTO_A3A_FAST_RELEASE_SECONDS
                } else {
                    OPTO_A2A_FAST_RELEASE_SECONDS
                };
                T::from(half_life).unwrap() / T::from(std::f64::consts::LN_2).unwrap()
            } else {
                self.slow_release_seconds
            };
            let coefficient = Self::coefficient(release_seconds, self.sample_rate);
            self.reduction_db = target + coefficient * (self.reduction_db - target);
        }

        self.reduction_db
    }

    pub fn reset(&mut self) {
        self.reduction_db = T::zero();
        self.program_memory = T::zero();
        self.fast_release = false;
        self.release_start_db = T::zero();
        self.slow_release_seconds = T::one();
    }

    fn coefficient(time_seconds: T, sample_rate: T) -> T {
        (-T::one() / (time_seconds * sample_rate)).exp()
    }
}

pub fn target_reduction_db<T: Float>(input_level_db: T, peak_reduction_db: T, mode: OptoMode) -> T {
    target_reduction_db_for_character(input_level_db, peak_reduction_db, mode, false)
}

pub fn target_solid_state_reduction_db<T: Float>(
    input_level_db: T,
    peak_reduction_db: T,
    mode: OptoMode,
) -> T {
    target_reduction_db_for_character(input_level_db, peak_reduction_db, mode, true)
}

fn target_reduction_db_for_character<T: Float>(
    input_level_db: T,
    peak_reduction_db: T,
    mode: OptoMode,
    solid_state: bool,
) -> T {
    if !input_level_db.is_finite() || !peak_reduction_db.is_finite() {
        return T::zero();
    }

    let threshold_db = T::from(-6.0).unwrap() - peak_reduction_db.max(T::zero());
    let over_threshold = input_level_db - threshold_db;
    let half_knee = T::one();
    if over_threshold <= -half_knee {
        return T::zero();
    }

    let positive_over = over_threshold.max(T::zero());
    let ratio_at_high_level = T::one() - (-positive_over / T::from(12.0).unwrap()).exp();
    let (base_ratio, ratio_range) = match (solid_state, mode) {
        (false, OptoMode::Compress) => (T::from(2.5).unwrap(), T::from(2.5).unwrap()),
        (false, OptoMode::Limit) => (T::from(5.0).unwrap(), T::from(5.0).unwrap()),
        (true, OptoMode::Compress) => (T::from(3.5).unwrap(), T::from(2.5).unwrap()),
        (true, OptoMode::Limit) => (T::from(6.0).unwrap(), T::from(4.0).unwrap()),
    };
    let ratio = base_ratio + ratio_range * ratio_at_high_level;
    let slope = T::one() - T::one() / ratio;

    if over_threshold < half_knee {
        slope * (over_threshold + half_knee) * (over_threshold + half_knee) / T::from(4.0).unwrap()
    } else {
        slope * over_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OPTO_A2A_ATTACK_SECONDS, OPTO_A2A_FAST_RELEASE_SECONDS, OPTO_A2A_SLOW_RELEASE_MAX_SECONDS,
        OPTO_A2A_SLOW_RELEASE_MIN_SECONDS, OPTO_A3A_ATTACK_SECONDS, OPTO_A3A_FAST_RELEASE_SECONDS,
        OPTO_A3A_SLOW_RELEASE_MAX_SECONDS, OPTO_A3A_SLOW_RELEASE_MIN_SECONDS, OptoCell, OptoMode,
        target_reduction_db, target_solid_state_reduction_db,
    };

    #[test]
    fn historical_opto_nominal_ranges_are_represented() {
        assert_eq!(OPTO_A2A_ATTACK_SECONDS, 0.010);
        assert_eq!(OPTO_A2A_FAST_RELEASE_SECONDS, 0.060);
        assert_eq!(OPTO_A2A_SLOW_RELEASE_MIN_SECONDS, 0.5);
        assert_eq!(OPTO_A2A_SLOW_RELEASE_MAX_SECONDS, 5.0);
        assert_eq!(OPTO_A3A_ATTACK_SECONDS, 0.0035);
        assert_eq!(OPTO_A3A_FAST_RELEASE_SECONDS, 0.025);
        assert_eq!(OPTO_A3A_SLOW_RELEASE_MIN_SECONDS, 0.12);
        assert_eq!(OPTO_A3A_SLOW_RELEASE_MAX_SECONDS, 1.5);
    }

    #[test]
    fn sample_rate_is_required_and_shapes_attack_in_seconds() {
        let mut cell = OptoCell::<f32>::default();
        assert!(!cell.initialize(0.0));
        assert!(cell.initialize(48_000.0));

        let mut at_96k = OptoCell::<f32>::default();
        assert!(at_96k.initialize(96_000.0));
        for _ in 0..479 {
            cell.process(12.0);
        }
        for _ in 0..959 {
            at_96k.process(12.0);
        }
        assert!((cell.process(12.0) - at_96k.process(12.0)).abs() < 0.001);
    }

    #[test]
    fn program_dependent_memory_extends_the_slow_release() {
        let mut short_program = OptoCell::<f64>::default();
        let mut long_program = OptoCell::<f64>::default();
        assert!(short_program.initialize(48_000.0));
        assert!(long_program.initialize(48_000.0));

        for _ in 0..480 {
            short_program.process(12.0);
        }
        for _ in 0..48_000 {
            long_program.process(12.0);
        }
        for _ in 0..48_000 {
            short_program.process(0.0);
            long_program.process(0.0);
        }

        assert!(long_program.process(0.0) > short_program.process(0.0));
    }

    #[test]
    fn release_has_a_fast_first_stage_then_a_slow_tail() {
        let mut cell = OptoCell::<f64>::default();
        assert!(cell.initialize(48_000.0));
        for _ in 0..48_000 {
            cell.process(12.0);
        }
        let peak = cell.process(12.0);

        for _ in 0..2_880 {
            cell.process(0.0);
        }
        let after_fast_stage = cell.process(0.0);
        assert!(after_fast_stage > peak * 0.45 && after_fast_stage < peak * 0.55);

        for _ in 0..48_000 {
            cell.process(0.0);
        }
        let after_slow_tail = cell.process(0.0);
        assert!(after_slow_tail > 0.0 && after_slow_tail < after_fast_stage);
    }

    #[test]
    fn peak_reduction_and_limit_mode_increase_attenuation() {
        let low = target_reduction_db(-8.0_f64, 0.0, OptoMode::Compress);
        let compressed = target_reduction_db(-6.0_f64, 18.0, OptoMode::Compress);
        let limited = target_reduction_db(-6.0_f64, 18.0, OptoMode::Limit);

        assert_eq!(low, 0.0);
        assert!(compressed > 0.0);
        assert!(limited > compressed);
    }

    #[test]
    fn solid_state_variant_tracks_faster_than_tube_variant() {
        let mut tube = OptoCell::<f32>::default();
        let mut solid_state = OptoCell::<f32>::default();
        assert!(tube.initialize(48_000.0));
        assert!(solid_state.initialize(48_000.0));

        for _ in 0..240 {
            tube.process(12.0);
            solid_state.process_solid_state(12.0);
        }
        assert!(solid_state.process_solid_state(12.0) > tube.process(12.0));

        let tube_target = target_reduction_db(-8.0_f32, 18.0, OptoMode::Compress);
        let solid_target = target_solid_state_reduction_db(-8.0_f32, 18.0, OptoMode::Compress);
        assert!(solid_target > tube_target);
    }
}
