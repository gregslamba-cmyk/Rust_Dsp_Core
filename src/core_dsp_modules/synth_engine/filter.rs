//! Multi-mode state-variable filter (SVF) with zero-delay feedback topology.
//!
//! Modes: Low-pass, High-pass, Band-pass, Notch.
//! All processing is `f64`, no heap allocation, NaN/Inf sanitized to 0.0.

use crate::core_dsp_modules::zdf_filters::{ZdfSvf};
use crate::core_dsp_modules::filters::FilterType;

/// Filter mode types matching ZDF SVF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    LowPass,
    HighPass,
    BandPass,
    Notch,
}

/// SVF wrapper with mode switching.
#[derive(Clone, Copy, Debug)]
pub struct MultiModeFilter {
    pub svf: ZdfSvf,
    pub mode: FilterMode,
}

impl MultiModeFilter {
    pub fn new() -> Self {
        Self {
            svf: ZdfSvf::new(),
            mode: FilterMode::LowPass,
        }
    }

    pub fn configure(&mut self, sample_rate: f64, freq_hz: f64, q: f64, mode: FilterMode) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz >= sample_rate * 0.49 {
            return false;
        }
        if !q.is_finite() || q <= 0.0 {
            return false;
        }
        self.mode = mode;
        let zdf_type = match mode {
            FilterMode::LowPass => FilterType::LowPass,
            FilterMode::HighPass => FilterType::HighPass,
            FilterMode::BandPass => FilterType::BandPass,
            FilterMode::Notch => FilterType::Notch,
        };
        self.svf.configure(sample_rate, zdf_type, freq_hz, q, 0.0, 0.0)
    }

    pub fn set_mode(&mut self, mode: FilterMode) {
        self.mode = mode;
        let zdf_type = match mode {
            FilterMode::LowPass => FilterType::LowPass,
            FilterMode::HighPass => FilterType::HighPass,
            FilterMode::BandPass => FilterType::BandPass,
            FilterMode::Notch => FilterType::Notch,
        };
        self.svf.configure(
            48000.0, // placeholder, actual SR set via configure()
            zdf_type,
            1000.0,
            1.0,
            0.0,
            0.0,
        );
    }

    pub fn reset(&mut self) {
        self.svf.reset();
    }

    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        if !input.is_finite() {
            self.reset();
            return 0.0;
        }
        self.svf.process(input)
    }
}

impl Default for MultiModeFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_lowpass_passes() {
        let mut f = MultiModeFilter::new();
        assert!(f.configure(48000.0, 1000.0, 1.0, FilterMode::LowPass));
        let mut sum = 0.0;
        for i in 0..1000 {
            let v = f.process(if i % 2 == 0 { 1.0 } else { -1.0 });
            assert!(v.is_finite());
            sum += v.abs();
        }
        assert!(sum > 0.0);
    }

    #[test]
    fn filter_all_modes_finite() {
        for mode in [FilterMode::LowPass, FilterMode::HighPass, FilterMode::BandPass, FilterMode::Notch] {
            let mut f = MultiModeFilter::new();
            assert!(f.configure(48000.0, 2000.0, 2.0, mode));
            for _ in 0..100 {
                let v = f.process(0.5);
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn filter_invalid_freq_rejects() {
        let mut f = MultiModeFilter::new();
        assert!(!f.configure(48000.0, -100.0, 1.0, FilterMode::LowPass));
        assert!(!f.configure(48000.0, f64::NAN, 1.0, FilterMode::LowPass));
        assert!(!f.configure(48000.0, 48000.0 * 0.5, 1.0, FilterMode::LowPass));
    }
}