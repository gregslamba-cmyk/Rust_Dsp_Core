use std::sync::atomic::{AtomicU32, Ordering};

const METER_FLOOR_DBFS: f32 = -120.0;

/// Visualization data snapshot exposed to the plugin/clap host.
/// Contains input, output, and phase correlation values for display/metering purposes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeterVisualization {
    /// Input peak level in dBFS
    pub input_peak_dbfs: f32,
    /// Input RMS level in dBFS
    pub input_rms_dbfs: f32,
    /// Output peak level in dBFS
    pub output_peak_dbfs: f32,
    /// Output RMS level in dBFS
    pub output_rms_dbfs: f32,
    /// Gain reduction applied in dB
    pub gain_reduction_db: f32,
    /// Phase correlation between L/R channels (-1.0 to 1.0)
    /// - 1.0: perfectly correlated (in-phase)
    /// - -1.0: perfectly anti-correlated (out-of-phase)
    /// - 0.0: uncorrelated
    pub phase_correlation: f32,
    /// Sidechain input peak level in dBFS (if sidechain is active)
    /// Note: The sidechain processor accumulates internally; this is exposed
    /// via the Metering::snapshot() method. Direct access to sidechain input
    /// accumulators is not provided (sidechain module doesn't expose input).
    pub sidechain_input_peak_dbfs: f32,
    /// Sidechain input RMS level in dBFS
    pub sidechain_input_rms_dbfs: f32,
}

impl Default for MeterVisualization {
    fn default() -> Self {
        Self {
            input_peak_dbfs: METER_FLOOR_DBFS,
            input_rms_dbfs: METER_FLOOR_DBFS,
            output_peak_dbfs: METER_FLOOR_DBFS,
            output_rms_dbfs: METER_FLOOR_DBFS,
            gain_reduction_db: 0.0,
            phase_correlation: 0.0,
            sidechain_input_peak_dbfs: METER_FLOOR_DBFS,
            sidechain_input_rms_dbfs: METER_FLOOR_DBFS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeterSnapshot {
    pub input_peak_dbfs: f32,
    pub input_rms_dbfs: f32,
    pub output_peak_dbfs: f32,
    pub output_rms_dbfs: f32,
    pub gain_reduction_db: f32,
    /// Phase correlation between left and right channels (-1.0 to 1.0).
    /// Only meaningful for stereo (2-channel) signals.
    pub phase_correlation: f32,
    /// Sidechain input peak level in dBFS.
    pub sidechain_input_peak_dbfs: f32,
    /// Sidechain input RMS level in dBFS.
    pub sidechain_input_rms_dbfs: f32,
}

impl Default for MeterSnapshot {
    fn default() -> Self {
        Self {
            input_peak_dbfs: METER_FLOOR_DBFS,
            input_rms_dbfs: METER_FLOOR_DBFS,
            output_peak_dbfs: METER_FLOOR_DBFS,
            output_rms_dbfs: METER_FLOOR_DBFS,
            gain_reduction_db: 0.0,
            phase_correlation: 0.0,
            sidechain_input_peak_dbfs: METER_FLOOR_DBFS,
            sidechain_input_rms_dbfs: METER_FLOOR_DBFS,
        }
    }
}

#[derive(Debug)]
pub struct Metering {
    input_peak: f32,
    input_sum_squares: f64,
    output_peak: f32,
    output_sum_squares: f64,
    gain_reduction_peak: f32,
    sample_count: u64,
    // Phase correlation accumulators (for stereo)
    phase_corr_sum_lr: f64,
    phase_corr_sum_ll: f64,
    phase_corr_sum_rr: f64,
    // Sidechain input accumulators
    sidechain_input_peak: f32,
    sidechain_input_sum_squares: f64,
    published: [AtomicU32; 8],
}

impl Default for Metering {
    fn default() -> Self {
        let silent = METER_FLOOR_DBFS.to_bits();
        Self {
            input_peak: 0.0,
            input_sum_squares: 0.0,
            output_peak: 0.0,
            output_sum_squares: 0.0,
            gain_reduction_peak: 0.0,
            sample_count: 0,
            phase_corr_sum_lr: 0.0,
            phase_corr_sum_ll: 0.0,
            phase_corr_sum_rr: 0.0,
            sidechain_input_peak: 0.0,
            sidechain_input_sum_squares: 0.0,
            published: [
                AtomicU32::new(silent),       // input_peak_dbfs
                AtomicU32::new(silent),       // input_rms_dbfs
                AtomicU32::new(silent),       // output_peak_dbfs
                AtomicU32::new(silent),       // output_rms_dbfs
                AtomicU32::new(0.0_f32.to_bits()), // gain_reduction_db
                AtomicU32::new(0.0_f32.to_bits()), // phase_correlation
                AtomicU32::new(silent),       // sidechain_input_peak_dbfs
                AtomicU32::new(silent),       // sidechain_input_rms_dbfs
            ],
        }
    }
}

impl Metering {
    pub fn begin_block(&mut self) {
        self.input_peak = 0.0;
        self.input_sum_squares = 0.0;
        self.output_peak = 0.0;
        self.output_sum_squares = 0.0;
        self.gain_reduction_peak = 0.0;
        self.sample_count = 0;
        self.phase_corr_sum_lr = 0.0;
        self.phase_corr_sum_ll = 0.0;
        self.phase_corr_sum_rr = 0.0;
        self.sidechain_input_peak = 0.0;
        self.sidechain_input_sum_squares = 0.0;
    }

    #[inline]
    pub fn record_sample(&mut self, input: f32, output: f32, gain_reduction_db: f32) {
        let input = finite(input);
        let output = finite(output);
        self.input_peak = self.input_peak.max(input.abs());
        self.input_sum_squares += f64::from(input) * f64::from(input);
        self.output_peak = self.output_peak.max(output.abs());
        self.output_sum_squares += f64::from(output) * f64::from(output);
        if gain_reduction_db.is_finite() {
            self.gain_reduction_peak = self.gain_reduction_peak.max(gain_reduction_db.max(0.0));
        }
        self.sample_count = self.sample_count.saturating_add(1);
    }

    /// Record a stereo sample pair for phase correlation calculation.
    /// Call this in addition to `record_sample` for stereo signals.
    #[inline]
    pub fn record_stereo_sample(&mut self, left: f32, right: f32) {
        let left = finite(left);
        let right = finite(right);
        self.phase_corr_sum_lr += f64::from(left) * f64::from(right);
        self.phase_corr_sum_ll += f64::from(left) * f64::from(left);
        self.phase_corr_sum_rr += f64::from(right) * f64::from(right);
    }

    /// Record sidechain input sample for metering.
    #[inline]
    pub fn record_sidechain_input(&mut self, sidechain_input: f32) {
        let sidechain_input = finite(sidechain_input);
        self.sidechain_input_peak = self.sidechain_input_peak.max(sidechain_input.abs());
        self.sidechain_input_sum_squares += f64::from(sidechain_input) * f64::from(sidechain_input);
    }

    pub fn publish_block(&self) {
        let (input_peak_dbfs, input_rms_dbfs, output_peak_dbfs, output_rms_dbfs) =
            if self.sample_count == 0 {
                (
                    METER_FLOOR_DBFS,
                    METER_FLOOR_DBFS,
                    METER_FLOOR_DBFS,
                    METER_FLOOR_DBFS,
                )
            } else {
                let count = self.sample_count as f64;
                (
                    amplitude_to_dbfs(self.input_peak),
                    amplitude_to_dbfs((self.input_sum_squares / count).sqrt() as f32),
                    amplitude_to_dbfs(self.output_peak),
                    amplitude_to_dbfs((self.output_sum_squares / count).sqrt() as f32),
                )
            };

        // Calculate phase correlation (Pearson correlation coefficient for stereo)
        let phase_correlation = if self.sample_count >= 2 && self.phase_corr_sum_ll > 0.0 && self.phase_corr_sum_rr > 0.0 {
            let corr = self.phase_corr_sum_lr / (self.phase_corr_sum_ll * self.phase_corr_sum_rr).sqrt();
            corr.clamp(-1.0, 1.0) as f32
        } else {
            0.0
        };

        let (sidechain_input_peak_dbfs, sidechain_input_rms_dbfs) =
            if self.sample_count == 0 {
                (METER_FLOOR_DBFS, METER_FLOOR_DBFS)
            } else {
                let count = self.sample_count as f64;
                (
                    amplitude_to_dbfs(self.sidechain_input_peak),
                    amplitude_to_dbfs((self.sidechain_input_sum_squares / count).sqrt() as f32),
                )
            };

        let values = [
            input_peak_dbfs,
            input_rms_dbfs,
            output_peak_dbfs,
            output_rms_dbfs,
            self.gain_reduction_peak,
            phase_correlation,
            sidechain_input_peak_dbfs,
            sidechain_input_rms_dbfs,
        ];
        for (published, value) in self.published.iter().zip(values) {
            published.store(value.to_bits(), Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> MeterSnapshot {
        MeterSnapshot {
            input_peak_dbfs: f32::from_bits(self.published[0].load(Ordering::Relaxed)),
            input_rms_dbfs: f32::from_bits(self.published[1].load(Ordering::Relaxed)),
            output_peak_dbfs: f32::from_bits(self.published[2].load(Ordering::Relaxed)),
            output_rms_dbfs: f32::from_bits(self.published[3].load(Ordering::Relaxed)),
            gain_reduction_db: f32::from_bits(self.published[4].load(Ordering::Relaxed)),
            phase_correlation: f32::from_bits(self.published[5].load(Ordering::Relaxed)),
            sidechain_input_peak_dbfs: f32::from_bits(self.published[6].load(Ordering::Relaxed)),
            sidechain_input_rms_dbfs: f32::from_bits(self.published[7].load(Ordering::Relaxed)),
        }
    }
}

#[inline]
fn finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[inline]
fn amplitude_to_dbfs(amplitude: f32) -> f32 {
    if amplitude <= 0.0 {
        METER_FLOOR_DBFS
    } else {
        (20.0 * amplitude.log10()).max(METER_FLOOR_DBFS)
    }
}

#[cfg(test)]
mod tests {
    use super::{METER_FLOOR_DBFS, MeterSnapshot, Metering};

    #[test]
    fn reports_block_peak_rms_and_maximum_gain_reduction() {
        let mut metering = Metering::default();
        metering.begin_block();
        metering.record_sample(1.0, 0.5, 3.0);
        metering.record_sample(-1.0, -0.5, 6.0);
        metering.publish_block();

        assert_eq!(
            metering.snapshot(),
            MeterSnapshot {
                input_peak_dbfs: 0.0,
                input_rms_dbfs: 0.0,
                output_peak_dbfs: 20.0 * 0.5_f32.log10(),
                output_rms_dbfs: 20.0 * 0.5_f32.log10(),
                gain_reduction_db: 6.0,
                phase_correlation: 0.0,
                sidechain_input_peak_dbfs: METER_FLOOR_DBFS,
                sidechain_input_rms_dbfs: METER_FLOOR_DBFS,
            }
        );
    }

    #[test]
    fn clears_each_block_and_sanitizes_non_finite_samples() {
        let mut metering = Metering::default();
        metering.begin_block();
        metering.record_sample(f32::NAN, f32::INFINITY, f32::NAN);
        metering.publish_block();
        assert_eq!(metering.snapshot(), MeterSnapshot::default());

        metering.begin_block();
        metering.publish_block();
        assert_eq!(
            metering.snapshot(),
            MeterSnapshot {
                input_peak_dbfs: METER_FLOOR_DBFS,
                input_rms_dbfs: METER_FLOOR_DBFS,
                output_peak_dbfs: METER_FLOOR_DBFS,
                output_rms_dbfs: METER_FLOOR_DBFS,
                gain_reduction_db: 0.0,
                phase_correlation: 0.0,
                sidechain_input_peak_dbfs: METER_FLOOR_DBFS,
                sidechain_input_rms_dbfs: METER_FLOOR_DBFS,
            }
        );
    }

    #[test]
    fn phase_correlation_perfect_positive() {
        let mut metering = Metering::default();
        metering.begin_block();
        for _ in 0..100 {
            metering.record_sample(0.5, 0.5, 0.0);
            metering.record_stereo_sample(0.5, 0.5);
        }
        metering.publish_block();
        let snap = metering.snapshot();
        assert!((snap.phase_correlation - 1.0).abs() < 0.01);
    }

    #[test]
    fn phase_correlation_perfect_negative() {
        let mut metering = Metering::default();
        metering.begin_block();
        for _ in 0..100 {
            metering.record_sample(0.5, -0.5, 0.0);
            metering.record_stereo_sample(0.5, -0.5);
        }
        metering.publish_block();
        let snap = metering.snapshot();
        assert!((snap.phase_correlation + 1.0).abs() < 0.01);
    }

    #[test]
    fn phase_correlation_uncorrelated() {
        let mut metering = Metering::default();
        metering.begin_block();
        for i in 0..100 {
            let l = if i % 2 == 0 { 0.5 } else { -0.5 };
            let r = if i % 4 < 2 { 0.5 } else { -0.5 };
            metering.record_sample(l, r, 0.0);
            metering.record_stereo_sample(l, r);
        }
        metering.publish_block();
        let snap = metering.snapshot();
        assert!(snap.phase_correlation.abs() < 0.3);
    }

    #[test]
    fn sidechain_input_metering() {
        let mut metering = Metering::default();
        metering.begin_block();
        metering.record_sample(0.5, 0.5, 0.0);
        metering.record_sidechain_input(0.25);
        metering.record_sidechain_input(-0.25);
        metering.publish_block();
        let snap = metering.snapshot();
        // Peak: max(|0.25|, |-0.25|) = 0.25 -> ~-12.04 dBFS
        let expected_peak = 20.0 * 0.25_f32.log10();
        assert!((snap.sidechain_input_peak_dbfs - expected_peak).abs() < 0.01);
        // RMS: sqrt((0.25^2 + (-0.25)^2) / sample_count=1) = sqrt(0.125) ≈ 0.3536 -> ~-9.03 dBFS
        // The RMS reflects true energy detection, not just peak
        let rms_value = (0.125_f32).sqrt(); // = sqrt(0.25^2 + (-0.25)^2) / 1 = sqrt(0.125)
        let expected_rms = 20.0 * rms_value.log10();
        assert!((snap.sidechain_input_rms_dbfs - expected_rms).abs() < 0.01);
    }
}
