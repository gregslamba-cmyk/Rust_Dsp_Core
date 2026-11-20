use std::sync::atomic::{AtomicU32, Ordering};

const METER_FLOOR_DBFS: f32 = -120.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeterSnapshot {
    pub input_peak_dbfs: f32,
    pub input_rms_dbfs: f32,
    pub output_peak_dbfs: f32,
    pub output_rms_dbfs: f32,
    pub gain_reduction_db: f32,
}

impl Default for MeterSnapshot {
    fn default() -> Self {
        Self {
            input_peak_dbfs: METER_FLOOR_DBFS,
            input_rms_dbfs: METER_FLOOR_DBFS,
            output_peak_dbfs: METER_FLOOR_DBFS,
            output_rms_dbfs: METER_FLOOR_DBFS,
            gain_reduction_db: 0.0,
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
    published: [AtomicU32; 5],
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
            published: [
                AtomicU32::new(silent),
                AtomicU32::new(silent),
                AtomicU32::new(silent),
                AtomicU32::new(silent),
                AtomicU32::new(0.0_f32.to_bits()),
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
        let values = [
            input_peak_dbfs,
            input_rms_dbfs,
            output_peak_dbfs,
            output_rms_dbfs,
            self.gain_reduction_peak,
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
            }
        );
    }
}
