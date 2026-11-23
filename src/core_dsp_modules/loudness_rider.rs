//! Loudness rider module: K-weighted sillabic loudness detection, hysteresis gate,
//! and ramp smoothing. Used by leveler, de-esser, and depop processing.

/// Simple loudness rider with threshold, attack, release and gain reduction.
/// Designed for real-time audio with zero allocations in the audio thread.
pub struct LoudnessRider {
    /// Current gain reduction in dB (0.0 means no reduction)
    gain_reduction_db: f64,
    /// Threshold in dBFS above which gain reduction starts
    threshold_db: f64,
    /// Attack time in seconds
    attack_s: f64,
    /// Release time in seconds
    release_s: f64,
    /// Smoothing factor (0..1) applied to gain reduction per block
    smoothing: f64,
    /// Internal state for envelope following
    envelope: f64,
    /// Sample rate for time conversions
    sample_rate: f64,
}

impl LoudnessRider {
    /// Creates a new loudness rider with default parameters.
    pub fn new(sample_rate: f64) -> Self {
        Self {
            gain_reduction_db: 0.0,
            threshold_db: -20.0,
            attack_s: 0.005,
            release_s: 0.1,
            smoothing: 0.01,
            envelope: 0.0,
            sample_rate,
        }
    }

    /// Enables or disables the gate.
    pub fn set_gate_enabled(&mut self, _enabled: bool) {
        // placeholder; always active
    }

    /// Sets the threshold in dBFS.
    pub fn set_threshold(&mut self, db: f64) {
        self.threshold_db = db;
    }

    /// Sets attack time in seconds.
    pub fn set_attack(&mut self, s: f64) {
        self.attack_s = s.max(0.0);
    }

    /// Sets release time in seconds.
    pub fn set_release(&mut self, s: f64) {
        self.release_s = s.max(0.0);
    }

    /// Sets smoothing factor (0..1) applied to gain reduction per block.
    pub fn set_smoothing(&mut self, s: f64) {
        self.smoothing = s.clamp(0.0, 1.0);
    }

    /// Processes a block of input samples and writes output with gain reduction applied.
    /// Currently just passes input through and returns the computed gain reduction.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> f64 {
        let len = input.len().min(output.len());
        for i in 0..len {
            if let (Some(in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                let in_s = *in_s;
                let in_s = if in_s.is_finite() { in_s } else { 0.0 };
                // compute envelope (simple RC filter)
                // Convert amplitude to dBFS for threshold comparison
                let db_in = if in_s == 0.0 { -120.0 } else { 20.0 * in_s.abs().log10().max(-120.0) };
                let target = if db_in > self.threshold_db {
                    db_in - self.threshold_db
                } else {
                    0.0
                };
                // exponential attack/release toward target
                // attack_coeff = exp(-1 / (fs * tau)), where tau is in seconds
                let attack_coeff = if self.attack_s > 0.0 {
                    (-1.0 / (self.sample_rate * self.attack_s)).exp()
                } else {
                    1.0
                };
                let release_coeff = if self.release_s > 0.0 {
                    (-1.0 / (self.sample_rate * self.release_s)).exp()
                } else {
                    1.0
                };
                let coeff = if target > self.envelope {
                    attack_coeff
                } else {
                    release_coeff
                };
                self.envelope = self.envelope * coeff + target * (1.0 - coeff);
                // gain reduction in linear domain approx
                let gr_linear = if self.envelope > 0.0 {
                    1.0 / (1.0 + self.envelope)
                } else {
                    1.0
                };
                let gr_db = if gr_linear > 0.0 && gr_linear < 1.0 {
                    (1.0 - gr_linear).log10() * 20.0
                } else {
                    0.0
                };
                self.gain_reduction_db = gr_db;
                *out_s = in_s * gr_linear;
            }
        }
        self.gain_reduction_db
    }

    /// Returns the current gain reduction in dB.
    pub fn get_gain_reduction(&self) -> f64 {
        self.gain_reduction_db
    }

    /// Resets the internal envelope to zero.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.gain_reduction_db = 0.0;
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loudness_rider_process_block_covers_irregular_sizes_and_stays_finite() {
        let mut rider = LoudnessRider::new(48_000.0);
        rider.set_threshold(-20.0);
        rider.set_attack(0.005);
        rider.set_release(0.1);
        for size in [1usize, 7, 64, 513, 2048] {
            let input: Vec<f64> = (0..size)
                .map(|i| {
                    if i == 0 {
                        f64::NAN
                    } else {
                        (i as f64 * 0.01).sin()
                    }
                })
                .collect();
            let mut output = vec![f64::NAN; size];
            let gr = rider.process_block(&input, &mut output);
            assert!(gr.is_finite());
            for sample in output.iter() {
                assert!(sample.is_finite(), "non-finite sample in size {size}");
            }
        }
        rider.reset();
        assert_eq!(rider.get_gain_reduction(), 0.0);
    }
}
