//! Physical analog summing & crosstalk module (Phase 2 - `doit.txt`).

use std::f64::consts::PI;

/// Maximum number of channels supported by the summing bus.
pub const MAX_SUMMER_CHANNELS: usize = 128;

/// Adjacent‑channel crosstalk matrix with one‑pole high‑pass coupling.
pub struct AdjacentCrosstalkMatrix {
    crosstalk_db: f64,
    sample_rate: f64,
    coupling_alpha: f64,
    previous: Vec<f64>,
    num_channels: usize,
}

impl AdjacentCrosstalkMatrix {
    pub fn new(crosstalk_db: f64, sample_rate: f64) -> Self {
        let sr = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let omega = 2.0 * PI * 20.0 / sr;
        let alpha = 1.0 - (-omega).exp();
        let atten = 10.0f64.powf(-crosstalk_db / 20.0);
        Self {
            crosstalk_db,
            sample_rate: sr,
            coupling_alpha: if crosstalk_db <= 0.0 { 0.0 } else { alpha *atten },
            previous: vec![0.0; MAX_SUMMER_CHANNELS],
            num_channels: 2,
        }
    }

    pub fn set_crosstalk_db(&mut self, db: f64) {
        self.crosstalk_db = db;
        let omega = 2.0 * PI * 20.0 / self.sample_rate.max(1.0);
        let alpha = 1.0 - (-omega).exp();
        self.coupling_alpha = if db <= 0.0 { 0.0 } else { alpha * 10.0f64.powf(-db / 20.0) };
    }

    pub fn initialize(&mut self, sample_rate: f64) {
        self.sample_rate = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let omega = 2.0 * PI * 20.0 / self.sample_rate.max(1.0);
        let alpha = 1.0 - (-omega).exp();
        self.coupling_alpha = if self.crosstalk_db <= 0.0 { 0.0 } else { alpha * 10.0f64.powf(-self.crosstalk_db / 20.0) };
    }

    pub fn reset(&mut self) {
        for s in self.previous.iter_mut() { *s = 0.0; }
    }

    /// Sets number of channels dynamically (configurable up to MAX_SUMMER_CHANNELS).
    /// Must be called outside the audio thread.
    pub fn set_num_channels(&mut self, n: usize) {
        let n = n.clamp(1, MAX_SUMMER_CHANNELS);
        if self.previous.len() != n {
            self.previous = vec![0.0; n];
        }
        self.num_channels = n;
    }

    /// Returns the currently configured number of channels.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn process(&mut self, channels: &mut [&mut [f64]], num_samples: usize) {
        // Silence fast-path: if all inputs are silent and previous state has decayed, skip.
        let has_signal = channels.iter().take(self.previous.len()).any(|c| {
            c.iter().take(num_samples).any(|&x| x.abs() > 1.0e-20)
        });
        if !has_signal && self.previous.iter().all(|&p| p.abs() < 1.0e-20) {
            for ch in channels.iter_mut().take(self.previous.len()) {
                for sample in ch.iter_mut().take(num_samples) {
                    *sample = 0.0;
                }
            }
            return;
        }
        let nch = channels.len().min(self.previous.len()).max(1);
        let len = (0..nch)
            .map(|i| channels.get(i).map(|c| c.len()).unwrap_or(0))
            .min()
            .unwrap_or(0)
            .min(num_samples);
        let mut x: [f64; MAX_SUMMER_CHANNELS] = [0.0; MAX_SUMMER_CHANNELS];
        let mut y: [f64; MAX_SUMMER_CHANNELS] = [0.0; MAX_SUMMER_CHANNELS];
        let atten = if self.crosstalk_db <= 0.0 { 0.0 } else { 10.0f64.powf(-self.crosstalk_db / 20.0) };
        for k in 0..len {
            for i in 0..nch {
                let cur = channels.get(i).and_then(|c| c.get(k)).copied().unwrap_or(0.0);
                x[i] = cur;
                y[i] = cur - self.coupling_alpha * self.previous[i];
                self.previous[i] = cur;
            }
            for i in 0..nch {
                let mut val = x[i];
                if i > 0 { val -= y[i - 1] * atten; }
                if i + 1 < nch { val -= y[i + 1] *atten; }
                if let Some(v) = channels.get_mut(i).and_then(|c| c.get_mut(k)) {
                    *v = if v.is_finite() { val } else { 0.0 };
                }
            }
        }
    }
}

pub struct VirtualEarthSummer {
    max_channels: usize,
    sample_rate: f64,
    bus_drive: f64,
    psu_sag_amount: f64,
    supply_voltage: f64,
    saturation_state: f64,
}

impl VirtualEarthSummer {
    pub fn new(max_channels: usize, sample_rate: f64) -> Self {
        let n = max_channels.min(MAX_SUMMER_CHANNELS).max(1);
        Self {
            max_channels: n,
            sample_rate: if sample_rate > 0.0 { sample_rate } else { 48_000.0 },
            bus_drive: 0.0,
            psu_sag_amount: 0.0,
            supply_voltage: 1.0,
            saturation_state: 0.0,
        }
    }

    pub fn set_bus_drive(&mut self, drive_db: f64) {
        self.bus_drive = drive_db.clamp(-24.0, 24.0);
    }

    pub fn set_psu_sag(&mut self, sag_amount: f64) {
        self.psu_sag_amount = sag_amount.clamp(0.0, 1.0);
    }

    pub fn initialize(&mut self, sample_rate: f64) {
        self.sample_rate = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        self.max_channels = self.max_channels.min(MAX_SUMMER_CHANNELS).max(1);
    }

    pub fn reset(&mut self) {
        self.supply_voltage = 1.0;
        self.saturation_state = 0.0;
    }

    pub fn process_sum(
        &mut self,
        stereo_pairs: &[(&[f64], &[f64])],
        out_left: &mut [f64],
        out_right: &mut [f64],
        num_samples: usize,
    ) {
        // Silence fast-path: if no input signal and PSU/saturation at rest, output zeros.
        let has_signal = stereo_pairs.iter().take(self.max_channels / 2).any(|(l, r)| {
            l.iter().take(num_samples).any(|&x| x.abs() > 1.0e-20)
                || r.iter().take(num_samples).any(|&x| x.abs() > 1.0e-20)
        });
        if !has_signal && (self.supply_voltage - 1.0).abs() < 1.0e-6 && self.saturation_state < 1.0e-6 {
            let len = num_samples.min(out_left.len()).min(out_right.len());
            out_left[..len].fill(0.0);
            out_right[..len].fill(0.0);
            return;
        }
        let pairs = stereo_pairs.len().min(self.max_channels / 2).max(1);
        let len = num_samples.min(out_left.len()).min(out_right.len());
        let drive_lin = 10.0f64.powf(self.bus_drive / 20.0);
        let tau = 0.010;
        let alpha = 1.0 - (-1.0 / (tau * self.sample_rate)).exp();
        for n in 0..len {
            let mut sum_l = 0.0_f64;
            let mut sum_r = 0.0_f64;
            for (l, r) in stereo_pairs.iter().take(pairs) {
                sum_l += l.get(n).copied().unwrap_or(0.0);
                sum_r += r.get(n).copied().unwrap_or(0.0);
            }
            sum_l *= drive_lin;
            sum_r *= drive_lin;
            let total_current = (sum_l * sum_l + sum_r * sum_r).sqrt();
            let target_voltage = 1.0 - self.psu_sag_amount * total_current.tanh();
            self.supply_voltage += alpha * (target_voltage - self.supply_voltage);
            self.supply_voltage = self.supply_voltage.clamp(0.1, 1.0);
            sum_l *= self.supply_voltage;
            sum_r *= self.supply_voltage;
            let sat_input_l = sum_l.tanh();
            let sat_input_r = sum_r.tanh();
            self.saturation_state += 0.01 * (sat_input_l.abs().max(sat_input_r.abs()) - self.saturation_state);
            let saturation_gain = (1.0 - self.saturation_state * 0.3).max(0.5);
            let out_l = sat_input_l * saturation_gain;
            let out_r = sat_input_r * saturation_gain;
            out_left[n] = if out_l.is_finite() { out_l } else { 0.0 };
            out_right[n] = if out_r.is_finite() { out_r } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crosstalk_matrix_off_by_default() {
        let mut m = AdjacentCrosstalkMatrix::new(0.0, 48_000.0);
        let mut ch0 = vec![0.5_f64; 64];
        let mut ch1 = vec![0.5_f64; 64];
        let mut ch2 = vec![0.5_f64; 64];
        let mut channels: Vec<&mut [f64]> = vec![&mut ch0, &mut ch1, &mut ch2];
        m.process(&mut channels, 64);
        assert!((ch0[0] - 0.5).abs() < 1e-9);
        assert!((ch1[0] - 0.5).abs() < 1e-9);
        assert!((ch2[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn crosstalk_matrix_reduces_adjacent_channel() {
        let mut m = AdjacentCrosstalkMatrix::new(20.0, 48_000.0);
        let mut ch0 = vec![0.0_f64; 64];
        let mut ch1 = vec![1.0_f64; 64];
        let mut ch2 = vec![0.0_f64; 64];
        let mut channels: Vec<&mut [f64]> = vec![&mut ch0, &mut ch1, &mut ch2];
        m.process(&mut channels, 64);
        // Crosstalk from ch1 bleeds into ch0 and ch2 (attenuated by 20 dB ~ 0.1)
        // High-pass coupling can produce negative values, check absolute value
        assert!(ch0[0].abs() > 0.0, "ch0 should receive crosstalk");
        assert!(ch0[0].abs() < 0.2, "ch0 crosstalk should be attenuated");
        assert!(ch2[0].abs() > 0.0, "ch2 should receive crosstalk");
        assert!(ch2[0].abs() < 0.2, "ch2 crosstalk should be attenuated");
    }

    #[test]
    fn virtual_earth_summer_sums_channels() {
        let mut summer = VirtualEarthSummer::new(32, 48_000.0);
        let left_in = vec![0.5_f64; 64];
        let right_in = vec![0.5_f64; 64];
        let mut out_l = vec![0.0_f64; 64];
        let mut out_r = vec![0.0_f64; 64];
        summer.process_sum(&[(&left_in, &right_in)], &mut out_l, &mut out_r, 64);
        assert!(out_l[0] > 0.0);
        assert!(out_r[0] > 0.0);
        assert!(out_l[0].is_finite());
        assert!(out_r[0].is_finite());
    }

    #[test]
    fn virtual_earth_summer_psu_sag_with_many_inputs() {
        let mut summer = VirtualEarthSummer::new(32, 48_000.0);
        summer.set_psu_sag(1.0);
        let left_in = vec![0.5_f64; 64];
        let right_in = vec![0.5_f64; 64];
        let mut out_l = vec![0.0_f64; 64];
        let mut out_r = vec![0.0_f64; 64];
        summer.process_sum(&[(&left_in, &right_in)], &mut out_l, &mut out_r, 64);
        assert!(out_l[0] > 0.0);
        assert!(out_l[0] < 0.5);
        assert!(out_l[0].is_finite());
    }

    #[test]
    fn virtual_earth_summer_silence_at_rest() {
        let mut summer = VirtualEarthSummer::new(32, 48_000.0);
        let left_in = vec![0.0_f64; 64];
        let right_in = vec![0.0_f64; 64];
        let mut out_l = vec![0.0_f64; 64];
        let mut out_r = vec![0.0_f64; 64];
        summer.process_sum(&[(&left_in, &right_in)], &mut out_l, &mut out_r, 64);
        for v in &out_l { assert_eq!(*v, 0.0); }
        for v in &out_r { assert_eq!(*v, 0.0); }
    }
}
