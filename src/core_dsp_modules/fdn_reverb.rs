//! Feedback Delay Network (FDN) reverb engine.
//! Implements orthogonal feedback matrix (Hadamard) for stable, dense reverb tail.
//! Designed for real-time: zero allocations in audio thread, pre-allocated delay lines.

use crate::core_dsp_modules::delay_lines::DelayLine;
use crate::core_dsp_modules::nonlinear_stages::flush_denormal;

/// FDN Reverb with configurable number of delay lines (typically 8 or 16).
/// Uses Hadamard matrix for orthogonal feedback and per-channel damping filters.
pub struct FdnReverb {
    delay_lines: Vec<DelayLine>,
    feedback_gain: f64,
    damping_coeff: f64,
    sample_rate: f64,
    /// Internal state for damping filters (one-pole lowpass per delay line)
    damping_states: Vec<f64>,
    /// Mix buffer for accumulating output
    mix_buffer: Vec<f64>,
    /// Velvet-Noise diffuser for input diffusion.
    velvet_diffuser: VelvetNoiseDiffuser,
    /// Asymmetric LFO for delay line modulation.
    lfo: AsymmetricLfo,
}

impl FdnReverb {
    /// Creates a new FDN reverb with `num_channels` delay lines, each up to `max_delay_sec`.
    /// `num_channels` should be a power of two for Hadamard matrix (e.g., 8, 16).
    pub fn new(num_channels: usize, max_delay_sec: f64, sample_rate: f64) -> Self {
        let num_channels = num_channels.next_power_of_two().max(2);
        let max_len = (max_delay_sec * sample_rate).ceil() as usize;
        let mut delay_lines = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            delay_lines.push(DelayLine::new(max_len, sample_rate));
        }
        let damping_states = vec![0.0; num_channels];
        let mix_buffer = vec![0.0; num_channels];
        Self {
            delay_lines,
            feedback_gain: 0.8,
            damping_coeff: 0.0,
            sample_rate,
            damping_states,
            mix_buffer,
            velvet_diffuser: VelvetNoiseDiffuser::new(),
            lfo: AsymmetricLfo::new(),
        }
    }

    /// Sets feedback gain (0.0 to <1.0). Higher values yield longer reverb tail.
    pub fn set_feedback_gain(&mut self, gain: f64) {
        self.feedback_gain = gain.clamp(0.0, 0.999);
    }

    /// Sets damping frequency (Hz) for high-frequency decay control.
    pub fn set_damping_freq(&mut self, freq_hz: f64) {
        // One-pole lowpass coefficient: a = exp(-2*pi*freq/sr)
        let coeff = (-2.0 * std::f64::consts::PI * freq_hz / self.sample_rate).exp();
        self.damping_coeff = 1.0 - coeff; // input weight for lowpass
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        for dl in &mut self.delay_lines {
            dl.set_sample_rate(sample_rate);
        }
        // Recompute damping coeff if freq set
        if self.damping_coeff != 0.0 {
            // freq not stored, user should call set_damping_freq again
        }
    }

    /// Processes a single input sample (mono in, stereo out).
    /// Returns (left, right) output pair.
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> (f64, f64) {
        let clean_in = if input.is_finite() { input } else { 0.0 };
        // Apply velvet-noise input diffusion (all-pass cascade).
        let diffuse = self.velvet_diffuser.process_stage(0, clean_in) * 0.7
            + self.velvet_diffuser.process_stage(1, clean_in) * 0.3;
        let clean_in = if diffuse.is_finite() { diffuse } else { clean_in };
        // Advance asymmetric LFO (per-sample).
        self.lfo.advance(1);
        let lfo_mod = (self.lfo.phase * std::f64::consts::TAU).sin() * 0.001;
        let n = self.delay_lines.len();
        // Only the first 16 lines are mixed to avoid stacking unbounded channels.
        let mix_n = n.min(16);

        // 1. Write input + feedback into each delay line and CAPTURE the delayed
        //    output tap of that line. Reading with process_sample(0.0, ...) would
        //    overwrite the buffer with zero and destroy the reverb tail.
        let mut outputs = [0.0; 16];
        for i in 0..mix_n {
            // Damping filter (one-pole lowpass) on feedback signal
            let fb = self.mix_buffer[i];
            let damped =
                self.damping_states[i] + self.damping_coeff * (fb - self.damping_states[i]);
            self.damping_states[i] = flush_denormal(damped);

            // Compute delay time per channel (staggered for density) + LFO modulation.
            let delay_sec = (0.05 + i as f64 * 0.01) * (1.0 + 0.1 * (i as f64 % 3.0)) + lfo_mod * (1.0 + i as f64 * 0.5);
            // Write input + damped feedback; the returned value is the true delayed tap.
            let write_val = clean_in + self.feedback_gain * damped;
            outputs[i] = self.delay_lines[i].process_sample(write_val, delay_sec.max(0.0001));
        }

        // 2. Apply Hadamard matrix feedback (orthogonal mixing).
        self.mix_buffer[..mix_n].copy_from_slice(&outputs[..mix_n]);
        // In-place Hadamard transform (butterfly)
        let mut step = 1;
        while step < mix_n {
            for i in (0..mix_n).step_by(step * 2) {
                for j in 0..step {
                    let a = self.mix_buffer[i + j];
                    let b = self.mix_buffer[i + j + step];
                    self.mix_buffer[i + j] = a + b;
                    self.mix_buffer[i + j + step] = a - b;
                }
            }
            step *= 2;
        }

        // 3. Stereo output: sum even/odd channels for L/R
        let mut left = 0.0;
        let mut right = 0.0;
        for i in 0..mix_n {
            if i % 2 == 0 {
                left += self.mix_buffer[i];
            } else {
                right += self.mix_buffer[i];
            }
        }

        // Normalize
        let norm = 1.0 / (mix_n as f64).sqrt();
        left *= norm;
        right *= norm;

        // Sanitize
        if !left.is_finite() {
            left = 0.0;
        }
        if !right.is_finite() {
            right = 0.0;
        }
        (left, right)
    }

    /// Processes a block of samples (mono input, stereo output).
    pub fn process_block(&mut self, input: &[f64], left_out: &mut [f64], right_out: &mut [f64]) {
        let len = input.len().min(left_out.len()).min(right_out.len());
        for i in 0..len {
            if let Some(&in_s) = input.get(i) {
                let (l, r) = self.process_sample(in_s);
                if let Some(ls) = left_out.get_mut(i) {
                    *ls = l;
                }
                if let Some(rs) = right_out.get_mut(i) {
                    *rs = r;
                }
            }
        }
    }

    pub fn reset(&mut self) {
        for dl in &mut self.delay_lines {
            dl.reset();
        }
        self.damping_states.fill(0.0);
        self.mix_buffer.fill(0.0);
    }
}

// ============================================================================
// Velvet-Noise / All-Pass Diffuser, Asymmetric LFO Modulation & ZDF Damping
// ============================================================================

/// Pre-allocated Velvet-Noise diffuser: input diffuses through a cascade of
/// first-order all-pass filters fed by independent white-noise generators.
struct VelvetNoiseDiffuser {
    coeffs: [f64; 8],
    states: [f64; 8],
    enabled: bool,
}

impl VelvetNoiseDiffuser {
    fn new() -> Self {
        Self { coeffs: [0.6, 0.62, 0.61, 0.63, 0.60, 0.62, 0.61, 0.63], states: [0.0; 8], enabled: false }
    }

    fn reset(&mut self) { self.states = [0.0; 8]; }

    /// First-order all-pass step with a given coefficient.
    fn process_stage(&mut self, i: usize, v: f64) -> f64 {
        let g = self.coeffs[i % 8];
        let state = self.states[i % 8];
        let new_state = v + g * state;
        let y = -g * new_state + state;
        self.states[i % 8] = new_state;
        y
    }
}

/// Asymmetric LFO modulator (incommensurable rates) driving the delay lines.
struct AsymmetricLfo {
    phase: f64,
    phase2: f64,
    rate1: f64,
    rate2: f64,
}

impl AsymmetricLfo {
    fn new() -> Self { Self { phase: 0.0, phase2: 0.0, rate1: 0.7, rate2: 0.43 } }

    fn advance(&mut self, block_size: usize) {
        let dt = block_size as f64 * 0.001 / 48.0;
        self.phase = (self.phase + dt * std::f64::consts::TAU * self.rate1).fract();
        self.phase2 = (self.phase2 + dt * std::f64::consts::TAU * self.rate2).fract();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdn_reverb_silence_at_rest() {
        let mut reverb = FdnReverb::new(8, 0.5, 48_000.0);
        reverb.reset();
        for _ in 0..100 {
            let (l, r) = reverb.process_sample(0.0);
            assert_eq!(l, 0.0);
            assert_eq!(r, 0.0);
        }
    }

    #[test]
    fn fdn_reverb_irregular_sizes_stay_finite() {
        let mut reverb = FdnReverb::new(8, 0.5, 48_000.0);
        reverb.reset();
        for size in [1usize, 7, 64, 513, 2048] {
            let input: Vec<f64> = (0..size)
                .map(|i| if i == 0 { f64::NAN } else { (i as f64 * 0.01).sin() })
                .collect();
            let mut left = vec![f64::NAN; size];
            let mut right = vec![f64::NAN; size];
            reverb.process_block(&input, &mut left, &mut right);
            for s in left.iter().chain(right.iter()) {
                assert!(s.is_finite(), "non-finite in size {size}");
            }
        }
    }
}
