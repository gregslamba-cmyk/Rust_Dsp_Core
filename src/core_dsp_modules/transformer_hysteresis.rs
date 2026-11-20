/// Default split between the saturating low band and the transparent high band.
const DEFAULT_CROSSOVER_HZ: f64 = 180.0;
/// Corner of the output DC blocker that removes the static bias of the asymmetric core.
const DC_BLOCK_HZ: f64 = 5.0;

#[derive(Clone, Copy, Debug)]
pub struct TransformerHysteresis {
    flux: f64,
    low_state: f64,
    dc_state: f64,
    last_output: f64,
    low_alpha: f64,
    dc_alpha: f64,
    sample_rate: f64,
    max_slew_per_second: f64,
}

impl Default for TransformerHysteresis {
    fn default() -> Self {
        let mut transformer = Self {
            flux: 0.0,
            low_state: 0.0,
            dc_state: 0.0,
            last_output: 0.0,
            low_alpha: 0.0,
            dc_alpha: 0.0,
            sample_rate: 0.0,
            max_slew_per_second: 0.0,
        };
        transformer.configure(48_000.0, DEFAULT_CROSSOVER_HZ);
        transformer
    }
}

impl TransformerHysteresis {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the band split and derives every coefficient from the real sample rate.
    /// The high band stays linear ("silky highs"); only the low band is saturated.
    pub fn configure(&mut self, sample_rate: f64, crossover_hz: f64) -> bool {
        if !sample_rate.is_finite()
            || sample_rate <= 0.0
            || !crossover_hz.is_finite()
            || crossover_hz <= 0.0
            || crossover_hz >= sample_rate * 0.5
        {
            return false;
        }

        let warped = (std::f64::consts::PI * crossover_hz / sample_rate).tan();
        self.low_alpha = warped / (1.0 + warped);
        self.dc_alpha = 1.0 - (-std::f64::consts::TAU * DC_BLOCK_HZ / sample_rate).exp();
        self.sample_rate = sample_rate;
        self.reset();
        true
    }

    /// Enables an optional analog-style slew limit in units per second. Zero disables it.
    pub fn set_slew_limit(&mut self, max_units_per_second: f64) {
        self.max_slew_per_second = if max_units_per_second.is_finite() {
            max_units_per_second.max(0.0)
        } else {
            0.0
        };
    }

    #[inline]
    pub fn process(
        &mut self,
        input: f64,
        drive: f64,
        hysteresis: f64,
        frequency_dependency: f64,
    ) -> f64 {
        if !input.is_finite()
            || !drive.is_finite()
            || !hysteresis.is_finite()
            || !frequency_dependency.is_finite()
            || self.low_alpha <= 0.0
        {
            self.reset();
            return 0.0;
        }

        let hysteresis = hysteresis.clamp(0.0, 0.985);
        let bass_weight = frequency_dependency.clamp(0.0, 1.0);

        // Two-band split: the low band carries the core and its harmonic weight,
        // the high band is only gently limited so transients never turn glassy.
        self.low_state += self.low_alpha * (input - self.low_state);
        let low = self.low_state;
        let high = input - low;

        let bass_drive = drive * (1.0 + 1.5 * bass_weight);
        let bias = 0.12 * bass_weight;
        let bias_tanh = bias.tanh();
        self.flux = low * bass_drive + self.flux * hysteresis;
        // Zero-mean asymmetric core: at silence the output is exactly zero, while the
        // bias still provides the warm 2nd-harmonic content of a Class-A stage.
        let core = (self.flux + bias).tanh() - bias_tanh;
        self.flux = core * (1.0 - hysteresis);
        let core_norm = (bass_drive * (1.0 - bias_tanh * bias_tanh)).max(1.0e-4);
        let low_out = core / core_norm;

        let high_drive = 1.0 + 0.15 * (drive - 1.0).max(0.0) * (1.0 - 0.5 * bass_weight);
        let high_out = (high * high_drive).tanh() / high_drive;

        let mixed = low_out + high_out;

        // Remove the static bias of the asymmetric core while keeping its 2nd harmonic.
        self.dc_state += self.dc_alpha * (mixed - self.dc_state);
        let mut output = mixed - self.dc_state;

        if self.max_slew_per_second > 0.0 && self.sample_rate > 0.0 {
            let max_delta = self.max_slew_per_second / self.sample_rate;
            let delta = output - self.last_output;
            output = self.last_output + delta.clamp(-max_delta, max_delta);
        }
        self.last_output = output;

        if output.is_finite() {
            output
        } else {
            self.reset();
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.flux = 0.0;
        self.low_state = 0.0;
        self.dc_state = 0.0;
        self.last_output = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::TransformerHysteresis;

    const SAMPLE_RATE: f64 = 48_000.0;

    fn rms(signal: &[f64]) -> f64 {
        let energy: f64 = signal.iter().map(|sample| sample * sample).sum();
        (energy / signal.len() as f64).sqrt()
    }

    fn render(frequency: f64, drive: f64, bass_weight: f64) -> Vec<f64> {
        let mut transformer = TransformerHysteresis::new();
        assert!(transformer.configure(SAMPLE_RATE, super::DEFAULT_CROSSOVER_HZ));
        let length = (SAMPLE_RATE as usize) / 10;
        let mut output = Vec::with_capacity(length);
        for index in 0..length {
            let phase = std::f64::consts::TAU * frequency * index as f64 / SAMPLE_RATE;
            output.push(transformer.process(0.1 * phase.sin(), drive, 0.3, bass_weight));
        }
        output
    }

    #[test]
    fn test_zero_silence() {
        let mut th = TransformerHysteresis::new();
        for _ in 0..50 {
            assert_eq!(th.process(0.0, 1.0, 0.5, 0.1), 0.0);
        }
    }

    #[test]
    fn test_stability() {
        let mut th = TransformerHysteresis::new();
        for i in 0..1000 {
            let x = (i as f64 * 0.01).sin();
            let out = th.process(x, 2.0, 0.8, 0.2);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn invalid_sample_rate_is_rejected() {
        let mut th = TransformerHysteresis::new();
        assert!(!th.configure(0.0, 180.0));
        assert!(!th.configure(48_000.0, 30_000.0));
        assert!(th.configure(96_000.0, 200.0));
    }

    #[test]
    fn bass_is_compressed_while_highs_stay_near_unity() {
        let low = render(100.0, 4.0, 1.0);
        let low_unity = render(100.0, 1.0, 1.0);
        let low_ratio =
            rms(&low[SAMPLE_RATE as usize / 20..]) / rms(&low_unity[SAMPLE_RATE as usize / 20..]);
        assert!(
            low_ratio < 0.95,
            "low band should be compressed: {low_ratio}"
        );

        let high = render(10_000.0, 4.0, 1.0);
        let high_unity = render(10_000.0, 1.0, 1.0);
        let high_ratio =
            rms(&high[SAMPLE_RATE as usize / 20..]) / rms(&high_unity[SAMPLE_RATE as usize / 20..]);
        assert!(
            high_ratio > 0.95,
            "high band must stay linear: {high_ratio}"
        );
    }

    #[test]
    fn slew_limit_is_sample_rate_scaled() {
        let mut fast = TransformerHysteresis::new();
        assert!(fast.configure(48_000.0, super::DEFAULT_CROSSOVER_HZ));
        fast.set_slew_limit(48_000.0); // 1.0 per sample at 48 kHz

        let mut slow = TransformerHysteresis::new();
        assert!(slow.configure(96_000.0, super::DEFAULT_CROSSOVER_HZ));
        slow.set_slew_limit(48_000.0); // 0.5 per sample at 96 kHz

        let fast_step = fast.process(10.0, 1.0, 0.0, 0.0);
        let slow_step = slow.process(10.0, 1.0, 0.0, 0.0);
        assert!((fast_step - 1.0).abs() < 1.0e-6);
        assert!((slow_step - 0.5).abs() < 1.0e-6);
    }
}
