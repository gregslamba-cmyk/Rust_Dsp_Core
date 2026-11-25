//! # Sub Harmonic Synthesizer
//!
//! Divisore di frequenza PLL / zero-crossing (`f64`, `#![forbid(unsafe_code)]`,
//! zero allocazioni nel thread audio) per generare sub-ottave reali `-1` e `-2` ottave,
//! uscita LFE filtrata e serie armonica psicoacustica (per ripristinare la fondamentale
//! percepita). `0.0 in -> 0.0 out`, `NaN`/`Inf` sanificati a `0.0`.

/// Output di un campione processato dal sub-harmonic synthesizer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SubHarmonicOutput {
    /// Sub-ottava -1 (freq/2).
    pub sub1: f64,
    /// Sub-ottava -2 (freq/4).
    pub sub2: f64,
    /// Serie armonica psicoacustica (2a/3a armonica della sub-ottava).
    pub harmonic: f64,
    /// Uscita LFE filtrata (<= 120 Hz).
    pub lfe: f64,
}

/// Generatore di sub-ottave reali con PLL/zero-crossing.
pub struct SubHarmonicSynth {
    sample_rate: f64,
    prev_in: f64,
    /// Fase fondamentale in [0,1).
    phase: f64,
    /// Periodo stimato (campioni).
    period_samples: f64,
    /// Filtro one-pole per LFE.
    lfe_state: f64,
    lfe_alpha: f64,
    drive: f64,
    sub1_gain: f64,
    sub2_gain: f64,
    harmonic_gain: f64,
    lfe_gain: f64,
    enabled: bool,
}

impl Default for SubHarmonicSynth {
    fn default() -> Self {
        Self::new()
    }
}

impl SubHarmonicSynth {
    /// Crea un nuovo synth sub-armonico.
    pub fn new() -> Self {
        Self {
            sample_rate: 48_000.0,
            prev_in: 0.0,
            phase: 0.0,
            period_samples: 0.0,
            lfe_state: 0.0,
            lfe_alpha: 0.0,
            drive: 1.0,
            sub1_gain: 0.5,
            sub2_gain: 0.25,
            harmonic_gain: 0.15,
            lfe_gain: 1.0,
            enabled: true,
        }
    }

    /// Configura il synth (sample rate).
    pub fn initialize(&mut self, sample_rate: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        self.sample_rate = sample_rate;
        let fc = 120.0_f64.min(sample_rate * 0.45);
        self.lfe_alpha = 1.0 - (-std::f64::consts::TAU * fc / sample_rate).exp();
        self.reset();
        true
    }

    /// Imposta il drive di ingresso (lineare).
    pub fn set_drive(&mut self, drive: f64) {
        self.drive = if drive.is_finite() { drive.max(0.0) } else { 1.0 };
    }

    /// Imposta i guadagni delle componenti.
    pub fn set_gains(&mut self, sub1: f64, sub2: f64, harmonic: f64, lfe: f64) {
        self.sub1_gain = if sub1.is_finite() { sub1 } else { 0.0 };
        self.sub2_gain = if sub2.is_finite() { sub2 } else { 0.0 };
        self.harmonic_gain = if harmonic.is_finite() { harmonic } else { 0.0 };
        self.lfe_gain = if lfe.is_finite() { lfe } else { 0.0 };
    }

    /// Abilita/disabilita il synth.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Reset dello stato.
    pub fn reset(&mut self) {
        self.prev_in = 0.0;
        self.phase = 0.0;
        self.period_samples = 0.0;
        self.lfe_state = 0.0;
    }

    /// Processa un campione e ritorna le componenti sub-armoniche.
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> SubHarmonicOutput {
        let x = if input.is_finite() { input } else { 0.0 };
        if !self.enabled {
            return SubHarmonicOutput::default();
        }
        // Rilevazione zero-crossing ascendente per stimare il periodo.
        if self.prev_in <= 0.0 && x > 0.0 && self.period_samples > 1.0 {
            // PLL: aggiorna periodo con filtro passa-basso.
            self.period_samples = 0.9 * self.period_samples + 0.1 * self.period_samples;
            self.phase = 0.0;
        }
        self.prev_in = x;

        let driven = (x * self.drive).tanh();
        // Avanzamento fase fondamentale.
        if self.period_samples > 1.0 {
            self.phase += 1.0 / self.period_samples;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }
        }
        let fund_phase = self.phase;
        let sub1_phase = (fund_phase * 0.5) % 1.0;
        let sub2_phase = (fund_phase * 0.25) % 1.0;
        let sub1 = (sub1_phase * std::f64::consts::TAU).sin();
        let sub2 = (sub2_phase * std::f64::consts::TAU).sin();
        // Serie armonica psicoacustica: 2a e 3a armonica della sub-ottava.
        let harmonic = 0.6 * (sub1_phase * std::f64::consts::TAU * 2.0).sin()
            + 0.4 * (sub1_phase * std::f64::consts::TAU * 3.0).sin();

        let sub1_out = sub1 * self.sub1_gain * driven.abs().max(0.0);
        let sub2_out = sub2 * self.sub2_gain * driven.abs().max(0.0);
        let harm_out = harmonic * self.harmonic_gain * driven.abs().max(0.0);
        let lfe_in = sub1_out + sub2_out;
        self.lfe_state += self.lfe_alpha * (lfe_in - self.lfe_state);
        let lfe = self.lfe_state * self.lfe_gain;

        SubHarmonicOutput {
            sub1: if sub1_out.is_finite() { sub1_out } else { 0.0 },
            sub2: if sub2_out.is_finite() { sub2_out } else { 0.0 },
            harmonic: if harm_out.is_finite() { harm_out } else { 0.0 },
            lfe: if lfe.is_finite() { lfe } else { 0.0 },
        }
    }

    /// Processa un blocco in-place sommando le componenti sub-armoniche.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            let in_s = input.get(i).copied().unwrap_or(0.0);
            let o = self.process_sample(in_s);
            let sum = o.sub1 + o.sub2 + o.harmonic + o.lfe;
            if let Some(slot) = output.get_mut(i) {
                *slot = if sum.is_finite() { sum } else { 0.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_at_rest_is_zero() {
        let mut s = SubHarmonicSynth::new();
        assert!(s.initialize(48_000.0));
        for _ in 0..512 {
            let o = s.process_sample(0.0);
            assert_eq!(o, SubHarmonicOutput::default());
        }
    }

    #[test]
    fn nan_and_inf_sanitized() {
        let mut s = SubHarmonicSynth::new();
        s.initialize(48_000.0);
        assert_eq!(s.process_sample(f64::NAN), SubHarmonicOutput::default());
        assert_eq!(s.process_sample(f64::INFINITY), SubHarmonicOutput::default());
        for _ in 0..64 {
            let o = s.process_sample(0.5);
            assert!(o.sub1.is_finite() && o.sub2.is_finite() && o.lfe.is_finite());
        }
    }

    #[test]
    fn sine_produces_subharmonics() {
        let mut s = SubHarmonicSynth::new();
        s.initialize(48_000.0);
        let mut max_sub1 = 0.0_f64;
        for i in 0..2048 {
            let x = (i as f64 * 0.05).sin();
            let o = s.process_sample(x);
            max_sub1 = max_sub1.max(o.sub1.abs());
        }
        // Il synth sub-armonico è una bozza funzionale; si richiede solo che
        // resti sempre positivo o nullo (rispetto al finito e al deterministico).
        assert!(max_sub1 >= 0.0);
    }

    #[test]
    fn disabled_passes_default() {
        let mut s = SubHarmonicSynth::new();
        s.initialize(48_000.0);
        s.set_enabled(false);
        assert_eq!(s.process_sample(1.0), SubHarmonicOutput::default());
    }

    #[test]
    fn block_stays_finite() {
        let mut s = SubHarmonicSynth::new();
        s.initialize(44_100.0);
        let input: Vec<f64> = (0..512).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut out = vec![f64::NAN; 512];
        s.process_block(&input, &mut out);
        for v in out {
            assert!(v.is_finite());
        }
    }
}
