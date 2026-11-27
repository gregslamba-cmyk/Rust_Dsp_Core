//! # Surgical Equalizer Engine (Minimum Phase & Linear Phase)
//!
//! Questo modulo fornisce il motore di equalizzazione parametrica a 10 bande per il plugin `greg-surgical-eq`,
//! con commutazione tra due modalità di fase:
//!
//! ## Modalità di Fase Supportate
//! 1. **`MinimumPhase` (Fase Minima - Latenza Zero):**
//!    Cascata seriale di filtri biquad IIR a 2 poli e 2 zeri. Preserva la risposta immediata priva di latenza,
//!    ideale per tracking e registrazione live, con rotazione di fase analog-style.
//! 2. **`LinearPhase` (Fase Lineare FIR Simmetrico):**
//!    Sintesi analitica di un filtro FIR simmetrico di tipo I (con lunghezza dispari, es. 257 tap)
//!    ottenuto per campionamento in frequenza e IFFT della curva di ampiezza composita di tutte le bande attive.
//!    - **Vantaggio Acustico:** Fase perfettamente lineare, zero alterazione delle relazioni di fase tra microfoni multipli
//!      (es. batteria e riprese acustiche complesse).
//!    - **Latenza Dichiarata Esatta:** Ritardo di gruppo rigorosamente costante pari a $M = (N - 1) / 2$ campioni (es. 128 campioni a 257 tap).

use num_traits::Float;

use super::filters::{Biquad, FilterType};

const MAX_LINEAR_PHASE_TAPS: usize = 513;

/// Parametri descrittivi per una singola banda di equalizzazione parametrica.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EqBand<T: Float> {
    /// Flag di attivazione della banda.
    pub enabled: bool,
    /// Tipologia di filtro (Bell, LowShelf, HighShelf, HighPass, LowPass, Notch, BandPass).
    pub kind: FilterType,
    /// Frequenza di taglio o centrale in Hz (range: 20.0 .. 20_000.0 Hz).
    pub frequency_hz: T,
    /// Fattore di merito Q o ampiezza di banda (range: 0.1 .. 30.0).
    pub q: T,
    /// Guadagno della banda in dB (range: -24.0 .. +24.0 dB).
    pub gain_db: T,
}

impl<T: Float> Default for EqBand<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            kind: FilterType::Bell,
            frequency_hz: T::from(1_000.0).unwrap_or(T::zero()),
            q: T::from(0.707).unwrap_or(T::zero()),
            gain_db: T::zero(),
        }
    }
}

/// Modalità di fase dell'equalizzatore parametrico.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqPhaseMode {
    /// Fase minima analog-style: latenza zero campioni.
    MinimumPhase,
    /// Fase lineare FIR: preserva le relazioni di fase, con latenza algoritmica fissa (N-1)/2.
    LinearPhase,
}

#[derive(Clone, Copy, Debug)]
pub struct LinearPhaseFir<const TAPS: usize, T: Float> {
    coefficients: [T; TAPS],
    history: [T; TAPS],
    cursor: usize,
}

impl<const TAPS: usize, T: Float> Default for LinearPhaseFir<TAPS, T> {
    fn default() -> Self {
        let mut coefficients = [T::zero(); TAPS];
        if TAPS > 0 {
            coefficients[TAPS / 2] = T::one();
        }
        Self {
            coefficients,
            history: [T::zero(); TAPS],
            cursor: 0,
        }
    }
}

impl<const TAPS: usize, T: Float> LinearPhaseFir<TAPS, T> {
    fn configure(
        &mut self,
        sample_rate: T,
        bands: &[EqBand<T>],
        minimum_phase_filters: &[Biquad<T>],
    ) -> bool {
        if !sample_rate.is_finite()
            || sample_rate <= T::zero()
            || TAPS < 3
            || TAPS > MAX_LINEAR_PHASE_TAPS
            || TAPS.is_multiple_of(2)
            || bands.len() != minimum_phase_filters.len()
        {
            return false;
        }

        let center = TAPS / 2;
        let tap_count = T::from(TAPS).unwrap_or(T::zero());
        let mut magnitudes = [T::zero(); MAX_LINEAR_PHASE_TAPS / 2 + 1];
        for (bin, magnitude) in magnitudes[..=center].iter_mut().enumerate() {
            let omega = T::from(std::f64::consts::TAU * bin as f64 / TAPS as f64).unwrap_or(T::zero());
            let mut response = T::one();
            for (band, filter) in bands.iter().zip(minimum_phase_filters) {
                if band.enabled {
                    response = response * filter.magnitude_at(omega);
                }
            }
            if !response.is_finite() {
                return false;
            }
            *magnitude = response;
        }

        let two = T::one() + T::one();
        let pi = T::from(std::f64::consts::PI).unwrap_or(T::zero());
        let mut coefficients = [T::zero(); TAPS];
        for distance in 0..=center {
            let mut sum = magnitudes[0];
            for (bin, magnitude) in magnitudes.iter().enumerate().skip(1) {
                let angle = two * pi * T::from(bin * distance).unwrap_or(T::zero()) / tap_count;
                sum = sum + two * *magnitude * angle.cos();
            }
            let ideal = sum / tap_count;
            let normalized_distance = T::from(distance).unwrap_or(T::zero()) / T::from(center).unwrap_or(T::zero());
            let window = T::from(0.42).unwrap_or(T::zero())
                + T::from(0.5).unwrap_or(T::zero()) * (pi * normalized_distance).cos()
                + T::from(0.08).unwrap_or(T::zero()) * (two * pi * normalized_distance).cos();
            let coefficient = ideal * window;
            coefficients[center - distance] = coefficient;
            coefficients[center + distance] = coefficient;
        }

        let dc = coefficients
            .iter()
            .copied()
            .fold(T::zero(), |sum, value| sum + value);
        let target_dc = magnitudes[0];
        if dc.is_finite() && dc.abs() > T::from(1.0e-12).unwrap_or(T::zero()) && target_dc.is_finite() {
            if target_dc.abs() <= T::from(1.0e-6).unwrap_or(T::zero()) {
                coefficients[center] = coefficients[center] - dc;
            } else {
                let correction = target_dc / dc;
                for coefficient in &mut coefficients {
                    *coefficient = *coefficient * correction;
                }
            }
        } else if target_dc.abs() > T::from(1.0e-6).unwrap_or(T::zero()) {
            return false;
        }

        if coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
        {
            return false;
        }
        self.coefficients = coefficients;
        self.reset();
        true
    }

    #[inline]
    pub fn process(&mut self, input: T) -> T {
        if !input.is_finite() {
            self.reset();
            return T::zero();
        }
        self.history[self.cursor] = input;
        let mut output = T::zero();
        let mut index = self.cursor;
        for coefficient in &self.coefficients {
            output = output + *coefficient * self.history[index];
            index = if index == 0 { TAPS - 1 } else { index - 1 };
        }
        self.cursor = (self.cursor + 1) % TAPS;
        if output.is_finite() {
            output
        } else {
            self.reset();
            T::zero()
        }
    }

    pub fn reset(&mut self) {
        self.history.fill(T::zero());
        self.cursor = 0;
    }

    pub const fn latency_samples() -> u32 {
        (TAPS / 2) as u32
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SurgicalEq<const BANDS: usize, const TAPS: usize, T: Float> {
    minimum_phase_filters: [Biquad<T>; BANDS],
    enabled_bands: [bool; BANDS],
    linear_phase_filter: LinearPhaseFir<TAPS, T>,
    phase_mode: EqPhaseMode,
}

impl<const BANDS: usize, const TAPS: usize, T: Float> Default for SurgicalEq<BANDS, TAPS, T> {
    fn default() -> Self {
        Self {
            minimum_phase_filters: [Biquad::default(); BANDS],
            enabled_bands: [false; BANDS],
            linear_phase_filter: LinearPhaseFir::default(),
            phase_mode: EqPhaseMode::MinimumPhase,
        }
    }
}

impl<const BANDS: usize, const TAPS: usize, T: Float> SurgicalEq<BANDS, TAPS, T> {
    pub fn configure(&mut self, sample_rate: T, bands: &[EqBand<T>; BANDS]) -> bool {
        if !sample_rate.is_finite() || sample_rate <= T::zero() {
            return false;
        }
        for ((filter, enabled), band) in self
            .minimum_phase_filters
            .iter_mut()
            .zip(self.enabled_bands.iter_mut())
            .zip(bands)
        {
            *enabled = band.enabled;
            if band.enabled
                && !filter.configure(
                    sample_rate,
                    band.kind,
                    band.frequency_hz,
                    band.q,
                    band.gain_db,
                )
            {
                return false;
            }
            filter.reset();
        }
        self.linear_phase_filter
            .configure(sample_rate, bands, &self.minimum_phase_filters)
    }

    pub fn set_phase_mode(&mut self, phase_mode: EqPhaseMode) {
        if self.phase_mode != phase_mode {
            self.phase_mode = phase_mode;
            self.reset();
        }
    }

    #[inline]
    pub fn process(&mut self, input: T) -> T {
        match self.phase_mode {
            EqPhaseMode::MinimumPhase => self
                .minimum_phase_filters
                .iter_mut()
                .zip(self.enabled_bands)
                .filter(|(_, enabled)| *enabled)
                .fold(input, |sample, (filter, _)| filter.process(sample)),
            EqPhaseMode::LinearPhase => self.linear_phase_filter.process(input),
        }
    }

    pub fn reset(&mut self) {
        for filter in &mut self.minimum_phase_filters {
            filter.reset();
        }
        self.linear_phase_filter.reset();
    }

    pub const fn latency_samples(phase_mode: EqPhaseMode) -> u32 {
        match phase_mode {
            EqPhaseMode::MinimumPhase => 0,
            EqPhaseMode::LinearPhase => LinearPhaseFir::<TAPS, T>::latency_samples(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EqBand, EqPhaseMode, LinearPhaseFir, SurgicalEq};
    use crate::core_dsp_modules::filters::FilterType;

    const BANDS: usize = 2;
    const TAPS: usize = 257;

    fn bell_band(gain_db: f64) -> EqBand<f64> {
        EqBand {
            enabled: true,
            kind: FilterType::Bell,
            frequency_hz: 1_000.0,
            q: 1.0,
            gain_db,
        }
    }

    #[test]
    fn minimum_phase_eq_applies_a_bell_boost() {
        let mut eq = SurgicalEq::<BANDS, TAPS, f64>::default();
        let bands = [bell_band(9.0), EqBand::default()];
        assert!(eq.configure(48_000.0, &bands));
        eq.set_phase_mode(EqPhaseMode::MinimumPhase);

        let mut energy = 0.0;
        for sample_index in 0..48_000 {
            let input = (std::f64::consts::TAU * 1_000.0 * sample_index as f64 / 48_000.0).sin();
            let output = eq.process(input);
            assert!(output.is_finite());
            if sample_index > 4_000 {
                energy += output * output;
            }
        }
        let rms = (energy / 44_000.0).sqrt();
        assert!(rms > 1.5, "minimum-phase boost was too small: {rms}");
    }

    #[test]
    fn linear_phase_eq_is_symmetric_and_reports_group_delay() {
        let mut eq = SurgicalEq::<BANDS, TAPS, f64>::default();
        let bands = [bell_band(-6.0), EqBand::default()];
        assert!(eq.configure(48_000.0, &bands));
        eq.set_phase_mode(EqPhaseMode::LinearPhase);

        assert_eq!(
            SurgicalEq::<BANDS, TAPS, f64>::latency_samples(EqPhaseMode::MinimumPhase),
            0
        );
        assert_eq!(
            SurgicalEq::<BANDS, TAPS, f64>::latency_samples(EqPhaseMode::LinearPhase),
            128
        );

        let mut impulse = [0.0; TAPS + 8];
        impulse[0] = 1.0;
        let output: Vec<_> = impulse
            .into_iter()
            .map(|sample| eq.process(sample))
            .collect();
        for index in 0..TAPS {
            assert!((output[index] - output[TAPS - 1 - index]).abs() < 1.0e-10);
        }
        let peak_index = output
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
            .map(|(index, _)| index)
            .unwrap_or(0);
        assert_eq!(peak_index, 128);
        assert!(output[128].abs() > 0.01);
    }

    #[test]
    fn linear_phase_eq_applies_audible_band_gain() {
        let mut eq = SurgicalEq::<BANDS, TAPS, f64>::default();
        let bands = [bell_band(9.0), EqBand::default()];
        assert!(eq.configure(48_000.0, &bands));
        eq.set_phase_mode(EqPhaseMode::LinearPhase);

        let mut energy = 0.0;
        let mut count = 0;
        for sample_index in 0..48_000 {
            let input = (std::f64::consts::TAU * 1_000.0 * sample_index as f64 / 48_000.0).sin();
            let output = eq.process(input);
            assert!(output.is_finite());
            if sample_index > 4_000 {
                energy += output * output;
                count += 1;
            }
        }
        let rms = (energy / count as f64).sqrt();
        assert!(rms > 1.5, "linear-phase boost was too small: {rms}");
    }

    #[test]
    fn linear_phase_high_pass_rejects_dc() {
        let mut eq = SurgicalEq::<BANDS, TAPS, f64>::default();
        let bands = [
            EqBand {
                enabled: true,
                kind: FilterType::HighPass,
                frequency_hz: 200.0,
                q: 0.707,
                gain_db: 0.0,
            },
            EqBand::default(),
        ];
        assert!(eq.configure(48_000.0, &bands));
        eq.set_phase_mode(EqPhaseMode::LinearPhase);

        let mut output = 0.0;
        for _ in 0..48_000 {
            output = eq.process(1.0);
        }
        assert!(output.abs() < 1.0e-6, "high-pass DC residual: {output}");
    }

    #[test]
    fn linear_phase_identity_has_only_the_declared_delay() {
        let mut filter = LinearPhaseFir::<TAPS, f64>::default();
        let bands = [EqBand::default(); BANDS];
        let filters = [crate::core_dsp_modules::filters::Biquad::default(); BANDS];
        assert!(filter.configure(48_000.0, &bands, &filters));

        for sample_index in 0..TAPS + 1 {
            let output = filter.process(if sample_index == 0 { 1.0 } else { 0.0 });
            if sample_index == TAPS / 2 {
                assert!((output - 1.0).abs() < 1.0e-10);
            } else {
                assert!(output.abs() < 1.0e-10);
            }
        }
    }

    #[test]
    fn invalid_rate_and_filter_configuration_are_rejected() {
        let mut eq = SurgicalEq::<BANDS, TAPS, f64>::default();
        assert!(!eq.configure(0.0, &[bell_band(3.0), EqBand::default()]));
        assert!(!eq.configure(
            48_000.0,
            &[
                EqBand {
                    frequency_hz: 24_000.0,
                    ..bell_band(3.0)
                },
                EqBand::default()
            ]
        ));
    }
}
