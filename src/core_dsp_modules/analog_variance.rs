//! # Analog Variance & Component Tolerances
//!
//! Questo modulo modella le micro-tolleranze fisiche dei componenti elettronici analogici
//! (resistenze all'1-5%, condensatori al 5-10%, potenziometri a doppio cursore disallineati)
//! e il crosstalk inter-canale tipico dei circuiti stampati (PCB) e dei cablaggi discreti.
//!
//! Ogni istanza plugin calcola uno stato di varianza deterministico a partire da un `instance_id`
//! o seed casuale, garantendo differenze microscopiche e realistiche tra i canali Left e Right
//! (guadagno, costanti di tempo di attacco/rilascio, frequenze di taglio) scalabili o
//! disattivabili tramite il parametro globale `analog_drift`.

/// Varianza analogica statica per canale e fattore di crosstalk inter-canale.
///
/// Modella la discrepanza fisica tra i canali Left e Right dovuta alle tolleranze reali dei componenti,
/// evitando la sterilità del digitale perfettamente simmetrico senza introdurre instabilità dinamica.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalogVariance {
    /// Moltiplicatore lineare di guadagno per il canale sinistro (es. 0.98 - 1.02 per ±2% di tolleranza).
    pub left_gain: f64,
    /// Moltiplicatore lineare di guadagno per il canale destro (es. 0.98 - 1.02 per ±2% di tolleranza).
    pub right_gain: f64,
    /// Fattore di scala per i tempi di attacco/rilascio o frequenze del canale sinistro.
    pub left_time_scale: f64,
    /// Fattore di scala per i tempi di attacco/rilascio o frequenze del canale destro.
    pub right_time_scale: f64,
    /// Coefficiente lineare di accoppiamento/crosstalk tra i canali Left e Right (tipicamente < -60 dB, 0.001).
    pub crosstalk: f64,
}

impl AnalogVariance {
    /// Genera una configurazione deterministica di varianza a partire da un seme a 64 bit.
    ///
    /// # Parametri
    /// - `seed`: Identificativo univoco dell'istanza hardware/plugin per generare valori ripetibili.
    /// - `tolerance`: Tolleranza nominale dei componenti (es. 0.02 per ±2%), limitata per sicurezza a max 5%.
    /// - `crosstalk_max`: Livello massimo di perdita inter-canale (lineare, max 0.05).
    /// - `analog_drift`: Controllo utente [0.0, 1.0] per disattivare (0.0) o scalare l'effetto analogico.
    ///
    /// # Vincoli Real-Time
    /// Lock-free, wait-free, zero allocazioni heap; eseguibile sia in `initialize()` che durante il reset dei parametri.
    pub fn from_seed(seed: u64, tolerance: f64, crosstalk_max: f64, analog_drift: f64) -> Self {
        // Sanitizzazione parametro drift utente [0.0, 1.0]
        let drift = if analog_drift.is_finite() {
            analog_drift.clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Se il drift analogico è azzerato dall'utente, restituisce simmetria digitale assoluta
        if drift <= 0.0 {
            return Self {
                left_gain: 1.0,
                right_gain: 1.0,
                left_time_scale: 1.0,
                right_time_scale: 1.0,
                crosstalk: 0.0,
            };
        }

        // Scala la tolleranza reale dei componenti in base al controllo drift utente (max 5%)
        let tolerance = if tolerance.is_finite() {
            tolerance.abs().min(0.05) * drift
        } else {
            0.02 * drift
        };

        // Accoppiamento parassita capacitivo/induttivo scalato per il drift analogico
        let crosstalk_max = if crosstalk_max.is_finite() {
            crosstalk_max.abs().min(0.05) * drift
        } else {
            0.0
        };

        // Generatore pseudocasuale deterministico Xorshift64 a zero allocazioni
        let mut rng = seed.max(1);
        let mut factor = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Mappatura lineare nell'intervallo simmetrico [-1.0, +1.0]
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        Self {
            // Deviazione percentuale asimmetrica per modellare la tolleranza dei resistori L/R
            left_gain: 1.0 + factor() * tolerance,
            right_gain: 1.0 + factor() * tolerance,
            // Variazione delle costanti di tempo RC per disallineamento naturale tra i canali
            left_time_scale: 1.0 + factor() * tolerance,
            right_time_scale: 1.0 + factor() * tolerance,
            // Crosstalk residuo L->R e R->L per accoppiamento di massa e vicinanza piste PCB
            crosstalk: factor() * crosstalk_max,
        }
    }
}

/// Block-rate thermal drift for slow analog component variation.
///
/// Models the slow thermal drift of analog components (bias points, time constants)
/// using two asynchronous oscillators per channel with fixed periods (~17.3 s and ~23.7 s).
/// Updated once per audio block (not per sample) for CPU efficiency.
/// Modulates ±0.2% bias DC and time constants. Deterministic from instance_id.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThermalDrift {
    phase1_l: f64,
    phase1_r: f64,
    phase2_l: f64,
    phase2_r: f64,
    inc1_per_sample: f64,
    inc2_per_sample: f64,
}

impl ThermalDrift {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure with sample rate. Must be called before use.
    pub fn configure(&mut self, sample_rate: f64) {
        let sr = sample_rate.max(1.0);
        let period1 = 17.3;
        let period2 = 23.7;
        self.inc1_per_sample = std::f64::consts::TAU / (period1 * sr);
        self.inc2_per_sample = std::f64::consts::TAU / (period2 * sr);
    }

    pub fn reset(&mut self) {
        self.phase1_l = 0.0;
        self.phase1_r = 0.0;
        self.phase2_l = 0.0;
        self.phase2_r = 0.0;
    }

    /// Update thermal drift for one audio block.
    ///
    /// Returns (bias_l, bias_r, time_scale_l, time_scale_r) where:
    /// - bias: ±0.002 (±0.2%) DC offset modulation
    /// - time_scale: ±0.002 (±0.2%) time constant modulation
    /// `block_size` is the number of samples in the current block.
    /// `analog_drift` is the user control [0.0, 1.0] (0 = off).
    /// `instance_id` is used to deterministically offset phases per plugin instance.
    #[inline]
    pub fn update_block(
        &mut self,
        block_size: usize,
        analog_drift: f64,
        instance_id: u64,
    ) -> (f64, f64, f64, f64) {
        if !analog_drift.is_finite() || analog_drift <= 0.0 || block_size == 0 {
            return (0.0, 0.0, 1.0, 1.0);
        }

        let drift = analog_drift.clamp(0.0, 1.0);

        // Deterministic phase offsets from instance_id
        let mut rng = instance_id.max(1);
        let mut next_u64 = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let phase_offset1_l = (next_u64() as f64 / u64::MAX as f64) * std::f64::consts::TAU;
        let phase_offset1_r = (next_u64() as f64 / u64::MAX as f64) * std::f64::consts::TAU;
        let phase_offset2_l = (next_u64() as f64 / u64::MAX as f64) * std::f64::consts::TAU;
        let phase_offset2_r = (next_u64() as f64 / u64::MAX as f64) * std::f64::consts::TAU;

        // Advance phases
        let inc1 = self.inc1_per_sample * block_size as f64;
        let inc2 = self.inc2_per_sample * block_size as f64;

        self.phase1_l = (self.phase1_l + inc1) % std::f64::consts::TAU;
        self.phase1_r = (self.phase1_r + inc1) % std::f64::consts::TAU;
        self.phase2_l = (self.phase2_l + inc2) % std::f64::consts::TAU;
        self.phase2_r = (self.phase2_r + inc2) % std::f64::consts::TAU;

        // Combine two oscillators for more organic variation
        let osc1_l = (self.phase1_l + phase_offset1_l).sin();
        let osc1_r = (self.phase1_r + phase_offset1_r).sin();
        let osc2_l = (self.phase2_l + phase_offset2_l).sin();
        let osc2_r = (self.phase2_r + phase_offset2_r).sin();

        // Weighted sum, scaled by drift and max ±0.2%
        let bias_scale = 0.002 * drift;
        let time_scale_mod = 0.002 * drift;

        let bias_l = (osc1_l * 0.6 + osc2_l * 0.4) * bias_scale;
        let bias_r = (osc1_r * 0.6 + osc2_r * 0.4) * bias_scale;
        let time_scale_l = 1.0 + (osc1_l * 0.4 + osc2_l * 0.6) * time_scale_mod;
        let time_scale_r = 1.0 + (osc1_r * 0.4 + osc2_r * 0.6) * time_scale_mod;

        (bias_l, bias_r, time_scale_l, time_scale_r)
    }
}

#[cfg(test)]
mod tests {
    use super::{AnalogVariance, ThermalDrift};

    #[test]
    fn seed_produces_repeatable_instance_variation() {
        assert_eq!(
            AnalogVariance::from_seed(123, 0.03, 0.01, 1.0),
            AnalogVariance::from_seed(123, 0.03, 0.01, 1.0)
        );
    }

    #[test]
    fn zero_drift_disables_variance() {
        let v = AnalogVariance::from_seed(123, 0.03, 0.01, 0.0);
        assert_eq!(v.left_gain, 1.0);
        assert_eq!(v.right_gain, 1.0);
        assert_eq!(v.crosstalk, 0.0);
    }

    #[test]
    fn thermal_drift_zero_when_disabled() {
        let mut td = ThermalDrift::new();
        td.configure(48000.0);
        let (bias_l, bias_r, ts_l, ts_r) = td.update_block(64, 0.0, 12345);
        assert_eq!(bias_l, 0.0);
        assert_eq!(bias_r, 0.0);
        assert_eq!(ts_l, 1.0);
        assert_eq!(ts_r, 1.0);
    }

    #[test]
    fn thermal_drift_produces_small_variation() {
        let mut td = ThermalDrift::new();
        td.configure(48000.0);
        let (bias_l, bias_r, ts_l, ts_r) = td.update_block(512, 1.0, 42);
        assert!(bias_l.abs() <= 0.0021);
        assert!(bias_r.abs() <= 0.0021);
        assert!((ts_l - 1.0).abs() <= 0.0021);
        assert!((ts_r - 1.0).abs() <= 0.0021);
    }

    #[test]
    fn thermal_drift_deterministic() {
        let mut td1 = ThermalDrift::new();
        td1.configure(48000.0);
        let mut td2 = ThermalDrift::new();
        td2.configure(48000.0);
        for _ in 0..10 {
            let b1 = td1.update_block(64, 1.0, 999);
            let b2 = td2.update_block(64, 1.0, 999);
            assert_eq!(b1, b2);
        }
    }
}
