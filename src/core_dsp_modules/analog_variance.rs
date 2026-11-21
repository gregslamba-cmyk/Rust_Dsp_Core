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

#[cfg(test)]
mod tests {
    use super::AnalogVariance;

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
}
