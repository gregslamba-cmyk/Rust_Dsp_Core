//! # Variable-Mu Tube Compression & Time Constant Network (Vari-Mu valve comp topology / stadio valvolare a transconduttanza variabile)
//!
//! Questo modulo emula la risposta circuitale dei compressori/limiter valvolari Vari-Mu
//! (in particolare il topologia Vari-Mu valvolare per compressore stereo e mono), basati sulla modulazione
//! dinamica del punto di lavoro di tubi termoionici a cut-off remoto (come i doppi triodi 6386).
//!
//! ## Topologia Circuitale Emulata
//! - **Controllo di Guadagno Vari-Mu:** Il segnale audio rettificato genera una tensione negativa di bias DC
//!   inviata alle griglie di controllo delle valvole. All'aumentare dell'ampiezza, la transconduttanza ($g_m$)
//!   del tubo diminuisce progressivamente, realizzando una compressione fluida a rapporto continuamente variabile
//!   (da ~2:1 per lievi riduzioni fino a oltre 20:1 per picchi intensi).
//! - **Rete di Costanti Temporali (Time Constants I–VI):** Modella le 6 posizioni dell'hardware originale,
//!   comprese le posizioni composte V e VI con recupero a due stadi (rilascio rapido sui picchi brevi e
//!   rilascio prolungato fino a 25 secondi su passaggi ad alta energia continua).
//! - **Attacco Rapido:** Attacco nominale di ~200 µs (0.2 ms) per catturare i transienti più ripidi
//!   senza artefatti digitali di clipping.

use num_traits::Float;

/// Tempo di attacco nominale storico del circuito Vari-Mu (~0.2 ms / 200 microsecondi).
pub const VARI_MU_ATTACK_SECONDS: f64 = 0.0002;

/// Le sei posizioni storiche del selettore di costante temporale del Vari-Mu valvolare.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum TimeConstant {
    /// Posizione I: Rilascio veloce fisso di 0.3 s.
    #[default]
    I,
    /// Posizione II: Rilascio medio-veloce fisso di 0.8 s.
    II,
    /// Posizione III: Rilascio medio fisso di 2.0 s.
    III,
    /// Posizione IV: Rilascio lento fisso di 5.0 s.
    IV,
    /// Posizione V: Rilascio automatico a doppio stadio (2.0 s per il primo 50%, coda lenta da 10.0 s).
    V,
    /// Posizione VI: Rilascio automatico a doppio stadio dipendente dal programma (0.3 s per il primo 50%, coda lenta da 25.0 s).
    VI,
}

impl TimeConstant {
    /// Restituisce la coppia di costanti di tempo (rilascio veloce, rilascio lento opzionale) in secondi.
    fn release_seconds<T: Float>(self) -> (T, Option<T>) {
        let (first, second) = match self {
            Self::I => (0.3, None),
            Self::II => (0.8, None),
            Self::III => (2.0, None),
            Self::IV => (5.0, None),
            Self::V => (2.0, Some(10.0)),
            Self::VI => (0.3, Some(25.0)),
        };
        (
            T::from(first).unwrap_or(T::zero()),
            second.map(|seconds| T::from(seconds).unwrap_or(T::zero())),
        )
    }
}

/// Simulatore del circuito di rilevamento e controllo guadagno Vari-Mu.
///
/// Modella la pendenza progressiva di compressione e la rete temporale RC delle griglie.
#[derive(Clone, Copy, Debug)]
pub struct VariMuCell<T: Float> {
    /// Frequenza di campionamento attiva in Hz.
    sample_rate: T,
    /// Livello lineare dell'inviluppo del detector rettificato.
    envelope: T,
    /// Riduzione di guadagno istantanea corrente in dB positivi.
    gain_reduction_db: T,
    /// Riduzione in dB memorizzata all'inizio dell'evento di rilascio.
    release_start_db: T,
    /// Flag indicante se il rilascio si trova nella seconda fase lenta (per le posizioni V e VI).
    in_slow_release: bool,
    /// Posizione attiva della costante temporale (I–VI).
    time_constant: TimeConstant,
}

impl<T: Float> Default for VariMuCell<T> {
    fn default() -> Self {
        Self {
            sample_rate: T::zero(),
            envelope: T::zero(),
            gain_reduction_db: T::zero(),
            release_start_db: T::zero(),
            in_slow_release: false,
            time_constant: TimeConstant::default(),
        }
    }
}

impl<T: Float> VariMuCell<T> {
    /// Inizializza la cella Vari-Mu configurando frequenza di campionamento e costante temporale.
    ///
    /// # Parametri
    /// - `sample_rate`: Frequenza di campionamento in Hz (> 0.0).
    /// - `time_constant`: Costante temporale iniziale (I–VI).
    pub fn initialize(&mut self, sample_rate: T, time_constant: TimeConstant) -> bool {
        if !sample_rate.is_finite() || sample_rate <= T::zero() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.time_constant = time_constant;
        self.reset();
        true
    }

    /// Modifica la costante temporale attiva senza resettare bruscamente l'inviluppo audio.
    pub fn set_time_constant(&mut self, time_constant: TimeConstant) {
        if self.time_constant != time_constant {
            self.time_constant = time_constant;
            self.release_start_db = T::zero();
            self.in_slow_release = false;
        }
    }

    /// Esegue il rilevamento del segnale e calcola la riduzione di guadagno in dB istante per istante.
    ///
    /// # Parametri
    /// - `detector_level`: Ampiezza lineare assoluta del segnale nel percorso sidechain.
    /// - `threshold_db`: Soglia di inizio compressione in dBFS (tipicamente -30.0 .. 0.0 dB).
    ///
    /// # Restituisce
    /// Attenuazione istantanea in dB (valore >= 0.0).
    #[inline]
    pub fn process(&mut self, detector_level: T, threshold_db: T) -> T {
        if !self.sample_rate.is_finite() || self.sample_rate <= T::zero() {
            return T::zero();
        }

        let detector_level = if detector_level.is_finite() {
            detector_level.abs()
        } else {
            T::zero()
        };
        let threshold_db = if threshold_db.is_finite() {
            threshold_db
        } else {
            T::zero()
        };

        // Inseguimento rapido del picco di tensione rettificata sulle griglie del tubo
        if detector_level > self.envelope {
            // Calcolo del polo di attacco: alpha = exp(-1 / (fs * t_attack))
            let attack =
                (-T::one() / (self.sample_rate * T::from(VARI_MU_ATTACK_SECONDS).unwrap_or(T::one()))).exp();
            self.envelope = detector_level + attack * (self.envelope - detector_level);
        } else {
            // Rilascio dell'inviluppo primario pilotato dalla rete RC di scarica successiva
            self.envelope = detector_level;
        }

        // Conversione dell'inviluppo di tensione in dBFS (con floor di sicurezza a -240 dBFS)
        let level_db =
            T::from(20.0).unwrap_or(T::zero()) * self.envelope.max(T::from(1.0e-12).unwrap_or(T::zero())).log10();
        // Calcolo della riduzione teorica non lineare secondo la caratteristica curva Vari-Mu
        let target = vari_mu_target_reduction_db(level_db, threshold_db);

        // FASE DI ATTACCO: la tensione di bias negativo aumenta, riducendo il guadagno delle valvole
        if target >= self.gain_reduction_db {
            self.release_start_db = T::zero();
            self.in_slow_release = false;
            let attack =
                (-T::one() / (self.sample_rate * T::from(VARI_MU_ATTACK_SECONDS).unwrap_or(T::one()))).exp();
            self.gain_reduction_db = target + attack * (self.gain_reduction_db - target);

        // FASE DI RILASCIO: la carica accumulata nelle capacità di griglia decade secondo le costanti I-VI
        } else {
            if self.release_start_db <= T::zero() {
                self.release_start_db = self.gain_reduction_db;
            }
            let (fast_release, slow_release) = self.time_constant.release_seconds::<T>();

            // Per le costanti V e VI, dopo aver recuperato il primo 50% di dB si entra nella fase di coda lenta
            if !self.in_slow_release
                && slow_release.is_some()
                && self.gain_reduction_db <= self.release_start_db * T::from(0.5).unwrap_or(T::zero())
            {
                self.in_slow_release = true;
            }

            let release = if self.in_slow_release {
                slow_release.unwrap_or(fast_release)
            } else {
                fast_release
            };
            let coefficient = (-T::one() / (self.sample_rate * release)).exp();
            self.gain_reduction_db = target + coefficient * (self.gain_reduction_db - target);
        }
        self.gain_reduction_db
    }

    /// Resetta a riposo tutti i condensatori e l'inviluppo del detector Vari-Mu.
    pub fn reset(&mut self) {
        self.envelope = T::zero();
        self.gain_reduction_db = T::zero();
        self.release_start_db = T::zero();
        self.in_slow_release = false;
    }
}

/// Calcola la riduzione statica di guadagno in dB applicando la caratteristica curva Vari-Mu.
///
/// Modella il comportamento delle valvole a cut-off remoto con rapporto di compressione
/// che aumenta dolcemente da 2:1 a 20:1 all'aumentare dell'escursione sopra la soglia.
///
/// # Parametri
/// - `input_level_db`: Livello del segnale audio in dB.
/// - `threshold_db`: Livello di soglia in dB.
pub fn vari_mu_target_reduction_db<T: Float>(input_level_db: T, threshold_db: T) -> T {
    if !input_level_db.is_finite() || !threshold_db.is_finite() {
        return T::zero();
    }

    let over_threshold = input_level_db - threshold_db;
    let knee = T::from(2.0).unwrap_or(T::one());
    if over_threshold <= -knee || !over_threshold.is_finite() {
        return T::zero();
    }

    // Soft-knee: parabolica raccordo su [-knee, +knee]: reduction = (over + knee)^2 / (4 * knee)
    if over_threshold < T::zero() {
        let o = over_threshold + knee;
        let reduction = o * o / (T::from(4.0).unwrap_or(T::one()) * knee);
        if reduction.is_finite() {
            return reduction;
        }
        return T::zero();
    }

    let over = over_threshold;
    // Rapporto di compressione non lineare asintotico: ratio = 2.0 + 18.0 * (1 - exp(-over / 18.0))
    // Produce un ratio che varia gradualmente da 2:1 (piccole escursioni) a 20:1 (forti picchi)
    let ratio = T::from(2.0).unwrap_or(T::one())
        + T::from(18.0).unwrap_or(T::one()) * (T::one() - (-over / T::from(18.0).unwrap_or(T::one())).exp());
    // Attenuazione: delta_dB = over * (1 - 1 / ratio)
    let reduction = over * (T::one() - T::one() / ratio);

    if reduction.is_finite() { reduction } else { T::zero() }
}

#[cfg(test)]
mod tests {
    use super::{TimeConstant, VARI_MU_ATTACK_SECONDS, VariMuCell, vari_mu_target_reduction_db};

    #[test]
    fn historical_vari_mu_attack_nominal_is_represented() {
        assert_eq!(VARI_MU_ATTACK_SECONDS, 0.0002);
    }

    #[test]
    fn ratio_increases_with_gain_reduction() {
        let low = vari_mu_target_reduction_db(-17.0_f64, -18.0);
        let high = vari_mu_target_reduction_db(6.0_f64, -18.0);

        assert!(low > 0.0);
        assert!(high > low * 10.0);
        assert_eq!(vari_mu_target_reduction_db(-25.0_f64, -18.0), 0.0);
    }

    #[test]
    fn six_time_constants_have_distinct_release_behaviors() {
        let mut results = Vec::new();
        for time_constant in [
            TimeConstant::I,
            TimeConstant::II,
            TimeConstant::III,
            TimeConstant::IV,
            TimeConstant::V,
            TimeConstant::VI,
        ] {
            let mut cell = VariMuCell::<f32>::default();
            assert!(cell.initialize(48_000.0, time_constant));
            for _ in 0..48_000 {
                cell.process(1.0, -18.0);
            }
            for _ in 0..144_000 {
                cell.process(0.0, -18.0);
            }
            results.push(cell.process(0.0, -18.0));
        }

        assert!(results[0] < results[1]);
        assert!(results[1] < results[2]);
        assert!(results[2] < results[3]);
        assert!(results[4] > results[2]);
        assert!(results[5] > results[0]);
    }

    #[test]
    fn sample_rate_is_required_and_invalid_samples_are_sanitized() {
        let mut cell = VariMuCell::<f64>::default();
        assert!(!cell.initialize(0.0, TimeConstant::I));
        assert!(cell.initialize(48_000.0, TimeConstant::I));
        assert_eq!(cell.process(f64::NAN, f64::NAN), 0.0);
    }
}
