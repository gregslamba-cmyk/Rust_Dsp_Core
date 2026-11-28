//! # Electro-Optical Attenuator Modeling (Greg-A2A / Greg-A3A)
//!
//! Questo modulo emula la risposta dinamica non lineare delle celle elettro-ottiche
//! storiche (Greg-A2A / Greg-A3A a elettroluminescenza e fotoresistori al solfuro di cadmio CdS),
//! utilizzate nel livellatore valvolare `greg-a2a` e nel modello a stato solido `greg-a3a`.
//!
//! ## Topologia Circuitale e Comportamento Fisico
//! - **Pannello Elettroluminescente:** Converte la tensione audio amplificata in flusso luminoso.
//! - **Fotoresistore CdS con Memoria di Programma:** La resistenza diminuisce all'aumentare della luce,
//!   attenuando passivamente il segnale audio nel circuito di partizione. La persistenza chimica
//!   del drogaggio crea una memoria d'inviluppo organica ("program dependency").
//! - **Rilascio a Doppio Stadio (Two-Stage Release):** Il primo 50% della riduzione di guadagno
//!   decade rapidamente (~60 ms per Greg-A2A, ~25 ms per Greg-A3A), mentre la restante parte rilascia
//!   molto lentamente (da 0.5 s fino a 5 secondi), prevenendo l'effetto "pumping" tipico dei VCA convenzionali.
//! - **Modalità Compress vs Limit:** Modula la pendenza di compressione da un soft-knee musicale (~3:1)
//!   a una limitazione più decisa (~8:1 - 10:1).

use num_traits::Float;

/// Tempo di attacco nominale per cella ottica valvolare Greg-A2A (~10 ms per raggiungere il 63% dell'attenuazione).
pub const OPTO_A2A_ATTACK_SECONDS: f64 = 0.010;
/// Tempo di emivita del rilascio veloce per Greg-A2A (~60 ms per il primo 50% di recupero).
pub const OPTO_A2A_FAST_RELEASE_SECONDS: f64 = 0.060;
/// Rilascio lento minimo per Greg-A2A in assenza di accumulo storico di programma (~0.5 s).
pub const OPTO_A2A_SLOW_RELEASE_MIN_SECONDS: f64 = 0.5;
/// Rilascio lento massimo per Greg-A2A dopo prolungata e intensa compressione (~5.0 s).
pub const OPTO_A2A_SLOW_RELEASE_MAX_SECONDS: f64 = 5.0;

/// Tempo di attacco nominale per cella ottica a stato solido Greg-A3A (~3.5 ms, transitorio più scattante).
pub const OPTO_A3A_ATTACK_SECONDS: f64 = 0.0035;
/// Tempo di emivita del rilascio veloce per Greg-A3A (~25 ms).
pub const OPTO_A3A_FAST_RELEASE_SECONDS: f64 = 0.025;
/// Rilascio lento minimo per Greg-A3A (~0.12 s).
pub const OPTO_A3A_SLOW_RELEASE_MIN_SECONDS: f64 = 0.12;
/// Rilascio lento massimo per Greg-A3A (~1.5 s).
pub const OPTO_A3A_SLOW_RELEASE_MAX_SECONDS: f64 = 1.5;

/// Modalità operativa della cella opto (Compress vs Limit).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptoMode {
    /// Modalità compressione morbida (rapporto nominale progressivo ~3:1).
    #[default]
    Compress,
    /// Modalità limitazione di picco (rapporto nominale più rigido ~8:1 - 10:1).
    Limit,
}

/// Simulatore dinamico della cella fotoresistiva elettro-ottica Greg-A2A.
///
/// Modella memoria fotochimica di programma, attacco non lineare e recupero a due stadi.
#[derive(Clone, Copy, Debug)]
pub struct OptoCell<T: Float> {
    /// Frequenza di campionamento attiva in Hz.
    sample_rate: T,
    /// Riduzione di guadagno istantanea corrente espressa in dB positivi (es. 6.0 = -6 dB di attenuazione).
    reduction_db: T,
    /// Accumulo energetico storico della cella (memoria fotoresistiva [0.0, 1.0]).
    program_memory: T,
    /// Flag attivo durante la prima fase veloce del rilascio (primo 50% di recupero dB).
    fast_release: bool,
    /// Valore di riduzione in dB registrato al momento dell'inizio della fase di rilascio.
    release_start_db: T,
    /// Costante di tempo corrente per la seconda fase lenta di rilascio (in secondi).
    slow_release_seconds: T,
}

impl<T: Float> Default for OptoCell<T> {
    fn default() -> Self {
        Self {
            sample_rate: T::zero(),
            reduction_db: T::zero(),
            program_memory: T::zero(),
            fast_release: false,
            release_start_db: T::zero(),
            slow_release_seconds: T::one(),
        }
    }
}

impl<T: Float> OptoCell<T> {
    /// Helper generico per flush denormali senza usare .unwrap() nel percorso audio.
    #[inline]
    fn flush_denormal(x: T) -> T {
        let cutoff = T::from(1.0e-30).unwrap_or_else(T::zero);
        if !x.is_finite() || x.abs() < cutoff {
            T::zero()
        } else {
            x
        }
    }
    /// Inizializza la cella ottica configurando la frequenza di campionamento ed eseguendo il reset dello stato.
    ///
    /// # Parametri
    /// - `sample_rate`: Frequenza di campionamento in Hz (> 0.0).
    pub fn initialize(&mut self, sample_rate: T) -> bool {
        if !sample_rate.is_finite() || sample_rate <= T::zero() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.reset();
        true
    }

    /// Processa la riduzione di guadagno istantanea per profilo valvolare classico (Greg-A2A).
    ///
    /// # Parametri
    /// - `target_reduction_db`: Attenuazione target richiesta dal detector in dB.
    ///
    /// # Restituisce
    /// Valore effettivo smussato della riduzione in dB da applicare al segnale.
    #[inline]
    pub fn process(&mut self, target_reduction_db: T) -> T {
        self.process_with_character(target_reduction_db, false)
    }

    /// Processa la riduzione di guadagno istantanea per profilo a stato solido più reattivo (Greg-A3A).
    ///
    /// # Parametri
    /// - `target_reduction_db`: Attenuazione target richiesta dal detector in dB.
    #[inline]
    pub fn process_solid_state(&mut self, target_reduction_db: T) -> T {
        self.process_with_character(target_reduction_db, true)
    }

    /// Motore interno unificato di integrazione differenziale della cella ottica.
    #[inline]
    fn process_with_character(&mut self, target_reduction_db: T, solid_state: bool) -> T {
        if !self.sample_rate.is_finite() || self.sample_rate <= T::zero() {
            return T::zero();
        }

        // Morsetto di sicurezza sulla riduzione target per prevenire overflow o valori negativi
        let target = if target_reduction_db.is_finite() {
            target_reduction_db
                .max(T::zero())
                .min(T::from(60.0).unwrap_or(T::zero()))
        } else {
            T::zero()
        };

        // Inseguimento della memoria di programma della cella fotoresistiva CdS:
        // L'esposizione prolungata a forte riduzione (fino a 24 dB) aumenta la memoria lenta
        let activity_target = (target / T::from(24.0).unwrap()).min(T::one());
        let memory_seconds = if solid_state {
            T::from(0.5).unwrap()
        } else {
            T::from(2.0).unwrap()
        };
        let memory_coefficient = Self::coefficient(memory_seconds, self.sample_rate);
        self.program_memory =
            activity_target + memory_coefficient * (self.program_memory - activity_target);

        // FASE DI ATTACCO: la luce sul fotoresistore aumenta, riducendo rapidamente la resistenza
        if target > self.reduction_db {
            // Un nuovo attacco azzera il latch di rilascio: il release_start_db verra'
            // catturato come reduction_db reale nel momento in cui inizia il rilascio.
            self.fast_release = false;
            self.release_start_db = T::zero();

            // L'attacco ottico diventa leggermente più veloce su transienti più forti (caratteristica Greg-A2A)
            let attack_base = if solid_state {
                OPTO_A3A_ATTACK_SECONDS
            } else {
                OPTO_A2A_ATTACK_SECONDS
            };
            let attack_sensitivity = if solid_state { 0.15 } else { 0.25 };
            let attack_seconds = T::from(attack_base).unwrap()
                / (T::one()
                    + T::from(attack_sensitivity).unwrap()
                        * (target / T::from(20.0).unwrap()).min(T::one()));
            let coefficient = Self::coefficient(attack_seconds, self.sample_rate);
            // Integrazione esponenziale dell'attacco
            self.reduction_db = Self::flush_denormal(
                target + coefficient * (self.reduction_db - target),
            );

        // FASE DI RILASCIO: il pannello si spegne, il drogaggio CdS dissipa lentamente i portatori di carica
        } else if target < self.reduction_db {
            // Inizializzazione della fase di rilascio: cattura la riduzione REALE al momento
            // dell'ingresso in release (non il target, che potrebbe essere molto piu' alto).
            if !self.fast_release && self.release_start_db <= T::zero() {
                self.release_start_db = self.reduction_db;
                self.fast_release = true;
                let (minimum, range) = if solid_state {
                    (
                        OPTO_A3A_SLOW_RELEASE_MIN_SECONDS,
                        OPTO_A3A_SLOW_RELEASE_MAX_SECONDS - OPTO_A3A_SLOW_RELEASE_MIN_SECONDS,
                    )
                } else {
                    (
                        OPTO_A2A_SLOW_RELEASE_MIN_SECONDS,
                        OPTO_A2A_SLOW_RELEASE_MAX_SECONDS - OPTO_A2A_SLOW_RELEASE_MIN_SECONDS,
                    )
                };
                // Calcolo della costante di rilascio lento in funzione della memoria fotoresistiva accumulata
                self.slow_release_seconds =
                    T::from(minimum).unwrap() + T::from(range).unwrap() * self.program_memory;
            }

            // Transizione dal rilascio veloce al rilascio lento una volta recuperato il primo 50% di dB
            if self.fast_release
                && self.reduction_db <= self.release_start_db * T::from(0.5).unwrap()
            {
                self.fast_release = false;
                let (minimum, range) = if solid_state {
                    (
                        OPTO_A3A_SLOW_RELEASE_MIN_SECONDS,
                        OPTO_A3A_SLOW_RELEASE_MAX_SECONDS - OPTO_A3A_SLOW_RELEASE_MIN_SECONDS,
                    )
                } else {
                    (
                        OPTO_A2A_SLOW_RELEASE_MIN_SECONDS,
                        OPTO_A2A_SLOW_RELEASE_MAX_SECONDS - OPTO_A2A_SLOW_RELEASE_MIN_SECONDS,
                    )
                };
                self.slow_release_seconds =
                    T::from(minimum).unwrap() + T::from(range).unwrap() * self.program_memory;
            }

            // Selezione della costante temporale: emivita esponenziale per la fase veloce (tau = t_half / ln(2))
            let release_seconds = if self.fast_release {
                let half_life = if solid_state {
                    OPTO_A3A_FAST_RELEASE_SECONDS
                } else {
                    OPTO_A2A_FAST_RELEASE_SECONDS
                };
                T::from(half_life).unwrap_or_else(T::zero) / T::from(std::f64::consts::LN_2).unwrap_or_else(T::zero)
            } else {
                self.slow_release_seconds
            };
            let coefficient = Self::coefficient(release_seconds, self.sample_rate);
            // Integrazione esponenziale del rilascio (con flush denormal)
            self.reduction_db = Self::flush_denormal(
                target + coefficient * (self.reduction_db - target),
            );
        }

        self.reduction_db
    }

    /// Ripristina a riposo tutti gli stati fisici di memoria e riduzione della cella ottica.
    pub fn reset(&mut self) {
        self.reduction_db = T::zero();
        self.program_memory = T::zero();
        self.fast_release = false;
        self.release_start_db = T::zero();
        self.slow_release_seconds = T::one();
    }

    /// Calcola il coefficiente di polo IIR esponenziale a partire da tempo in secondi e sample rate:
    /// alpha = exp(-1 / (time * sample_rate))
    #[inline]
    fn coefficient(time_seconds: T, sample_rate: T) -> T {
        (-T::one() / (time_seconds * sample_rate)).exp()
    }
}

/// Calcola l'attenuazione ideale statica target in dB per il circuito Greg-A2A.
///
/// # Parametri
/// - `input_level_db`: Livello RMS o di picco del segnale in ingresso in dBFS.
/// - `peak_reduction_db`: Controllo potenziometro di compressione "Peak Reduction" [0.0, 100.0 dB].
/// - `mode`: Modalità Compress o Limit.
pub fn target_reduction_db<T: Float>(input_level_db: T, peak_reduction_db: T, mode: OptoMode) -> T {
    target_reduction_db_for_character(input_level_db, peak_reduction_db, mode, false)
}

/// Calcola l'attenuazione ideale statica target in dB per il circuito Greg-A3A a stato solido.
pub fn target_solid_state_reduction_db<T: Float>(
    input_level_db: T,
    peak_reduction_db: T,
    mode: OptoMode,
) -> T {
    target_reduction_db_for_character(input_level_db, peak_reduction_db, mode, true)
}

/// Calcolo unificato della curva di compressione non lineare con ginocchio morbido (soft-knee parabolico).
fn target_reduction_db_for_character<T: Float>(
    input_level_db: T,
    peak_reduction_db: T,
    mode: OptoMode,
    solid_state: bool,
) -> T {
    if !input_level_db.is_finite() || !peak_reduction_db.is_finite() {
        return T::zero();
    }

    // La soglia si abbassa all'aumentare del controllo Peak Reduction
    let threshold_db = T::from(-6.0).unwrap() - peak_reduction_db.max(T::zero());
    let over_threshold = input_level_db - threshold_db;
    let half_knee = T::one(); // Zona di transizione morbida di 2 dB (±1 dB attorno alla soglia)

    // Al di sotto del ginocchio inferiore non viene applicata alcuna compressione
    if over_threshold <= -half_knee {
        return T::zero();
    }

    // Rapporto di compressione dinamico progressivo: aumenta dolcemente man mano che il segnale supera la soglia
    let positive_over = over_threshold.max(T::zero());
    let ratio_at_high_level = T::one() - (-positive_over / T::from(12.0).unwrap()).exp();
    let (base_ratio, ratio_range) = match (solid_state, mode) {
        (false, OptoMode::Compress) => (T::from(2.5).unwrap(), T::from(2.5).unwrap()),
        (false, OptoMode::Limit) => (T::from(5.0).unwrap(), T::from(5.0).unwrap()),
        (true, OptoMode::Compress) => (T::from(3.5).unwrap(), T::from(2.5).unwrap()),
        (true, OptoMode::Limit) => (T::from(6.0).unwrap(), T::from(4.0).unwrap()),
    };
    let ratio = base_ratio + ratio_range * ratio_at_high_level;
    // Pendenza di attenuazione: slope = 1 - 1 / ratio
    let slope = T::one() - T::one() / ratio;

    // Interpolazione quadratica dolce all'interno del ginocchio (evita discontinuità derivata prima)
    if over_threshold < half_knee {
        slope * (over_threshold + half_knee) * (over_threshold + half_knee) / T::from(4.0).unwrap()
    } else {
        // Al di sopra del ginocchio: pendenza asintotica standard
        slope * over_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OPTO_A2A_ATTACK_SECONDS, OPTO_A2A_FAST_RELEASE_SECONDS, OPTO_A2A_SLOW_RELEASE_MAX_SECONDS,
        OPTO_A2A_SLOW_RELEASE_MIN_SECONDS, OPTO_A3A_ATTACK_SECONDS, OPTO_A3A_FAST_RELEASE_SECONDS,
        OPTO_A3A_SLOW_RELEASE_MAX_SECONDS, OPTO_A3A_SLOW_RELEASE_MIN_SECONDS, OptoCell, OptoMode,
        target_reduction_db, target_solid_state_reduction_db,
    };

    #[test]
    fn historical_opto_nominal_ranges_are_represented() {
        assert_eq!(OPTO_A2A_ATTACK_SECONDS, 0.010);
        assert_eq!(OPTO_A2A_FAST_RELEASE_SECONDS, 0.060);
        assert_eq!(OPTO_A2A_SLOW_RELEASE_MIN_SECONDS, 0.5);
        assert_eq!(OPTO_A2A_SLOW_RELEASE_MAX_SECONDS, 5.0);
        assert_eq!(OPTO_A3A_ATTACK_SECONDS, 0.0035);
        assert_eq!(OPTO_A3A_FAST_RELEASE_SECONDS, 0.025);
        assert_eq!(OPTO_A3A_SLOW_RELEASE_MIN_SECONDS, 0.12);
        assert_eq!(OPTO_A3A_SLOW_RELEASE_MAX_SECONDS, 1.5);
    }

    #[test]
    fn sample_rate_is_required_and_shapes_attack_in_seconds() {
        let mut cell = OptoCell::<f32>::default();
        assert!(!cell.initialize(0.0));
        assert!(cell.initialize(48_000.0));

        let mut at_96k = OptoCell::<f32>::default();
        assert!(at_96k.initialize(96_000.0));
        for _ in 0..479 {
            cell.process(12.0);
        }
        for _ in 0..959 {
            at_96k.process(12.0);
        }
        assert!((cell.process(12.0) - at_96k.process(12.0)).abs() < 0.001);
    }

    #[test]
    fn program_dependent_memory_extends_the_slow_release() {
        let mut short_program = OptoCell::<f64>::default();
        let mut long_program = OptoCell::<f64>::default();
        assert!(short_program.initialize(48_000.0));
        assert!(long_program.initialize(48_000.0));

        for _ in 0..480 {
            short_program.process(12.0);
        }
        for _ in 0..48_000 {
            long_program.process(12.0);
        }
        for _ in 0..48_000 {
            short_program.process(0.0);
            long_program.process(0.0);
        }

        assert!(long_program.process(0.0) > short_program.process(0.0));
    }

    #[test]
    fn release_has_a_fast_first_stage_then_a_slow_tail() {
        let mut cell = OptoCell::<f64>::default();
        assert!(cell.initialize(48_000.0));
        for _ in 0..48_000 {
            cell.process(12.0);
        }
        let peak = cell.process(12.0);

        for _ in 0..2_880 {
            cell.process(0.0);
        }
        let after_fast_stage = cell.process(0.0);
        assert!(after_fast_stage > peak * 0.45 && after_fast_stage < peak * 0.55);

        for _ in 0..48_000 {
            cell.process(0.0);
        }
        let after_slow_tail = cell.process(0.0);
        assert!(after_slow_tail > 0.0 && after_slow_tail < after_fast_stage);
    }

    #[test]
    fn peak_reduction_and_limit_mode_increase_attenuation() {
        let low = target_reduction_db(-8.0_f64, 0.0, OptoMode::Compress);
        let compressed = target_reduction_db(-6.0_f64, 18.0, OptoMode::Compress);
        let limited = target_reduction_db(-6.0_f64, 18.0, OptoMode::Limit);

        assert_eq!(low, 0.0);
        assert!(compressed > 0.0);
        assert!(limited > compressed);
    }

    #[test]
    fn solid_state_variant_tracks_faster_than_tube_variant() {
        let mut tube = OptoCell::<f32>::default();
        let mut solid_state = OptoCell::<f32>::default();
        assert!(tube.initialize(48_000.0));
        assert!(solid_state.initialize(48_000.0));

        for _ in 0..240 {
            tube.process(12.0);
            solid_state.process_solid_state(12.0);
        }
        assert!(solid_state.process_solid_state(12.0) > tube.process(12.0));

        let tube_target = target_reduction_db(-8.0_f32, 18.0, OptoMode::Compress);
        let solid_target = target_solid_state_reduction_db(-8.0_f32, 18.0, OptoMode::Compress);
        assert!(solid_target > tube_target);
    }
}
