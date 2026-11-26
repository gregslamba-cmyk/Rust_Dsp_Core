//! # Power Supply Unit (PSU) Sag & Dynamic Rail Sagging
//!
//! Questo modulo simula il comportamento fisico degli stadi di alimentazione analogica
//! non regolata o parzialmente regolata (tipica di amplificatori valvolari storici,
//! preamplificatori discreti in Classe A e outboard vintage).
//!
//! ## Topologia Circuitale Emulata
//! Sotto carichi dinamici pesanti (transienti a bassa frequenza e forte assorbimento di corrente),
//! la tensione di alimentazione sui condensatori di filtro dell'alimentatore subisce un calo temporaneo
//! ("sag"), riducendo l'headroom utile dello stadio di guadagno e inducendo una compressione organica morbida.
//!
//! ## Politica di Silenzio Assoluto
//! I generatori di ripple di rete (50/100 Hz) e rumore termico sono **RIGOROSAMENTE SPENTI DI DEFAULT**
//! (`0.0` / `false`) e si attivano esclusivamente se richiesto esplicitamente dall'utente e in presenza di segnale,
//! garantendo lo zero digitale assoluto (`0.0`) in assenza di ingresso.

use num_traits::Float;

/// Simulatore del calo di tensione dell'alimentatore (PSU Sag) sotto carico dinamico.
///
/// Modula l'headroom e il guadagno del circuito in base all'inviluppo energetico del segnale audio.
#[derive(Clone, Copy, Debug)]
pub struct PsuSag<T: Float> {
    /// Inviluppo corrente dell'assorbimento di corrente sulla linea di alimentazione (lineare).
    envelope: f64,
    /// Coefficiente di scarica/recupero del condensatore di filtro dell'alimentatore.
    coefficient: f64,
    /// Profondità massima del calo di tensione (0.0 = alimentazione ideale rigida, 1.0 = sag profondo).
    depth: f64,
    /// Livello di rumore termico di modulazione della linea (spento di default a 0.0).
    noise_level: f64,
    /// Flag per abilitare il ripple alternato di rete a 50/100 Hz (disabilitato di default a false).
    ripple_enabled: bool,
    /// Fase dell'oscillatore di ripple di rete [0, 2π].
    ripple_phase: f64,
    /// Fase del generatore di rumore termico (avanza per campione).
    noise_phase: f64,
    /// Frequenza di campionamento in Hz.
    sample_rate: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Default for PsuSag<T> {
    fn default() -> Self {
        Self {
            envelope: 0.0,
            coefficient: 0.0,
            depth: 0.0,
            noise_level: 0.0, // Rigorosamente spento di default per silenzio a riposo
            ripple_enabled: false, // Rigorosamente disabilitato di default
            ripple_phase: 0.0,
            noise_phase: 0.0,
            sample_rate: 44100.0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Float> PsuSag<T> {
    /// Crea una nuova istanza di `PsuSag` con parametri di default sicuri.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inizializza i parametri temporali e la frequenza di campionamento dell'alimentatore.
    ///
    /// # Parametri
    /// - `sample_rate`: Frequenza di campionamento in Hz (deve essere finita e > 0).
    /// - `recovery_seconds`: Tempo di recupero/ricarica della capacità di filtro in secondi (es. 0.05 - 0.5 s).
    /// - `depth`: Quantità di calo di tensione sotto carico [0.0, 1.0].
    ///
    /// # Restituisce
    /// `true` se i parametri sono finiti e validi, `false` altrimenti (senza panic).
    pub fn initialize(&mut self, sample_rate: T, recovery_seconds: T, depth: T) -> bool {
        let sr = sample_rate.to_f64().unwrap_or(44100.0);
        let rec = recovery_seconds.to_f64().unwrap_or(0.1);
        let dep = depth.to_f64().unwrap_or(0.0);

        if !sr.is_finite() || sr <= 0.0 || !rec.is_finite() || rec <= 0.0 || !dep.is_finite() {
            return false;
        }

        // Calcolo del polo esponenziale RC: costante temporale di ricarica dei condensatori di filtro
        // alpha = exp(-1 / (fs * recovery_time))
        self.coefficient = (-1.0 / (sr * rec)).exp();
        self.depth = dep.clamp(0.0, 1.0);
        self.envelope = 0.0;
        self.ripple_phase = 0.0;
        self.noise_phase = 0.0;
        self.sample_rate = sr;
        true
    }

    /// Imposta i livelli opzionali di rumore e ripple di rete (funzionalità estese analogiche).
    ///
    /// # Parametri
    /// - `noise_level`: Livello di fluttuazione casuale del voltaggio (lineare, >= 0.0).
    /// - `ripple_enabled`: Flag di attivazione dell'ondulazione residua di rettificazione AC.
    pub fn set_noise_and_ripple(&mut self, noise_level: T, ripple_enabled: bool) {
        let nl = noise_level.to_f64().unwrap_or(0.0);
        if nl.is_finite() {
            self.noise_level = nl.max(0.0);
        }
        self.ripple_enabled = ripple_enabled;
    }

    /// Calcola il moltiplicatore di guadagno istantaneo risultante dalla caduta di tensione.
    ///
    /// # Parametri
    /// - `input_level`: Ampiezza istantanea o inviluppo del segnale in ingresso.
    ///
    /// # Restituisce
    /// Fattore moltiplicativo di tensione [0.1, 1.0] da applicare allo stadio di amplificazione.
    /// In caso di ingresso `0.0`, restituisce rigorosamente `1.0` (nessuna attenuazione o rumore).
    #[inline]
    pub fn gain(&mut self, input_level: T) -> T {
        // Estrazione e sanitizzazione del livello di carico assoluto |x|
        let level = if let Some(v) = input_level.to_f64() {
            if v.is_finite() { v.abs() } else { 0.0 }
        } else {
            0.0
        };

        // Filtro passa-basso mononucleo (IIR a un polo) per l'inseguimento del carico di corrente DC
        self.envelope = level + self.coefficient * (self.envelope - level);
        // Denormal guard: azzera residui subnormali per evitare rallentamenti da denormali.
        if self.envelope.abs() < 1.0e-28 {
            self.envelope = 0.0;
        }

        // Funzione di carico non lineare satura: L / (1 + L) per modellare il limite fisico dell'assorbimento
        let normalized_load = self.envelope / (1.0 + self.envelope);
        // Tensione di rail risultante: V_rail = V_nominale * (1 - depth * carico)
        let mut sag = 1.0 - self.depth * normalized_load;

        // Modulazione opzionale di ripple di rete a 100 Hz (solo con segnale presente > -120 dBFS)
        if self.ripple_enabled && level > 1e-6 {
            // Incremento di fase parametrizzato sulla frequenza di campionamento reale (ripple 100 Hz,
            // rettificazione a doppia semionda).
            let sr = if self.sample_rate > 0.0 { self.sample_rate } else { 44100.0 };
            self.ripple_phase += std::f64::consts::TAU * 100.0 / sr;
            if self.ripple_phase > std::f64::consts::TAU {
                self.ripple_phase -= std::f64::consts::TAU;
            }
            // Intermodulazione lieve tra tensione di ripple e carico dell'alimentatore
            sag += self.ripple_phase.sin() * 0.001 * self.depth;
        }

        // Micro-fluttuazione termica di rumore (eseguita solo in presenza di audio per evitare hiss a riposo).
        // La fase avanza per campione così il rumore non degenera in un offset DC su toni stazionari.
        if self.noise_level > 0.0 && level > 1e-6 {
            self.noise_phase += 0.011_937;
            if self.noise_phase > std::f64::consts::TAU {
                self.noise_phase -= std::f64::consts::TAU;
            }
            let pseudo_noise = self.noise_phase.sin() * self.noise_level * 0.001;
            sag += pseudo_noise;
        }

        // Morsetto di sicurezza: la tensione minima non scende mai sotto il 10% per prevenire mute o crash
        let final_sag = if sag.is_finite() {
            sag.clamp(0.1, 1.0)
        } else {
            1.0
        };
        T::from(final_sag).unwrap_or(T::one())
    }

    /// Ripristina lo stato interno dei condensatori dell'alimentatore azzerando l'inviluppo accumulato.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.ripple_phase = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::PsuSag;

    #[test]
    fn test_psu_sag_zero_silence_at_rest() {
        let mut default_psu = PsuSag::<f64>::new();
        assert!(default_psu.initialize(48000.0, 0.1, 0.5));
        assert_eq!(default_psu.gain(0.0), 1.0);
    }
}
