//! # Universal Sidechain Processor & Detector Filtering
//!
//! Questo modulo fornisce l'infrastruttura condivisa per il trattamento del segnale pilota
//! (detector / key) nei processori di dinamica (compressori, limiter, expander, transient shaper).
//!
//! ## Funzionalità Principali
//! - **Routing Chiave Flessibile:** Supporto nativo per sidechain interna (il segnale stesso) ed
//!   esterna (bus aux da altra traccia DAW), con fallback automatico e sicuro su quella interna
//!   qualora l'ingresso esterno risulti disconnesso o non finito (`NaN` / `Inf`).
//! - **Filtraggio Completo del Detector:** Equalizzazione mirata del percorso di rilevamento con 5 topologie
//!   biquad selezionabili (High-Pass per evitare il pompaggio sulle sub-basse, Low-Pass per de-essing,
//!   Bell parametrico e filtri a mensola Low/High-Shelf).
//! - **Modalità Sidechain Listen:** Permette all'utente di ascoltare direttamente il segnale filtrato
//!   del detector prima del raddrizzatore per tarare con precisione frequenza e risonanza del filtro.

use num_traits::Float;

use super::filters::{Biquad, FilterType};

use nih_plug::prelude::Enum;

/// Modalità di linking del detector multicanale per processori di dinamica.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Enum)]
pub enum MultiChannelLinkMode {
    /// Multi-Mono / Dual Mono: ogni canale ha detector e gain reduction indipendenti.
    #[default]
    #[id = "unlinked"]
    #[name = "Unlinked"]
    Unlinked,
    /// Stereo Link / Bus Link: gain reduction comune su tutti i canali attivi.
    #[id = "all_linked"]
    #[name = "All Linked"]
    AllLinked,
    /// Surround Stem Link (5.1/7.1): calcola GR comune su Front + Surround escludendo LFE (.1).
    #[id = "surround_stem_link"]
    #[name = "Surround Stem Link"]
    SurroundStemLink,
    /// Mid/Side: detector separati per componenti Mid e Side.
    #[id = "mid_side"]
    #[name = "Mid/Side"]
    MidSide,
}

/// Topologie di filtro disponibili per la modellazione della risposta in frequenza del detector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Enum)]
pub enum SidechainFilter {
    /// Filtro passa-alto (HPF): attenua le basse frequenze per evitare pompaggio eccessivo della cassa/basso.
    #[default]
    HighPass,
    /// Filtro passa-basso (LPF): isola le basse o medie frequenze escludendo i picchi di alte.
    LowPass,
    /// Filtro a campana (Bell): enfatizza o attenua una banda specifica (es. 2-5 kHz per de-essing o rullante).
    Bell,
    /// Filtro a mensola bassa (Low-Shelf): modifica il peso energetico delle basse frequenze nel detector.
    LowShelf,
    /// Filtro a mensola alta (High-Shelf): modula la sensibilità del compressore sui piatti e sulle sibilanti.
    HighShelf,
}

impl From<SidechainFilter> for FilterType {
    fn from(value: SidechainFilter) -> Self {
        match value {
            SidechainFilter::HighPass => Self::HighPass,
            SidechainFilter::LowPass => Self::LowPass,
            SidechainFilter::Bell => Self::Bell,
            SidechainFilter::LowShelf => Self::LowShelf,
            SidechainFilter::HighShelf => Self::HighShelf,
        }
    }
}

/// Processore di filtraggio e selezione sorgente per il detector di dinamica.
#[derive(Clone, Copy, Debug)]
pub struct SidechainProcessor<T: Float> {
    /// Filtro biquad dedicato al percorso sidechain.
    filter: Biquad<T>,
    /// Flag di attivazione del filtro di equalizzazione sidechain.
    filter_enabled: bool,
    /// Flag di attivazione della modalità di ascolto diretto "Sidechain Listen".
    listen: bool,
}

impl<T: Float> Default for SidechainProcessor<T> {
    fn default() -> Self {
        Self {
            filter: Biquad::default(),
            filter_enabled: false,
            listen: false,
        }
    }
}

impl<T: Float> SidechainProcessor<T> {
    /// Configura i parametri del filtro biquad sidechain (frequenza di taglio, Q e guadagno).
    ///
    /// # Parametri
    /// - `sample_rate`: Frequenza di campionamento in Hz.
    /// - `kind`: Tipo di filtro (`FilterType`).
    /// - `frequency`: Frequenza di taglio o centrale in Hz.
    /// - `q`: Fattore di risonanza o larghezza di banda Q.
    /// - `gain_db`: Guadagno in dB (usato per Bell e filtri Shelf).
    pub fn configure(
        &mut self,
        sample_rate: T,
        kind: FilterType,
        frequency: T,
        q: T,
        gain_db: T,
    ) -> bool {
        self.filter
            .configure(sample_rate, kind, frequency, q, gain_db)
    }

    /// Abilita o disabilita l'ascolto diretto del segnale sidechain prima dell'inviluppo.
    pub fn set_listen(&mut self, listen: bool) {
        self.listen = listen;
    }

    /// Abilita o disabilita il passaggio del segnale attraverso il filtro sidechain.
    pub fn set_filter_enabled(&mut self, enabled: bool) {
        self.filter_enabled = enabled;
    }

    /// Processa un campione mono utilizzando la sola sorgente interna.
    #[inline]
    pub fn process(&mut self, input: T) -> T {
        self.process_key(input, None, false)
    }

    /// Processa il segnale di controllo scegliendo dinamicamente tra sorgente interna ed esterna.
    ///
    /// # Parametri
    /// - `internal`: Campione del canale audio principale interno.
    /// - `external`: Campione proveniente dal bus sidechain ausiliario opzionale.
    /// - `use_external`: Flag per forzare l'uso della chiave esterna se presente e finita.
    ///
    /// # Restituisce
    /// Ampiezza raddrizzata per l'envelope follower (oppure segnale bipolare grezzo in modalità Listen).
    #[inline]
    pub fn process_key(&mut self, internal: T, external: Option<T>, use_external: bool) -> T {
        // Selezione sorgente: se l'esterno è richiesto ma non valido/connesso, fallback sicuro all'interno
        let source = if use_external {
            external
                .filter(|sample| sample.is_finite())
                .unwrap_or(internal)
        } else {
            internal
        };

        // Applicazione del filtraggio biquad di equalizzazione del detector se abilitato
        let filtered = if self.filter_enabled {
            self.filter.process(source)
        } else {
            source
        };

        // In modalità Listen restituisce il segnale bipolare per il monitoraggio acustico;
        // per il calcolo dell'inviluppo restituisce il modulo assoluto (raddrizzamento a doppia semionda)
        if self.listen {
            filtered
        } else {
            filtered.abs()
        }
    }

    /// Resetta la memoria interna del filtro biquad sidechain azzerando i ritardi.
    pub fn reset(&mut self) {
        self.filter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{SidechainFilter, SidechainProcessor};

    #[test]
    fn external_key_is_selected_and_invalid_external_falls_back_to_internal() {
        let mut processor = SidechainProcessor::<f32>::default();
        assert!(processor.configure(
            48_000.0,
            SidechainFilter::LowPass.into(),
            1_000.0,
            0.707,
            0.0
        ));
        processor.set_listen(true);

        assert!(processor.process_key(0.25, Some(-0.5), true) < 0.0);
        processor.reset();
        assert!(processor.process_key(0.25, Some(f32::NAN), true) > 0.0);
        processor.set_listen(false);
        assert!(processor.process_key(-0.25, Some(0.5), false) > 0.0);
    }

    #[test]
    fn exposes_supported_sidechain_filters() {
        let filters = [
            SidechainFilter::HighPass,
            SidechainFilter::LowPass,
            SidechainFilter::Bell,
            SidechainFilter::LowShelf,
            SidechainFilter::HighShelf,
        ];
        for filter in filters {
            let mut processor = SidechainProcessor::<f32>::default();
            assert!(processor.configure(48_000.0, filter.into(), 1_000.0, 0.707, 3.0));
        }
    }
}
