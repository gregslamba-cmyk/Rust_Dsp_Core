================================================================================
CORE-DSP — LIBRERIA AUDIO DSP CONDIVISA AD ALTA PRECISIONE (f64)
================================================================================

core-dsp e' il motore DSP condiviso a 64-bit dell'intero ecosistema Analog-Vibe.
Progettato per garantire la massima fedelta' acustica, stabilita' numerica e
rispetto rigoroso dei vincoli dei sistemi audio in tempo reale (real-time safe).


--------------------------------------------------------------------------------
1. PRINCIPI FONDAMENTALI & GARANZIE REAL-TIME
--------------------------------------------------------------------------------

- Precisione di Calcolo a 64-bit (f64):
  Tutti i coefficienti dei filtri, le equazioni differenziali non lineari, le
  linee di ritardo e i modelli fisici operano nativamente in doppia precisione
  f64 per eliminare rumore di quantizzazione ed errori di accumulo ricorsivo.

- Assenza Totale di Codice Non Sicuro (#![forbid(unsafe_code)]):
  Nessun blocco unsafe consentito all'interno della libreria. La correttezza e'
  garantita interamente dal type-system e dal borrow-checker di Rust.

- Zero Allocazioni Dinamiche nel Thread Audio:
  Nessun uso di Vec::new(), Box::new(), String o ridimensionamento di collezioni
  nel callback audio process() o process_sample(). Tutta la memoria (buffer,
  delay lines, tabelle FFT, stati dei filtri) deve essere pre-allocata durante
  le chiamate di configurazione iniziali: initialize() o prepare().

- Zero Lock Bloccanti:
  Nessun mutex bloccante (std::sync::Mutex) nel percorso audio. Lo scambio di dati
  tra thread di calcolo audio e thread GUI avviene esclusivamente tramite atomici
  (AtomicU64, AtomicU32, AtomicBool con Ordering::Relaxed) o strutture lock-free.

- Silenzio Digitale Assoluto a Riposo:
  Con ingresso 0.0, l'uscita prodotta da qualsiasi modulo e' rigorosamente 0.0.
  I generatori di rumore termico (Johnson-Nyquist) e il ripple di alimentazione
  sono tassativamente disattivati di default (OFF / 0.0).

- Sanitizzazione Numerica Continua:
  Tutti i moduli integrano controlli automatici contro NaN (Not a Number) e Inf
  (Infinito), con ripristino immediato a 0.0 e isolamento tramite PanicProtection.

- Routing Dinamico Multicanale (Mono -> 7.1 Surround):
  Supporto universale da 1 a 8 canali audio indipendenti gestito tramite
  RoutingMatrix (Mono, Stereo, LCR, Quad, 5.0, 5.1, 7.0, 7.1).


--------------------------------------------------------------------------------
2. ARCHITETTURA DEI MODULI PER FAMIGLIE
--------------------------------------------------------------------------------

La libreria core-dsp e' organizzata nelle seguenti macro-famiglie di moduli:

2.1 MODELLAZIONE FISICA COMPONENTI & NON-LINEARITA'
  - component_models:
    Modelli discreti a componenti fisici per circuiti vintage:
    * Tubi termoionici: doppi triodi ECC83 / 12AX7, 12AT7 (polarizzazione catodica,
      saturazione asimmetrica, griglia) e pentodo EF86.
    * Semiconduttori & Op-Amp discreti: API 2520 (Class A discreto), NE5534,
      coppia complementare Neve 2N3055 / BC108, transistor JFET 2N5457, celle
      VCA DBX 202 Gold Can.
    * Trasformatori audio: UTC O-12, UTC A-10, UTC HA-100X, Marinair LO1166,
      Carnhill VTB9045, API 2623.
  - transformer_hysteresis:
    Modello fisico di isteresi magnetica basato sulle equazioni differenziali di
    Jiles-Atherton. Cattura saturazione del nucleo, distorsione armonica dispari
    a bassa frequenza e perdite per correnti parassite (eddy currents).
  - nonlinear_stages:
    Stadi di saturazione modellati: roll-off dinamico Effetto Miller (filtro
    passa-basso a 1 polo dinamico che attenua le altissime frequenze all'aumentare
    del drive), clipping simmetrico/asimmetrico, ladder di diodi, wavefolding.
  - psu_sag:
    Modellazione del calo di tensione dell'alimentatore condiviso (rail droop / sag)
    sotto carico impulsivo istantaneo, generando compressione dinamica naturale.
  - analog_variance:
    Deriva termica stocastica aggiornata a block-rate (zero calcoli trigonometrici
    per campione) tramite oscillatori asincroni a bassa frequenza guidati da seed
    deterministico per variazioni realistiche canale/istanza.

2.2 SIMULAZIONE ACUSTICA, STANZE 3D & CABINET
  - cabinet_speaker:
    Emulazione fisica di altoparlanti e cabinet per chitarra e basso:
    * Modello non lineare di escursione del cono con curve Klippel Bl(x)
      (forza magnetica) e Kms(x) (cedevolezza delle sospensioni).
    * Compressione termica dinamica della bobina (voice-coil thermal heating).
    * Breakup asimmetrico del cono ad alti volumi.
    * Risonanza del mobile (1-polo risonante per casse chiuse o bass-reflex).
    * Posizionamento continuo di microfoni virtuali (CK12, U47, SM57) con circuiti
      preamp dedicati a valvole o trasformatori.
  - acoustic_room:
    Simulatore acustico ambientale 3D basato sul metodo Image-Source fino al
    2° ordine (30 sorgenti virtuali per canale calcolate in initialize()),
    combinato con una rete di diffusione FDN per il campo riverberato tardo.

2.3 AUDIO SPAZIALE & IMMERSIVO
  - spatial_3d:
    Motore di posizionamento tridimensionale immersivo:
    * Rendering Binaurale con filtri HRIR interpolati.
    * Higher-Order Ambisonics (HOA) fino al 3° ordine con supporto normalizzazioni
      Furse-Malham e AmbiNorm (SN3D / N3D).
    * Panner VBAP (Vector Base Amplitude Panning) configurabile su 1..8 altoparlanti.

2.4 METERING & ANALISI DELLA DINAMICA
  - loudness_metering:
    Engine conforme agli standard broadcast internazionali ITU-R BS.1770-4 / EBU R128:
    * Banco filtri K-weighting (KWeightHighShelf + KWeightHighPass).
    * True-Peak Detector a sovracampionamento inter-sample (1.25x).
    * Misura LUFS Momentaneo (400 ms), Short-Term (3 s) e Integrato con gating
      assoluto a -70 LUFS e relativo a -10 LU.
    * Misura del Loudness Range (LRA) e goniometro L/R circolare lock-free.
    * MasteringTelemetryPublisher con array [AtomicU64; 9] per aggiornamenti thread-safe.
  - metering:
    Modulo condiviso per il calcolo di picco, RMS e Gain Reduction (GR) per canale.
  - loudness_rider:
    Detector di loudness sillabica con lookahead e rampa regolabile in dB/s per
    automazioni vocali trasparenti (stile Vocal Rider).

2.5 ROUTING & GRAFO DI CALCOLO AUDIO
  - dsp_graph:
    Motore di compilazione ed esecuzione di grafi DSP audio (AudioGraphBuilder e
    CompiledAudioGraph):
    * Trait uniforme DspGraphNode: initialize(), reset(), latency_samples(), process_node().
    * Rilevamento cicli non ammessi (tranne feedback espliciti FeedbackDelayZ1Node).
    * Plugin Delay Compensation (PDC) automatico: calcola le latenze dei rami paralleli
      e inserisce automaticamente ritardi compensativi per garantire coerenza di fase.
    * Pre-allocazione statica di tutti i buffer intermedi AudioEdgeBuffer fino a 8 canali.
    * Nodi built-in: GainPanNode, SplitterNode, SummerMixerNode, ParallelDryWetNode, FeedbackDelayZ1Node.
  - routing_matrix:
    Gestione layout multicanale (1..8 ch) e matrice di codifica/decodifica Mid/Side.
  - analog_summing:
    Fisica della somma analogica da console:
    * AdjacentCrosstalkMatrix: diafonia induttivo-capacitiva filtrata tra canali adiacenti.
    * VirtualEarthSummer: simulatore del nodo di somma virtual earth fino a 32 canali
      con carico sull'alimentatore e saturazione non lineare.

2.6 FILTRI, EQ & CONVOLUZIONE
  - zdf_filters:
    Filtri State Variable (SVF) Zero-Delay Feedback con risoluzione implicita e
    saturazione non lineare negli integratori (Low-Pass, High-Pass, Band-Pass, Notch).
  - surgical_eq:
    Equalizzatore parametrico chirurgico a filtri biquad ad alta precisione.
  - linear_phase_fir:
    Filtri FIR simmetrici a fase lineare (finestrati Kaiser / Blackman) e crossover
    multibanda a perfetta ricostruzione (flat magnitude & phase).
  - partitioned_convolver:
    Motore di convoluzione partizionata uniforme (Overlap-Save su blocchi FFT)
    per risposte impulsive a bassissima latenza.
  - fft_convolution:
    Analizzatore spettrale STFT lock-free per elaborazioni nel dominio della frequenza.

2.7 MODULAZIONE, DELAY & TEMPO
  - delay_lines:
    Linee di ritardo con lettura frazionaria continua (interpolazione cubica e
    All-Pass) prive di zipper noise.
  - tempo_sync:
    Sincronizzazione BPM: modalita' Free (ms) e Host BPM Sync con tutte le
    suddivisioni musicali (1/1..1/32, Dotted, Triplet, Swing).
  - fdn_reverb:
    Riverbero Feedback Delay Network con matrici di rotazione ortogonali (Hadamard
    / Householder), damping ZDF e diffusori all-pass.
  - tape_physics:
    Fisica del nastro magnetico (wow & flutter, saturazione di flusso, curve NAB/CCIR).

2.8 RESTAURO, PITCH & CAMPIONAMENTO
  - pitch_shifter: Shifter di intonazione e formanti indipendenti ad alta trasparenza.
  - spectral_analyzer: Analizzatore FFT a frazioni d'ottava con ring-buffer lock-free.
  - spectral_restoration: Algoritmi di de-noise, de-click, de-hum e de-clip.
  - mix_mastering_fx: Processori stereo, clipper morbidi e saturatori di bus.
  - mpc_sampler_engine: Emulazione di convertitori e colorazioni vintage sampler.

2.9 SICUREZZA, CONTROLLI & ESPORTAZIONE
  - panic_protection: Barriera di cattura unwinding e sanitizzazione NaN/Inf.
  - realtime_controls: Interpolatori di parametri smussati anti-zipper.
  - resampler_bitdepth: Sovracampionamento sincrono a fase lineare anti-alias.
  - plugin_export: Macro export_greg_plugin! per esportare nativamente CLAP + VST3.


--------------------------------------------------------------------------------
3. ESEMPIO DI UTILIZZO PER SVILUPPATORI
--------------------------------------------------------------------------------

Esempio di inizializzazione e processamento con stadio a trasformatore e saturazione:

```rust
use core_dsp::core_dsp_modules::{
    component_models::{create_transformer, TransformerModel, TransformerStage},
    nonlinear_stages::MillerRollOff,
    routing_matrix::{RoutingMatrix, RoutingMode},
};

pub struct MyProcessor {
    routing: RoutingMatrix,
    input_xfmr: [TransformerStage; 2],
    miller_filter: [MillerRollOff; 2],
}

impl MyProcessor {
    pub fn new() -> Self {
        Self {
            routing: RoutingMatrix::new(),
            input_xfmr: [
                create_transformer(TransformerModel::UtcO12),
                create_transformer(TransformerModel::UtcO12),
            ],
            miller_filter: [MillerRollOff::new(), MillerRollOff::new()],
        }
    }

    pub fn initialize(&mut self, sample_rate: f64) {
        self.routing.configure(2, RoutingMode::Stereo);
        for ch in 0..2 {
            self.input_xfmr[ch].initialize(sample_rate);
            self.miller_filter[ch].initialize(sample_rate);
        }
    }

    pub fn process_sample(&mut self, left: f64, right: f64, drive: f64) -> (f64, f64) {
        let x0 = self.input_xfmr[0].process_sample(left * drive);
        let x1 = self.input_xfmr[1].process_sample(right * drive);
        let y0 = self.miller_filter[0].process(x0, drive);
        let y1 = self.miller_filter[1].process(x1, drive);
        (y0, y1)
    }
}
```


--------------------------------------------------------------------------------
4. COMANDI DI TEST E VERIFICA
--------------------------------------------------------------------------------

Per compilare e verificare l'integrita' e i test unitari deterministici di core-dsp:

  # Controllo statico
  cargo check -p core-dsp
  cargo clippy -p core-dsp -- -D warnings

  # Esecuzione di tutti i test unitari (oltre 150 test, verifica 0.0 in -> 0.0 out)
  cargo test -p core-dsp
================================================================================
