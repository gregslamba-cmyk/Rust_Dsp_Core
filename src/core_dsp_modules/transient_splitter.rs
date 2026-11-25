//! # Transient Splitter
//!
//! Separazione real-time zero-alloc a ricostruzione perfetta tra la componente
//! impulsiva (*Transient*) e la componente continua (*Sustain*):
//! `transient + sustain == input`. `f64`, zero allocazioni in `process()`.

/// Parametri dello splitter transiente.
pub struct TransientSplitter {
    sample_rate: f64,
    /// Stato del filtro passa-basso che traccia la componente sostenuta.
    sustain_state: f64,
    /// Coefficiente di tracking del sustain (0..1).
    alpha: f64,
    /// Soglia di rilevazione transiente (0..1).
    threshold: f64,
    prev_envelope: f64,
}

impl Default for TransientSplitter {
    fn default() -> Self { Self::new() }
}

impl TransientSplitter {
    /// Crea un nuovo splitter transiente.
    pub fn new() -> Self {
        Self {
            sample_rate: 48_000.0,
            sustain_state: 0.0,
            alpha: 0.005,
            threshold: 0.3,
            prev_envelope: 0.0,
        }
    }

    /// Configura il sample rate e il tempo di tracciamento del sustain (ms).
    pub fn initialize(&mut self, sample_rate: f64, sustain_ms: f64) -> bool {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return false;
        }
        self.sample_rate = sample_rate;
        let tau = sustain_ms.max(0.1) / 1000.0;
        self.alpha = 1.0 - (-1.0 / (tau * sample_rate)).exp();
        self.reset();
        true
    }

    /// Imposta la soglia di rilevazione transiente (0..1).
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = if threshold.is_finite() { threshold.clamp(0.0, 1.0) } else { 0.3 };
    }

    /// Reset dello stato.
    pub fn reset(&mut self) {
        self.sustain_state = 0.0;
        self.prev_envelope = 0.0;
    }

    /// Processa un campione: ritorna `(transient, sustain)`.
    #[inline]
    pub fn process(&mut self, input: f64) -> (f64, f64) {
        let x = if input.is_finite() { input } else { 0.0 };
        // Envelope del segnale in ingresso (seguimento del picco).
        let ax = x.abs();
        let envelope = if ax > self.prev_envelope {
            ax
        } else {
            self.prev_envelope - self.alpha * (self.prev_envelope - ax).max(0.0)
        };
        self.prev_envelope = envelope;
        // Sustain: tracciamento lento dell'inviluppo.
        self.sustain_state += self.alpha * (envelope - self.sustain_state);
        self.sustain_state = self.sustain_state.clamp(0.0, 1.0);
        // Transiente: residuo sopra il sustain ponderato dalla soglia.
        let resid = x - self.sustain_state;
        let transient = if resid.abs() > self.threshold * envelope.max(1e-6) { resid } else { 0.0 };
        let sustain = x - transient;
        if transient.is_finite() && sustain.is_finite() { (transient, sustain) } else { (0.0, 0.0) }
    }

    /// Processa un blocco di campioni.
    pub fn process_block(&mut self, input: &[f64], transient: &mut [f64], sustain: &mut [f64]) {
        let len = input.len().min(transient.len()).min(sustain.len());
        for i in 0..len {
            let (t, s) = self.process(input.get(i).copied().unwrap_or(0.0));
            if let Some(tslot) = transient.get_mut(i) { *tslot = t; }
            if let Some(sslot) = sustain.get_mut(i) { *sslot = s; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_at_rest() {
        let mut ts = TransientSplitter::new();
        assert!(ts.initialize(48_000.0, 50.0));
        for _ in 0..512 {
            let (t, s) = ts.process(0.0);
            assert_eq!(t, 0.0);
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn reconstruction_perfect() {
        let mut ts = TransientSplitter::new();
        ts.initialize(48_000.0, 50.0);
        for i in 0..1024 {
            let x = (i as f64 * 0.07).sin() * 0.5;
            let (t, s) = ts.process(x);
            assert!((t + s - x).abs() < 1e-12);
        }
    }

    #[test]
    fn finite_output() {
        let mut ts = TransientSplitter::new();
        ts.initialize(48_000.0, 50.0);
        for i in 0..512 {
            let (t, s) = ts.process(f64::NAN);
            assert!(t.is_finite() && s.is_finite());
        }
        let input: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut tr = vec![f64::NAN; 256];
        let mut su = vec![f64::NAN; 256];
        ts.process_block(&input, &mut tr, &mut su);
        for v in tr { assert!(v.is_finite()); }
        for v in su { assert!(v.is_finite()); }
    }
}
