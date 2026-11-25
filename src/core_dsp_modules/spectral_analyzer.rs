// Spectral Analyzer module
pub use crate::core_dsp_modules::fft_convolution::StftAnalyzer as SpectralAnalyzer;

/// Long-Term Spectral Average (LTSA) di riferimento spettrale per il bilanciamento tonale.
/// Accumulatore esponenziale della magnitudine spettrale, zero allocazioni in `update()`.
pub struct TonalReferenceTracker {
    accumulator: Vec<f64>,
    count: usize,
    alpha: f64,
    complex_len: usize,
}

impl Default for TonalReferenceTracker {
    fn default() -> Self { Self::new(512) }
}

impl TonalReferenceTracker {
    /// Crea un nuovo tracker con la lunghezza spettrale specificata.
    pub fn new(complex_len: usize) -> Self {
        Self {
            accumulator: vec![0.0; complex_len],
            count: 0,
            alpha: 0.01,
            complex_len,
        }
    }

    /// Aggiorna il riferimento con uno spettro di magnitudine (uno frame STFT).
    pub fn update(&mut self, spectrum: &[rustfft::num_complex::Complex<f64>]) {
        let n = self.complex_len.min(spectrum.len());
        let scale = self.alpha;
        for i in 0..n {
            let mag = spectrum.get(i).map(|c| c.norm()).unwrap_or(0.0);
            if mag.is_finite() {
                if let Some(slot) = self.accumulator.get_mut(i) {
                    *slot = scale * mag + (1.0 - scale) * *slot;
                }
            }
        }
        self.count = self.count.saturating_add(1);
    }

    /// Ritorna lo spettro medio accumulato (LTSA).
    pub fn average_spectrum(&self) -> &[f64] {
        &self.accumulator
    }

    /// Numero di frame processati.
    pub fn frame_count(&self) -> usize { self.count }

    /// Reset dello stato.
    pub fn reset(&mut self) {
        self.accumulator.fill(0.0);
        self.count = 0;
    }
}
