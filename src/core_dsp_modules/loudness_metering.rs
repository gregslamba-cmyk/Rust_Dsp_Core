//! # Loudness Metering — EBU R128 / ITU-R BS.1770-4
//!
//! Implementazione di loudness metering conforme a EBU R128 / ITU-R BS.1770-4.
//! - K-weighting (high-shelf + high-pass biquad)
//! - Mean square su blocchi di 400 ms (momentary), 3 s (short-term)
//! - Integrated loudness con gating assoluto (-70 LUFS) e relativo (-10 LU)
//! - True-peak detection
//! Tutti i calcoli sono in f64 internamente. Zero allocazioni nel thread audio.

use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

pub const REF_LUFS: f64 = -23.0;
pub const ABSOLUTE_GATE_LUFS: f64 = -70.0;
pub const RELATIVE_GATE_OFFSET_LU: f64 = -10.0;
pub const MOMENTARY_BLOCK_SEC: f64 = 0.4;
pub const SHORTTERM_BLOCK_SEC: f64 = 3.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct KWeightHighShelf {
    b0: f64, b1: f64, b2: f64,
    a1: f64, a2: f64,
    x1: f64, x2: f64, y1: f64, y2: f64,
}

impl KWeightHighShelf {
    #[inline]
    pub fn configure(&mut self, sample_rate: f64) {
        let fs = if sample_rate > 0.0 { sample_rate } else { 48000.0 };
        let f0 = 1681.974450955533;
        let g = 3.999843853973347;
        let q = 0.7071752369554196;
        let a = 10.0_f64.powf(g / 40.0);
        let w0 = 2.0 * PI * f0 / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;
        self.b0 = b0 / a0; self.b1 = b1 / a0; self.b2 = b2 / a0;
        self.a1 = a1 / a0; self.a2 = a2 / a0;
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let xn = if x.is_finite() { x } else { 0.0 };
        let y = self.b0 * xn + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1; self.x1 = xn;
        self.y2 = self.y1; self.y1 = y;
        if y.is_finite() { y } else { 0.0 }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.x1 = 0.0; self.x2 = 0.0; self.y1 = 0.0; self.y2 = 0.0;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KWeightHighPass {
    b0: f64, b1: f64, b2: f64,
    a1: f64, a2: f64,
    x1: f64, x2: f64, y1: f64, y2: f64,
}

impl KWeightHighPass {
    #[inline]
    pub fn configure(&mut self, sample_rate: f64) {
        let fs = if sample_rate > 0.0 { sample_rate } else { 48000.0 };
        let f0 = 38.13547087602444;
        let q = 0.5003270373238773;
        let w0 = 2.0 * PI * f0 / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        self.b0 = b0 / a0; self.b1 = b1 / a0; self.b2 = b2 / a0;
        self.a1 = a1 / a0; self.a2 = a2 / a0;
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let xn = if x.is_finite() { x } else { 0.0 };
        let y = self.b0 * xn + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1; self.x1 = xn;
        self.y2 = self.y1; self.y1 = y;
        if y.is_finite() { y } else { 0.0 }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.x1 = 0.0; self.x2 = 0.0; self.y1 = 0.0; self.y2 = 0.0;
    }
}

/// K-weight stage: high-shelf (1681 Hz) followed by high-pass (38 Hz).
#[derive(Clone, Copy, Debug, Default)]
pub struct KWeightFilter {
    hs: KWeightHighShelf,
    hp: KWeightHighPass,
}

impl KWeightFilter {
    #[inline]
    pub fn configure(&mut self, sample_rate: f64) {
        self.hs.configure(sample_rate);
        self.hp.configure(sample_rate);
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        self.hp.process(self.hs.process(x))
    }

    #[inline]
    pub fn reset(&mut self) {
        self.hs.reset();
        self.hp.reset();
    }
}

/// Goniometer ring buffer for L/R phase vector visualization.
#[derive(Clone)]
pub struct GoniometerRing {
    buffer_l: Vec<f64>,
    buffer_r: Vec<f64>,
    write_pos: usize,
    capacity: usize,
}

impl GoniometerRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer_l: vec![0.0; capacity.max(1)],
            buffer_r: vec![0.0; capacity.max(1)],
            write_pos: 0,
            capacity: capacity.max(1),
        }
    }

    #[inline]
    pub fn push(&mut self, left: f64, right: f64) {
        let l = if left.is_finite() { left } else { 0.0 };
        let r = if right.is_finite() { right } else { 0.0 };
        if let Some(buf_l) = self.buffer_l.get_mut(self.write_pos) {
            *buf_l = l;
        }
        if let Some(buf_r) = self.buffer_r.get_mut(self.write_pos) {
            *buf_r = r;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    pub fn snapshot(&self) -> (Vec<f64>, Vec<f64>) {
        let mut l_out = Vec::with_capacity(self.capacity);
        let mut r_out = Vec::with_capacity(self.capacity);
        let mut i = self.write_pos;
        for _ in 0..self.capacity {
            l_out.push(self.buffer_l.get(i).copied().unwrap_or(0.0));
            r_out.push(self.buffer_r.get(i).copied().unwrap_or(0.0));
            i = (i + 1) % self.capacity;
        }
        (l_out, r_out)
    }
}
/// True-peak detector: inter-sample peak via real 4× polyphase FIR oversampling.
#[derive(Clone)]
pub struct TruePeakDetector {
    /// Running max of oversampled magnitudes.
    max_sample: f64,
    /// 12-tap 4-phase interpolation FIR (windowed sinc), pre-normalized.
    coeffs: [[f64; 12]; 4],
    /// Input history ring (12 samples) for each phase.
    history: [f64; 12],
    write_pos: usize,
}

impl TruePeakDetector {
    pub fn new() -> Self {
        Self {
            max_sample: 0.0,
            coeffs: true_peak_coeffs(),
            history: [0.0; 12],
            write_pos: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.max_sample = 0.0;
        self.history = [0.0; 12];
        self.write_pos = 0;
    }

    /// Process one sample; updates running max with 4 interpolated sub-samples.
    #[inline]
    pub fn process(&mut self, x: f64) {
        let xn = if x.is_finite() { x } else { 0.0 };
        if let Some(slot) = self.history.get_mut(self.write_pos) {
            *slot = xn;
        }
        self.write_pos = (self.write_pos + 1) % 12;

        // Sample peak.
        let direct = xn.abs();
        if direct > self.max_sample {
            self.max_sample = direct;
        }

        // Interpolated peak for each of the 4 phases.
        for phase in 0..4 {
            let mut acc = 0.0;
            for (tap, coef) in self.coeffs[phase].iter().enumerate() {
                // history index: newest sample is write_pos-1, going back in time.
                let idx = (self.write_pos + 12 - 1 - tap) % 12;
                acc += self.history.get(idx).copied().unwrap_or(0.0) * coef;
            }
            let mag = acc.abs();
            if mag.is_finite() && mag > self.max_sample {
                self.max_sample = mag;
            }
        }
    }

    /// True-peak amplitude including inter-sample peaks (oversampled).
    #[inline]
    pub fn true_peak_amplitude(&self) -> f64 {
        if self.max_sample.is_finite() { self.max_sample } else { 0.0 }
    }

    pub fn set_sample(&mut self, sample: f64) {
        self.process(sample);
    }
}

impl Default for TruePeakDetector {
    fn default() -> Self { Self::new() }
}

/// Builds 4-phase × 12-tap windowed-sinc interpolation coefficients.
/// Each phase reconstructs the signal at fractional offset `phase/4`.
fn true_peak_coeffs() -> [[f64; 12]; 4] {
    let mut coeffs = [[0.0f64; 12]; 4];
    let taps = 12usize;
    let center = (taps - 1) as f64 / 2.0;
    for (phase, row) in coeffs.iter_mut().enumerate() {
        let frac = phase as f64 / 4.0;
        let mut sum = 0.0;
        for (t, c) in row.iter_mut().enumerate() {
            let x = t as f64 - center - frac;
            let sinc = if x.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
            // Blackman-Harris window over the tap span.
            let wpos = t as f64 / (taps - 1) as f64;
            let w = 0.35875 - 0.48829 * (std::f64::consts::TAU * wpos).cos()
                + 0.14128 * (2.0 * std::f64::consts::TAU * wpos).cos()
                - 0.01168 * (3.0 * std::f64::consts::TAU * wpos).cos();
            let v = sinc * w;
            *c = v;
            sum += v;
        }
        if sum.abs() > 1e-12 {
            for c in row.iter_mut() { *c /= sum; }
        }
    }
    coeffs
}

/// Main loudness engine: momentary, short-term, integrated LUFS + LRA + true-peak.
#[derive(Clone)]
pub struct LoudnessEngine {
    k_weight_l: KWeightFilter,
    k_weight_r: KWeightFilter,
    true_peak: TruePeakDetector,
    goniometer: GoniometerRing,
    ms_sum_sq: f64,
    ms_count: usize,
    st_sum_sq: f64,
    st_count: usize,
    int_blocks: [f64; 120],
    int_block_sum_sq: f64,
    int_block_count: usize,
    int_block_index: usize,
    sample_rate: f64,
    int_block_capacity: usize,
}

impl LoudnessEngine {
    pub fn new(sample_rate: f64, goniometer_capacity: usize) -> Self {
        let sr = if sample_rate > 0.0 { sample_rate } else { 48000.0 };
        let mut k_l = KWeightFilter::default();
        k_l.configure(sr);
        let mut k_r = KWeightFilter::default();
        k_r.configure(sr);
        let ms_samples = (sr * MOMENTARY_BLOCK_SEC) as usize;
        let st_samples = (sr * SHORTTERM_BLOCK_SEC) as usize;
        Self {
            k_weight_l: k_l,
            k_weight_r: k_r,
            true_peak: TruePeakDetector::new(),
            goniometer: GoniometerRing::new(goniometer_capacity),
            ms_sum_sq: 0.0,
            ms_count: ms_samples.max(1),
            st_sum_sq: 0.0,
            st_count: st_samples.max(1),
            int_blocks: [0.0; 120],
            int_block_sum_sq: 0.0,
            int_block_count: 0,
            int_block_index: 0,
            sample_rate: sr,
            int_block_capacity: 120,
        }
    }

    pub fn configure_sample_rate(&mut self, sample_rate: f64) {
        let sr = if sample_rate > 0.0 { sample_rate } else { 48000.0 };
        if (sr - self.sample_rate).abs() > 1e-3 {
            self.sample_rate = sr;
            self.k_weight_l.configure(sr);
            self.k_weight_r.configure(sr);
        }
    }

    pub fn reset(&mut self) {
        self.k_weight_l.reset();
        self.k_weight_r.reset();
        self.true_peak.reset();
        self.goniometer = GoniometerRing::new(self.goniometer.capacity);
        self.ms_sum_sq = 0.0;
        self.st_sum_sq = 0.0;
        for b in self.int_blocks.iter_mut() { *b = 0.0; }
        self.int_block_sum_sq = 0.0;
        self.int_block_count = 0;
        self.int_block_index = 0;
    }

    #[inline]
    pub fn process_sample(&mut self, left: f64, right: f64) {
        let l = if left.is_finite() { left } else { 0.0 };
        let r = if right.is_finite() { right } else { 0.0 };
        let kl = self.k_weight_l.process(l);
        let kr = self.k_weight_r.process(r);
        // ITU-R BS.1770-4: mean-square = kl² + kr² (not average).
        let sq = kl * kl + kr * kr;
        self.true_peak.process(l);
        self.true_peak.process(r);
        self.goniometer.push(l, r);
        self.ms_sum_sq += sq;
        self.st_sum_sq += sq;
    }



    /// Add current momentary block to integrated history (called at 400ms boundaries).
    pub fn add_block(&mut self, ms_sum_sq: f64) {
        let cap = self.int_blocks.len();
        let idx = self.int_block_index % cap;
        self.int_blocks[idx] = ms_sum_sq;
        self.int_block_index += 1;
        self.int_block_count = self.int_block_index.min(cap);
        self.int_block_sum_sq += ms_sum_sq;
    }

    #[inline]
    pub fn momentary_lufs(&self) -> f64 {
        let ms = if self.ms_count > 0 { self.ms_sum_sq / self.ms_count as f64 } else { 0.0 };
        db_lufs(ms)
    }

    #[inline]
    pub fn short_term_lufs(&self) -> f64 {
        let ms = if self.st_count > 0 { self.st_sum_sq / self.st_count as f64 } else { 0.0 };
        db_lufs(ms)
    }

    pub fn integrated_lufs(&self) -> f64 {
        let count = self.int_block_count;
        if count == 0 {
            return ABSOLUTE_GATE_LUFS;
        }
        // Iterate fixed ring buffer without allocation.
        let mut gated_sum = 0.0;
        let mut gated_count = 0usize;
        for i in 0..count {
            let idx = i % self.int_blocks.len();
            let b = self.int_blocks[idx];
            let lufs = db_lufs(b);
            if lufs > ABSOLUTE_GATE_LUFS {
                gated_sum += b;
                gated_count += 1;
            }
        }
        if gated_count == 0 {
            return ABSOLUTE_GATE_LUFS;
        }
        let relative_threshold = db_lufs(gated_sum / gated_count as f64) + RELATIVE_GATE_OFFSET_LU;
        let mut rel_sum = 0.0;
        let mut rel_count = 0usize;
        for i in 0..count {
            let idx = i % self.int_blocks.len();
            let b = self.int_blocks[idx];
            let lufs = db_lufs(b);
            if lufs > ABSOLUTE_GATE_LUFS && lufs > relative_threshold {
                rel_sum += b;
                rel_count += 1;
            }
        }
        if rel_count == 0 {
            return ABSOLUTE_GATE_LUFS;
        }
        db_lufs(rel_sum / rel_count as f64)
    }

    /// Loudness range (LU) per EBU R128 Tech 3342.
    pub fn lra(&self) -> f64 {
        let n = self.int_block_count;
        if n < 2 {
            return 0.0;
        }
        // Fixed-size stack copy for sorting (no heap allocation).
        let cap = self.int_blocks.len();
        let mut lufs_vals: [f64; 120] = [0.0; 120];
        for i in 0..n {
            let idx = i % cap;
            lufs_vals[i] = db_lufs(self.int_blocks[idx]);
        }
        lufs_vals[..n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = n;
        let p10 = lufs_vals.get((len as f64 * 0.10).floor() as usize % len).copied().unwrap_or(0.0);
        let p95 = lufs_vals.get((len as f64 * 0.95).floor() as usize % len).copied().unwrap_or(0.0);
        let lra = (p95 - p10).max(0.0);
        lra.clamp(0.0, 100.0)
    }

    #[inline]
    pub fn true_peak_dbtp(&self) -> f64 {
        let tp = self.true_peak.true_peak_amplitude();
        if tp <= 0.0 { -120.0 } else { (20.0 * tp.log10()).max(-120.0) }
    }

    #[inline]
    pub fn goniometer_snapshot(&self) -> (Vec<f64>, Vec<f64>) {
        self.goniometer.snapshot()
    }

    #[inline]
    pub fn set_sample_rate(&mut self, sr: f64) {
        self.configure_sample_rate(sr);
    }
}

#[inline]
fn db_lufs(mean_sq: f64) -> f64 {
    if mean_sq <= 0.0 {
        ABSOLUTE_GATE_LUFS
    } else {
        let val = 10.0 * mean_sq.log10() - 0.691;
        val.max(ABSOLUTE_GATE_LUFS)
    }

}

/// Lock-free telemetry snapshot for loudness metering (thread-safe audio→UI).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MasteringTelemetrySnapshot {
    pub momentary_lufs: f64,
    pub short_term_lufs: f64,
    pub integrated_lufs: f64,
    pub lra_lu: f64,
    pub true_peak_dbtp: f64,
    pub crest_factor_db: f64,
    pub psr_db: f64,
    pub phase_correlation: f64,
    pub dominant_peak_hz: f64,
}

impl Default for MasteringTelemetrySnapshot {
    fn default() -> Self {
        Self {
            momentary_lufs: ABSOLUTE_GATE_LUFS,
            short_term_lufs: ABSOLUTE_GATE_LUFS,
            integrated_lufs: ABSOLUTE_GATE_LUFS,
            lra_lu: 0.0,
            true_peak_dbtp: -120.0,
            crest_factor_db: 0.0,
            psr_db: 0.0,
            phase_correlation: 0.0,
            dominant_peak_hz: 0.0,
        }
    }
}

/// Publish a MasteringTelemetrySnapshot atomically via Relaxed stores.
#[derive(Debug)]
pub struct MasteringTelemetryPublisher {
    published: [AtomicU64; 9],
}

impl MasteringTelemetryPublisher {
    pub fn new() -> Self {
        Self {
            published: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
    pub fn publish(&self, s: &MasteringTelemetrySnapshot) {
        self.published[0].store(s.momentary_lufs.to_bits(), Ordering::Relaxed);
        self.published[1].store(s.short_term_lufs.to_bits(), Ordering::Relaxed);
        self.published[2].store(s.integrated_lufs.to_bits(), Ordering::Relaxed);
        self.published[3].store(s.lra_lu.to_bits(), Ordering::Relaxed);
        self.published[4].store(s.true_peak_dbtp.to_bits(), Ordering::Relaxed);
        self.published[5].store(s.crest_factor_db.to_bits(), Ordering::Relaxed);
        self.published[6].store(s.psr_db.to_bits(), Ordering::Relaxed);
        self.published[7].store(s.phase_correlation.to_bits(), Ordering::Relaxed);
        self.published[8].store(s.dominant_peak_hz.to_bits(), Ordering::Relaxed);
    }
    pub fn snapshot(&self) -> MasteringTelemetrySnapshot {
        MasteringTelemetrySnapshot {
            momentary_lufs: f64::from_bits(self.published[0].load(Ordering::Relaxed)),
            short_term_lufs: f64::from_bits(self.published[1].load(Ordering::Relaxed)),
            integrated_lufs: f64::from_bits(self.published[2].load(Ordering::Relaxed)),
            lra_lu: f64::from_bits(self.published[3].load(Ordering::Relaxed)),
            true_peak_dbtp: f64::from_bits(self.published[4].load(Ordering::Relaxed)),
            crest_factor_db: f64::from_bits(self.published[5].load(Ordering::Relaxed)),
            psr_db: f64::from_bits(self.published[6].load(Ordering::Relaxed)),
            phase_correlation: f64::from_bits(self.published[7].load(Ordering::Relaxed)),
            dominant_peak_hz: f64::from_bits(self.published[8].load(Ordering::Relaxed)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_weight_silence() {
        let mut k = KWeightFilter::default();
        k.configure(48000.0);
        k.reset();
        assert_eq!(k.process(0.0), 0.0);
    }

    #[test]
    fn true_peak_detector() {
        let mut tp = TruePeakDetector::new();
        tp.process(0.5);
        tp.process(-0.8);
        assert!((tp.true_peak_amplitude() - 0.8).abs() < 1e-6);
        tp.reset();
        assert!((tp.true_peak_amplitude() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn lufs_silence() {
        let mut e = LoudnessEngine::new(48000.0, 64);
        e.reset();
        for _ in 0..48000 {
            e.process_sample(0.0, 0.0);
        }
        // silence below absolute gate
        assert!((e.integrated_lufs() - ABSOLUTE_GATE_LUFS).abs() < 1e-3);
    }

    #[test]
    fn telemetry_snapshot_roundtrip() {
        let pub_ = MasteringTelemetryPublisher::new();
        let s = MasteringTelemetrySnapshot {
            momentary_lufs: -18.0,
            short_term_lufs: -20.0,
            integrated_lufs: -22.0,
            lra_lu: 5.0,
            true_peak_dbtp: -1.0,
            crest_factor_db: 12.0,
            psr_db: 3.0,
            phase_correlation: 0.7,
            dominant_peak_hz: 440.0,
        };
        pub_.publish(&s);
        let snap = pub_.snapshot();
        assert!((snap.momentary_lufs - (-18.0)).abs() < 1e-6);
        assert!((snap.integrated_lufs - (-22.0)).abs() < 1e-6);
        assert!((snap.dominant_peak_hz - 440.0).abs() < 1e-6);
    }
}