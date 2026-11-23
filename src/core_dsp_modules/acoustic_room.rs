//! # Acoustic Room — Image-Source + Diffusion Reverb
//!
//! Physical-model-inspired room acoustics simulation.
//! - Image-source method up to 2nd order (30 image sources per channel), pre-allocated
//!   in `initialize()`, zero heap in `process()`.
//! - Late-field diffusion stage (nested all-pass + feedback delay network).
//! - Supports arbitrary room dimensions, source/listener positions, wall absorption.
//!
//! ## Real-time safety
//! All DSP buffers pre-allocated in `initialize()`. `process_sample` performs zero
//! heap allocations, no locks, no I/O, and sanitizes NaN/Inf to 0.0.

use crate::core_dsp_modules::delay_lines::DelayLine;
use crate::core_dsp_modules::nonlinear_stages::flush_denormal;

/// Image source for early reflections.
#[derive(Clone, Copy, Debug)]
pub struct ImageSource {
    #[allow(dead_code)]
    delay: f64,
    gain: f64,
}

/// Room dimensions and absorption parameters.
#[derive(Clone, Copy, Debug)]
pub struct RoomConfig {
    pub width: f64,
    pub height: f64,
    pub depth: f64,
    pub absorption: f64,      // 0.0 = fully reflective, 1.0 = fully absorptive
    pub air_absorption: f64,  // dB/m at 1 kHz
    pub source_pos: [f64; 3],
    pub listener_pos: [f64; 3],
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self {
            width: 6.0,
            height: 3.0,
            depth: 8.0,
            absorption: 0.2,
            air_absorption: 0.01,
            source_pos: [2.0, 1.5, 3.0],
            listener_pos: [4.0, 1.5, 5.0],
        }
    }
}

/// All-pass delay for diffusion stage.
#[derive(Clone)]
struct DiffusionAllPass {
    delay: DelayLine,
    coeff: f64,
}

impl DiffusionAllPass {
    fn new(max_delay_sec: f64, sample_rate: f64) -> Self {
        let max_len = (max_delay_sec * sample_rate).ceil() as usize;
        Self {
            delay: DelayLine::new(max_len, sample_rate),
            coeff: 0.7,
        }
    }

    #[inline]
    fn process(&mut self, input: f64, delay_sec: f64) -> f64 {
        let x = if input.is_finite() { input } else { 0.0 };
        // All-pass: y[n] = -coeff * x[n] + x[n-d] + coeff * y[n-d]
        let delayed = self.delay.process_sample(x, delay_sec);
        let out = -self.coeff * x + delayed + self.coeff * delayed;
        flush_denormal(out)
    }

    fn reset(&mut self) {
        self.delay.reset();
    }
}

/// Diffusion network: 4 nested all-passes + feedback delay line.
#[derive(Clone)]
struct DiffusionNetwork {
    aps: [DiffusionAllPass; 4],
    fb_delay: DelayLine,
    fb_gain: f64,
}

impl DiffusionNetwork {
    fn new(sample_rate: f64) -> Self {
        let max_delay = 0.05 * sample_rate; // 50 ms max
        let max_len = max_delay.ceil() as usize;
        Self {
            aps: [
                DiffusionAllPass::new(0.01, sample_rate),
                DiffusionAllPass::new(0.012, sample_rate),
                DiffusionAllPass::new(0.015, sample_rate),
                DiffusionAllPass::new(0.018, sample_rate),
            ],
            fb_delay: DelayLine::new(max_len, sample_rate),
            fb_gain: 0.3,
        }
    }

    fn configure(&mut self, sample_rate: f64) {
        for ap in &mut self.aps {
            ap.delay.set_sample_rate(sample_rate);
        }
        self.fb_delay.set_sample_rate(sample_rate);
    }

    #[inline]
    fn process(&mut self, input: f64) -> f64 {
        let mut x = if input.is_finite() { input } else { 0.0 };
        // Nested all-passes
        x = self.aps[0].process(x, 0.002);
        x = self.aps[1].process(x, 0.003);
        x = self.aps[2].process(x, 0.004);
        x = self.aps[3].process(x, 0.005);

        // Feedback delay
        let fb = self.fb_delay.process_sample(x, 0.05);
        let out = x + self.fb_gain * fb;
        flush_denormal(out)
    }

    fn reset(&mut self) {
        for ap in &mut self.aps {
            ap.reset();
        }
        self.fb_delay.reset();
    }
}

/// Acoustic room engine: early reflections (image sources) + late diffusion.
#[derive(Clone)]
pub struct AcousticRoom {
    config: RoomConfig,
    images_l: [ImageSource; 30],
    images_r: [ImageSource; 30],
    image_count: usize,
    /// Per-image delay lines for true early-reflection taps (pre-allocated).
    early_delay_lines: Vec<DelayLine>,
    diffusion_l: DiffusionNetwork,
    diffusion_r: DiffusionNetwork,
    wet_gain: f64,
    dry_gain: f64,
    sample_rate: f64,
}

impl Default for AcousticRoom {
    fn default() -> Self {
        Self {
            config: RoomConfig::default(),
            images_l: [ImageSource { delay: 0.0, gain: 0.0 }; 30],
            images_r: [ImageSource { delay: 0.0, gain: 0.0 }; 30],
            image_count: 0,
            early_delay_lines: Vec::new(),
            diffusion_l: DiffusionNetwork::new(48000.0),
            diffusion_r: DiffusionNetwork::new(48000.0),
            wet_gain: 0.5,
            dry_gain: 1.0,
            sample_rate: 48000.0,
        }
    }
}

impl AcousticRoom {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize with sample rate and room config. Pre-allocates all buffers.
    pub fn initialize(&mut self, sample_rate: f64, config: RoomConfig) -> bool {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48000.0 };
        self.sample_rate = sr;
        self.config = config;

        // Build image sources up to 2nd order (max 30 per channel).
        self.build_image_sources();

        // Pre-allocate one delay line per image source for true early-reflection taps.
        self.early_delay_lines = Vec::with_capacity(self.image_count);
        for i in 0..self.image_count {
            let delay_len = ((self.images_l[i].delay * sr).ceil() as usize).max(1);
            self.early_delay_lines.push(DelayLine::new(delay_len, sr));
        }

        // Configure diffusion networks.
        self.diffusion_l.configure(sr);
        self.diffusion_r.configure(sr);

        self.reset();
        true
    }

    /// Build image sources using mirror-source method up to 2nd order.
    fn build_image_sources(&mut self) {
        let c = self.config;
        let w = c.width;
        let h = c.height;
        let d = c.depth;
        let src = c.source_pos;
        let lst = c.listener_pos;
        let c_sound = 343.0; // m/s

        // Wall planes: x=0, x=w, y=0, y=h, z=0, z=d
        // 1st order images (6 walls)
        let mirrors_1st = [
            [-src[0], src[1], src[2]],       // x=0
            [2.0 * w - src[0], src[1], src[2]], // x=w
            [src[0], -src[1], src[2]],       // y=0
            [src[0], 2.0 * h - src[1], src[2]], // y=h
            [src[0], src[1], -src[2]],       // z=0
            [src[0], src[1], 2.0 * d - src[2]], // z=d
        ];

        let mut idx = 0;
        // 1st order
        for mir in &mirrors_1st {
            if idx >= 30 { break; }
            let dist = distance(*mir, lst);
            let delay = dist / c_sound;
            let gain = (1.0 - c.absorption).powi(1) / dist.max(0.1);
            self.images_l[idx] = ImageSource { delay, gain };
            self.images_r[idx] = ImageSource { delay, gain };
            idx += 1;
        }

        // 2nd order: mirror each 1st-order image across all planes again.
        // 1st order images as base (hardcoded 6 base images)
        let first_order: [[f64; 3]; 6] = [
            [-src[0], src[1], src[2]],       // x=0 mirror
            [2.0 * w - src[0], src[1], src[2]], // x=w mirror
            [src[0], -src[1], src[2]],       // y=0 mirror
            [src[0], 2.0 * h - src[1], src[2]], // y=h mirror
            [src[0], src[1], -src[2]],       // z=0 mirror
            [src[0], src[1], 2.0 * d - src[2]], // z=d mirror
        ];

                // 2nd order: mirror each 1st-order image across all planes again.
        // Planes: x=0, x=w, y=0, y=h, z=0, z=d
        let planes_2nd = [
            (0, 0.0), (0, w),
            (1, 0.0), (1, h),
            (2, 0.0), (2, d),
        ];
        for base in first_order {
            for (axis, pos) in planes_2nd {
                if idx >= 30 { break; }
                let mut m: [f64; 3] = base;
                m[axis] = 2.0 * pos - m[axis];
                let dist = distance(m, lst);
                let delay = dist / c_sound;
                let gain = (1.0 - c.absorption).powi(2) / dist.max(0.1);
                self.images_l[idx] = ImageSource { delay, gain };
                self.images_r[idx] = ImageSource { delay, gain };
                idx += 1;
            }
            if idx >= 30 { break; }
        }

        self.image_count = idx;

        // Apply air absorption to each image using its own propagation distance.
        for i in 0..self.image_count {
            let img_distance = self.images_l[i].delay * c_sound;
            let air_db = c.air_absorption * img_distance;
            let air_lin = 10.0_f64.powf(-air_db / 20.0);
            self.images_l[i].gain *= air_lin;
            self.images_r[i].gain *= air_lin;
        }
    }

    pub fn reset(&mut self) {
        for dl in &mut self.early_delay_lines {
            dl.reset();
        }
        self.diffusion_l.reset();
        self.diffusion_r.reset();
    }

    pub fn set_wet_gain(&mut self, gain: f64) {
        self.wet_gain = if gain.is_finite() { gain.clamp(0.0, 1.0) } else { 0.5 };
    }

    pub fn set_dry_gain(&mut self, gain: f64) {
        self.dry_gain = if gain.is_finite() { gain.clamp(0.0, 2.0) } else { 1.0 };
    }

    /// Process mono input -> stereo output (early + late).
    #[inline]
    pub fn process_mono(&mut self, input: f64) -> (f64, f64) {
        let x = if input.is_finite() { input } else { 0.0 };
        if x == 0.0 {
            return (0.0, 0.0);
        }

        // Early reflections: true per-image delayed taps.
        let mut early_l = 0.0;
        let mut early_r = 0.0;
        for i in 0..self.image_count {
            let img_l = self.images_l[i];
            let img_r = self.images_r[i];
            let tap_l = self.early_delay_lines[i].process_sample(x, img_l.delay);
            let tap_r = self.early_delay_lines[i].process_sample(x, img_r.delay);
            early_l += img_l.gain * tap_l;
            early_r += img_r.gain * tap_r;
        }

        // Late diffusion
        let late_l = self.diffusion_l.process(early_l * 0.1);
        let late_r = self.diffusion_r.process(early_r * 0.1);

        let out_l = self.dry_gain * x + self.wet_gain * late_l;
        let out_r = self.dry_gain * x + self.wet_gain * late_r;

        (flush_denormal(out_l), flush_denormal(out_r))
    }

    /// Process stereo input -> stereo output.
    #[inline]
    pub fn process_stereo(&mut self, left: f64, right: f64) -> (f64, f64) {
        let l = if left.is_finite() { left } else { 0.0 };
        let r = if right.is_finite() { right } else { 0.0 };

        if l == 0.0 && r == 0.0 {
            return (0.0, 0.0);
        }

        // Early reflections: per-image delayed taps for each channel.
        let mut early_l = 0.0;
        let mut early_r = 0.0;
        for i in 0..self.image_count {
            let img_l = self.images_l[i];
            let img_r = self.images_r[i];
            let tap_l = self.early_delay_lines[i].process_sample(l, img_l.delay);
            let tap_r = self.early_delay_lines[i].process_sample(r, img_r.delay);
            early_l += img_l.gain * tap_l;
            early_r += img_r.gain * tap_r;
        }

        // Late diffusion
        let late_l = self.diffusion_l.process(early_l * 0.1);
        let late_r = self.diffusion_r.process(early_r * 0.1);

        let out_l = self.dry_gain * l + self.wet_gain * late_l;
        let out_r = self.dry_gain * r + self.wet_gain * late_r;

        (flush_denormal(out_l), flush_denormal(out_r))
    }
}

/// Euclidean distance between two 3D points.
#[inline]
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acoustic_room_initialize_and_process_finite() {
        let mut room = AcousticRoom::new();
        let config = RoomConfig::default();
        assert!(room.initialize(48000.0, config));
        assert!(room.image_count > 0 && room.image_count <= 30);

        for size in [1usize, 7, 64, 513, 2048] {
            for i in 0..size {
                let x = if i == 0 { f64::NAN } else { (i as f64 * 0.01).sin() };
                let (l, r) = room.process_mono(x);
                assert!(l.is_finite() && r.is_finite(), "non-finite at size {size} idx {i}");
            }
        }
    }

    #[test]
    fn acoustic_room_silence_at_rest() {
        let mut room = AcousticRoom::new();
        room.initialize(48000.0, RoomConfig::default());
        let (l, r) = room.process_mono(0.0);
        assert_eq!(l, 0.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn acoustic_room_stereo_process_finite() {
        let mut room = AcousticRoom::new();
        room.initialize(48000.0, RoomConfig::default());
        for size in [1usize, 7, 64, 513, 2048] {
            for i in 0..size {
                let x = if i == 0 { f64::NAN } else { (i as f64 * 0.01).sin() };
                let y = if i == 0 { f64::INFINITY } else { (i as f64 * 0.02).cos() };
                let (l, r) = room.process_stereo(x, y);
                assert!(l.is_finite() && r.is_finite(), "non-finite at size {size} idx {i}");
            }
        }
    }

    #[test]
    fn image_source_count_and_gains() {
        let mut room = AcousticRoom::new();
        room.initialize(48000.0, RoomConfig::default());
        // Should have 1st + 2nd order = up to 30
        assert!(room.image_count >= 6 && room.image_count <= 30);
        for i in 0..room.image_count {
            assert!(room.images_l[i].gain.is_finite() && room.images_l[i].gain >= 0.0);
            assert!(room.images_l[i].delay.is_finite() && room.images_l[i].delay >= 0.0);
        }
    }
}
