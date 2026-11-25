//! # Musical Theory Encyclopedia & World Metric Grids
//!
//! This module implements the theoretical music-theory components used by the
//! mini-DAW / synth engines: global world scales, rhythmic metric grids and
//! time signatures with Bar:Beat:Tick coordinates.

pub const WORLD_SCALE_MIN_HZ: f64 = 20.0;
pub const WORLD_SCALE_MAX_HZ: f64 = 20_000.0;
const MAX_SCALE_SEMITONES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldScale {
    pub name: &'static str,
    pub semitones: [i8; MAX_SCALE_SEMITONES],
    pub count: usize,
}

impl WorldScale {
    pub fn is_valid(&self) -> bool {
        self.count > 0 && self.count <= MAX_SCALE_SEMITONES
    }
}

pub struct TimeSignature {
    pub numerator: u8,
    pub denominator: u8,
}

impl Default for TimeSignature {
    fn default() -> Self { Self { numerator: 4, denominator: 4 } }
}

impl TimeSignature {
    pub fn new(numerator: u8, denominator: u8) -> Self {
        let n = if numerator == 0 { 4 } else { numerator };
        let d = if denominator == 0 { 4 } else { denominator };
        Self { numerator: n, denominator: d }
    }
    pub fn beats(&self) -> u8 { self.numerator }
    pub fn beat_ticks(&self, ticks_per_bar: u64) -> u64 { ticks_per_bar / self.numerator as u64 }
}

pub struct MetricGrid {
    pub name: &'static str,
    pub grid: [u8; 8],
    pub count: usize,
}

pub fn world_scales() -> &'static [WorldScale] {
    &[
        WorldScale { name: "Major", semitones: [0, 2, 4, 5, 7, 9, 11, 12], count: 7 },
        WorldScale { name: "Minor", semitones: [0, 2, 3, 5, 7, 8, 10, 12], count: 7 },
        WorldScale { name: "HarmonicMinor", semitones: [0, 2, 3, 5, 7, 8, 11, 12], count: 7 },
        WorldScale { name: "PentatonicMajor", semitones: [0, 2, 4, 7, 9, 12, 0, 0], count: 5 },
        WorldScale { name: "PentatonicMinor", semitones: [0, 3, 5, 7, 10, 12, 0, 0], count: 5 },
        WorldScale { name: "Dorian", semitones: [0, 2, 3, 5, 7, 9, 10, 12], count: 7 },
        WorldScale { name: "Mixolydian", semitones: [0, 2, 4, 5, 7, 9, 11, 12], count: 7 },
        WorldScale { name: "Locrian", semitones: [0, 1, 3, 5, 6, 8, 10, 12], count: 7 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_are_valid() {
        for s in world_scales() {
            assert!(s.is_valid());
        }
    }

    #[test]
    fn time_signature_beats() {
        let ts = TimeSignature::new(4, 4);
        assert_eq!(ts.beats(), 4);
        assert_eq!(ts.beat_ticks(960), 240);
    }
}
