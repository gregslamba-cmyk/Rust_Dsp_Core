# TODO — Rust_Dsp_Core (`core-dsp` — Collegato a `ROADMAP.md`)
Copyright (c) 2026 Alberto Gregorio Ferraris. All rights reserved.

- [ ] **v0.9:** Bonifica P0/P1 Parte B (`PitchShifter` COLA zero-alloc, `FdnReverb` Hadamard $1/\sqrt{8}$, `AcousticRoom`, `Spatial3d` ITD, `LoudnessEngine` ITU-1770-4, `WeightingFilter` IEC/ITU-468, `CompiledAudioGraph`, `Oversampler` polifase)
- [ ] **v0.9:** Re-export completo dei 46 moduli in `core-dsp/src/lib.rs` + Wavetable band-limited, YIN/pYIN Pitch Tracker e `AutoPhaseAligner`
- [ ] **v2.0:** Ottimizzazione SIMD (`f64x4`) ed offload GPU opzionale (`wgpu` / Vulkan Compute)
