# TODO — Rust_Dsp_Core (`core-dsp` — Collegato a `ROADMAP.md`)
Copyright (c) 2026 Alberto Gregorio Ferraris. All rights reserved.

- [ ] **v0.9:** Bonifica P0/P1 Parte B (`PitchShifter` COLA zero-alloc, `FdnReverb` Hadamard $1/\sqrt{8}$, `AcousticRoom`, `Spatial3d` ITD, `LoudnessEngine` ITU-1770-4, `WeightingFilter` IEC/ITU-468, `CompiledAudioGraph`, `Oversampler` polifase)
- [ ] **v0.9:** Re-export completo dei 46 moduli in `core-dsp/src/lib.rs` + Wavetable band-limited, YIN/pYIN Pitch Tracker e `AutoPhaseAligner`
- [ ] **v2.0:** Ottimizzazione SIMD (`f64x4`) ed offload GPU opzionale (`wgpu` / Vulkan Compute)

- [ ] **Modulo `feedback_matrix.rs` (`MultiResonatorBank`):** fino a 10 bande aggiungibili `[ResonatorBand; 10]` zero-alloc; per banda: **Pitch per Tonalità / Nota Musicale** (`C2..C6` + `Fine Cents`, agganciabile a scale `music_theory.rs`) o `Tempo (0.05–50 ms)` con `Match/Link` tra bande, `Stereo Pan`, **3 Saturazioni** (`Clean f64` / `Warm Triode` / `Iron-Tape`) ed **EQ a 3 bande** (`HPF` + `Mid Bell [Hz + Gain]` + `LPF`); condivisi: `Feedback Volume` `-100%..+100%`, `Gate con isteresi`, **Banco Comune EQ 3 bande** e **LFO unico** (`Sine/Tri/Saw/Square/S&H/Random`, sync BPM/Hz).
