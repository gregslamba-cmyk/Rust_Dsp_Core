//! Partitioned Convolution (FIR) engine using realfft or rubato.
//! Currently a minimal stub; later integrate oversampling and partitioned convolution.

/// Kernel partition size (samples per partition).
const DEFAULT_PARTITION_SIZE: usize = 512;

/// Partitioned convolution state placeholder.
pub struct PartitionedConvolver {
    /// Impulse response length (total).
    _ir_len: usize,
    /// Current partition index.
    part_idx: usize,
    /// Pre‑computed FFTs (future implementation).
    _ffts: Vec<f64>,
}

impl PartitionedConvolver {
    /// Creates a new PartitionedConvolver with a given impulse response length.
    pub fn new(ir_len: usize) -> Self {
        let parts = (ir_len + DEFAULT_PARTITION_SIZE - 1) / DEFAULT_PARTITION_SIZE;
        Self {
            _ir_len: ir_len,
            part_idx: 0,
            _ffts: vec![0.0; parts],
        }
    }

    /// Resets the internal buffers.
    pub fn reset(&mut self) {
        self.part_idx = 0;
        self._ffts.fill(0.0);
    }

    /// Processes a block of samples using the current partition.
    /// Returns the filtered output.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        // Silence fast-path: if input is silent, output zeros.
        let has_signal = input.iter().any(|&x| x.abs() > 1.0e-20);
        if !has_signal {
            for out in output.iter_mut() {
                *out = 0.0;
            }
            return;
        }
        let n = input.len().min(output.len());
        for i in 0..n {
            if let (Some(in_s), Some(out_s)) = (input.get(i), output.get_mut(i)) {
                *out_s = if in_s.is_finite() { *in_s } else { 0.0 };
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitioned_convolver_handles_irregular_sizes_and_stays_finite() {
        let mut convolver = PartitionedConvolver::new(1_024);
        convolver.reset();
        for size in [1usize, 7, 64, 513, 2048] {
            let input: Vec<f64> = (0..size)
                .map(|i| {
                    if i == 0 {
                        f64::NAN
                    } else {
                        (i as f64 * 0.001).cos()
                    }
                })
                .collect();
            let mut output = vec![f64::NAN; size];
            convolver.process_block(&input, &mut output);
            for sample in output.iter() {
                assert!(sample.is_finite(), "non-finite sample in size {size}");
            }
        }
    }
}
