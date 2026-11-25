//! # Disk Streamer
//!
//! Lock-free SPSC ring-buffer for multi-track playback/recording.
//! Pre-allocated, zero allocations in process().

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct DiskStreamer {
    buffer: Vec<f64>,
    capacity: usize,
    read_pos: AtomicUsize,
    write_pos: AtomicUsize,
}

impl DiskStreamer {
    pub fn new(capacity: usize) -> Self {
        let cap = if capacity > 0 { capacity } else { 1 };
        Self {
            buffer: vec![0.0; cap],
            capacity: cap,
            read_pos: AtomicUsize::new(0),
            write_pos: AtomicUsize::new(0),
        }
    }

    pub fn initialize(&mut self) -> bool {
        true
    }

    pub fn reset(&mut self) {
        self.read_pos.store(0, Ordering::Relaxed);
        self.write_pos.store(0, Ordering::Relaxed);
    }

    pub fn push(&mut self, value: f64) -> bool {
        let write = self.write_pos.load(Ordering::Relaxed);
        let read = self.read_pos.load(Ordering::Relaxed);
        if self.write_pos.load(Ordering::Relaxed) - self.read_pos.load(Ordering::Relaxed) >= self.capacity - 1 {
            return false;
        }
        let idx = write % self.capacity;
        self.buffer[idx] = if value.is_finite() { value } else { 0.0 };
        self.write_pos.store(write + 1, Ordering::Relaxed);
        true
    }

    pub fn pop(&mut self) -> Option<f64> {
        let read = self.read_pos.load(Ordering::Relaxed);
        let write = self.write_pos.load(Ordering::Relaxed);
        if read == write {
            return None;
        }
        let idx = read % self.capacity;
        let val = self.buffer[idx];
        self.read_pos.store(read + 1, Ordering::Relaxed);
        Some(val)
    }
}

impl Default for DiskStreamer {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop() {
        let mut d = DiskStreamer::new(8);
        assert!(d.push(1.0));
        assert!(d.push(2.0));
        assert_eq!(d.pop(), Some(1.0));
        assert_eq!(d.pop(), Some(2.0));
        assert_eq!(d.pop(), None);
    }

    #[test]
    fn full_buffer() {
        let mut d = DiskStreamer::new(3);
        assert!(d.push(1.0));
        assert!(d.push(2.0));
        assert!(!d.push(3.0)); // full
        assert_eq!(d.pop(), Some(1.0));
        assert!(d.push(3.0)); // now has space
        assert_eq!(d.pop(), Some(2.0));
        assert_eq!(d.pop(), Some(3.0));
        assert_eq!(d.pop(), None);
    }
}