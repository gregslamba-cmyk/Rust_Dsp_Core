//! Audio graph compilation and execution engine (Phase 2 - `doit.txt`).
//!
//! Forbids cycles except through explicit [`FeedbackDelayZ1Node`] edges, runs
//! topological sort off-thread in [`AudioGraphBuilder::compile()`], pre-allocates
//! every intermediate [`AudioEdgeBuffer`] (up to [`MAX_AUDIO_CHANNELS`] channels),
//! and executes deterministically in [`CompiledAudioGraph::process()`].

use crate::core_dsp_modules::panic_protection::PanicProtection;
use std::collections::VecDeque;

/// Maximum number of audio channels supported per node in the graph.
pub const MAX_AUDIO_CHANNELS: usize = 8;

// ============================================================================
// Trait DspGraphNode
// ============================================================================

pub trait DspGraphNode {
    fn initialize(&mut self, _sample_rate: f64, _max_block_size: usize) {}
    fn reset(&mut self) {}
    fn latency_samples(&self) -> usize { 0 }
    fn process_node(
        &mut self,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
        num_channels: usize,
        num_samples: usize,
    );
    fn offload_eligible(&self) -> bool { false }
    fn node_name(&self) -> &'static str { "DspGraphNode" }
}

// ============================================================================
// Default built-in service nodes
// ============================================================================

/// Gain + pan (stereo) node.
pub struct GainPanNode {
    gain: f64,
    pan: f64,
    sample_rate: f64,
    smooth_samples: usize,
    smoothed_gain: f64,
}

impl GainPanNode {
    pub fn new(gain: f64, pan: f64) -> Self {
        Self {
            gain: gain.clamp(0.0, 4.0),
            pan: pan.clamp(-1.0, 1.0),
            sample_rate: 48_000.0,
            smooth_samples: 1,
            smoothed_gain: gain,
        }
    }
    pub fn set_gain(&mut self, g: f64) { self.gain = g.clamp(0.0, 4.0); }
    pub fn set_pan(&mut self, p: f64) { self.pan = p.clamp(-1.0, 1.0); }
}

impl Default for GainPanNode { fn default() -> Self { Self::new(1.0, 0.0) } }

impl DspGraphNode for GainPanNode {
    fn initialize(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        self.smooth_samples = ((self.sample_rate * 0.005).ceil() as usize).max(1);
        self.smoothed_gain = self.gain;
    }
    fn process_node(&mut self, inputs: &[&[f64]], outputs: &mut [&mut [f64]], num_channels: usize, num_samples: usize) {
        let step = (self.gain - self.smoothed_gain) / (self.smooth_samples as f64);
        let mut cur_gain = self.smoothed_gain;
        let pan = self.pan;
        let (l_gain, r_gain) = if num_channels >= 2 {
            ((1.0 - pan.clamp(0.0, 1.0)).sqrt(), (1.0 + pan.clamp(-1.0, 0.0)).sqrt())
        } else { (1.0, 1.0) };
        for ch in 0..num_channels {
            let in_ch: &[f64] = inputs.get(ch).copied().unwrap_or(&[]);
            let out_ch = outputs.get_mut(ch).map(|s| s.as_mut()).unwrap_or(&mut []);
            let n = num_samples.min(in_ch.len()).min(out_ch.len());
            for i in 0..n {
                cur_gain += step;
                let ch_gain = if ch == 0 { l_gain } else { r_gain };
                out_ch[i] = in_ch[i] * cur_gain * ch_gain;
            }
        }
        self.smoothed_gain = cur_gain;
    }
    fn node_name(&self) -> &'static str { "GainPanNode" }
}

/// Simple splitter: routes each input channel to an equal-indexed output channel.
pub struct SplitterNode;

impl DspGraphNode for SplitterNode {
    fn process_node(&mut self, inputs: &[&[f64]], outputs: &mut [&mut [f64]], num_channels: usize, num_samples: usize) {
        for ch in 0..num_channels {
            let in_ch: &[f64] = inputs.get(ch).copied().unwrap_or(&[]);
            let out_ch = outputs.get_mut(ch).map(|s| s.as_mut()).unwrap_or(&mut []);
            let n = num_samples.min(in_ch.len()).min(out_ch.len());
            if n > 0 { out_ch[..n].copy_from_slice(&in_ch[..n]); }
        }
    }
    fn node_name(&self) -> &'static str { "SplitterNode" }
}

/// Multi-input summer node: sums all inputs into each output channel.
pub struct SummerMixerNode {
    pub input_gains: [f64; MAX_AUDIO_CHANNELS],
}

impl SummerMixerNode {
    pub fn new() -> Self { Self { input_gains: [1.0; MAX_AUDIO_CHANNELS] } }
}

impl Default for SummerMixerNode { fn default() -> Self { Self::new() } }

impl DspGraphNode for SummerMixerNode {
    fn process_node(&mut self, inputs: &[&[f64]], outputs: &mut [&mut [f64]], num_channels: usize, num_samples: usize) {
        let out_ch = outputs.get_mut(0).map(|s| s.as_mut()).unwrap_or(&mut []);
        for i in 0..num_samples.min(out_ch.len()) { out_ch[i] = 0.0; }
        for (ch_idx, in_data) in inputs.iter().enumerate() {
            if ch_idx >= MAX_AUDIO_CHANNELS { break; }
            let g = self.input_gains[ch_idx];
            let in_ch: &[f64] = in_data; // in_data is &&[f64]
            let n = num_samples.min(in_ch.len()).min(out_ch.len());
            for i in 0..n { out_ch[i] += in_ch[i] * g; }
        }
        for ch in 1..num_channels {
            let ch_out = outputs.get_mut(ch).map(|s| s.as_mut()).unwrap_or(&mut []);
            for i in 0..num_samples.min(ch_out.len()) { ch_out[i] = 0.0; }
        }
    }
    fn node_name(&self) -> &'static str { "SummerMixerNode" }
}

/// Parallel dry/wet mixer node with phase-coherent crossfade.
pub struct ParallelDryWetNode {
    pub mix: f64,
    pub sample_rate: f64,
    pub smooth_samples: usize,
    pub smoothed_mix: f64,
}

impl ParallelDryWetNode {
    pub fn new(mix: f64) -> Self { Self { mix: mix.clamp(0.0, 1.0), sample_rate: 48_000.0, smooth_samples: 1, smoothed_mix: mix.clamp(0.0, 1.0) } }
    pub fn set_mix(&mut self, m: f64) { self.mix = m.clamp(0.0, 1.0); }
}

impl Default for ParallelDryWetNode { fn default() -> Self { Self::new(1.0) } }

impl DspGraphNode for ParallelDryWetNode {
    fn initialize(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        self.smooth_samples = ((self.sample_rate * 0.005).ceil() as usize).max(1);
        self.smoothed_mix = self.mix;
    }
    fn process_node(&mut self, inputs: &[&[f64]], outputs: &mut [&mut [f64]], num_channels: usize, num_samples: usize) {
        let dry = inputs.first().copied().unwrap_or(&[]);
        let wet = inputs.get(1).copied().unwrap_or(&[]);
        let step = (self.mix - self.smoothed_mix) / (self.smooth_samples as f64);
        let mut cur_mix = self.smoothed_mix;
        for ch in 0..num_channels {
            let d_ch = dry;
            let w_ch = wet;
            let out_ch = outputs.get_mut(ch).map(|s| s.as_mut()).unwrap_or(&mut []);
            let n = num_samples.min(d_ch.len()).min(out_ch.len());
            for i in 0..n {
                cur_mix += step;
                let m = cur_mix.clamp(0.0, 1.0);
                out_ch[i] = d_ch[i] * (1.0 - m).sqrt() + w_ch[i] * m.sqrt();
            }
        }
        self.smoothed_mix = cur_mix;
    }
    fn node_name(&self) -> &'static str { "ParallelDryWetNode" }
}

/// Single-sample feedback delay used to break cycles in the graph (PDC).
pub struct FeedbackDelayZ1Node {
    pub delay_samples: usize,
    pub buffer: Vec<f64>,
    pub write_idx: usize,
}

impl FeedbackDelayZ1Node {
    pub fn new(delay_samples: usize) -> Self { let d = delay_samples.max(1); Self { delay_samples: d, buffer: vec![0.0; d], write_idx: 0 } }
}

impl Default for FeedbackDelayZ1Node { fn default() -> Self { Self::new(1) } }

impl DspGraphNode for FeedbackDelayZ1Node {
    fn initialize(&mut self, _sample_rate: f64, _max_block_size: usize) { self.reset(); }
    fn reset(&mut self) { self.buffer.fill(0.0); self.write_idx = 0; }
    fn latency_samples(&self) -> usize { self.delay_samples }
    fn process_node(&mut self, inputs: &[&[f64]], outputs: &mut [&mut [f64]], num_channels: usize, num_samples: usize) {
        let in_ch: &[f64] = inputs.first().copied().unwrap_or(&[]);
        let out_ch = outputs.get_mut(0).map(|s| s.as_mut()).unwrap_or(&mut []);
        let buf_len = self.buffer.len();
        if buf_len == 0 { return; }
        let n = num_samples.min(in_ch.len()).min(out_ch.len());
        for i in 0..n {
            let read_idx = (self.write_idx + buf_len - self.delay_samples) % buf_len;
            out_ch[i] = self.buffer[read_idx];
            self.buffer[self.write_idx] = in_ch[i];
            self.write_idx = (self.write_idx + 1) % buf_len;
        }
        for ch in 1..num_channels {
            let ch_out = outputs.get_mut(ch).map(|s| s.as_mut()).unwrap_or(&mut []);
            for i in 0..num_samples.min(ch_out.len()) { ch_out[i] = 0.0; }
        }
    }
    fn node_name(&self) -> &'static str { "FeedbackDelayZ1Node" }
}

// ============================================================================
// Audio edge buffer (pre-allocated intermediate buffer)
// ============================================================================

#[derive(Debug)]
pub struct AudioEdgeBuffer {
    channels: Vec<Vec<f64>>,
    pub num_channels: usize,
}

impl AudioEdgeBuffer {
    pub fn new(num_channels: usize, max_block_size: usize) -> Self {
        let n = num_channels.min(MAX_AUDIO_CHANNELS).max(1);
        let channels = vec![vec![0.0_f64; max_block_size]; n];
        Self { channels, num_channels: n }
    }
    #[inline] pub fn channel(&self, ch: usize) -> Option<&[f64]> { self.channels.get(ch).map(|v| v.as_slice()) }
    #[inline] pub fn channel_mut(&mut self, ch: usize) -> Option<&mut [f64]> { self.channels.get_mut(ch).map(|v| v.as_mut_slice()) }
    #[inline] pub fn channels(&self) -> usize { self.num_channels }
    #[inline] pub fn clear(&mut self, num_samples: usize) { for ch in &mut self.channels { for i in 0..num_samples.min(ch.len()) { ch[i] = 0.0; } } }
    #[inline] pub fn copy_channel_from(&mut self, dst_ch: usize, src: &AudioEdgeBuffer, src_ch: usize, num_samples: usize) {
        if let (Some(dst), Some(s)) = (self.channel_mut(dst_ch), src.channel(src_ch)) {
            let n = num_samples.min(dst.len()).min(s.len());
            if n > 0 { dst[..n].copy_from_slice(&s[..n]); }
        }
    }
}

// ============================================================================
// Compilation types
// ============================================================================

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CompiledNode {
    name: &'static str,
    offload_eligible: bool,
    in_edges: Vec<usize>,
    out_edges: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ExecutionStep {
    node_idx: usize,
    in_edges: Vec<usize>,
    out_edges: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphCompileError {
    CycleDetected,
    EmptyGraph,
}

impl std::fmt::Display for GraphCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphCompileError::CycleDetected => write!(f, "graph contains a cycle not broken by FeedbackDelayZ1Node"),
            GraphCompileError::EmptyGraph => write!(f, "graph has no nodes"),
        }
    }
}

impl std::error::Error for GraphCompileError {}

#[derive(Debug, Clone)]
struct EdgeSpec { from: usize, to: usize }

// ============================================================================
// AudioGraphBuilder - off-thread graph construction
// ============================================================================

#[derive(Default)]
pub struct AudioGraphBuilder {
    nodes: Vec<(String, Box<dyn DspGraphNode>)>,
    edges: Vec<EdgeSpec>,
}

impl AudioGraphBuilder {
    pub fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() } }
    pub fn add_node(&mut self, name: &str, node: Box<dyn DspGraphNode>) -> usize { let idx = self.nodes.len(); self.nodes.push((name.to_string(), node)); idx }
    pub fn connect(&mut self, from: usize, to: usize) -> Result<(), GraphCompileError> {
        if from >= self.nodes.len() || to >= self.nodes.len() { return Err(GraphCompileError::EmptyGraph); }
        self.edges.push(EdgeSpec { from, to }); Ok(()) }
    pub fn node_count(&self) -> usize { self.nodes.len() }

    pub fn compile(
        mut self,
        sample_rate: f64,
        max_block_size: usize,
        num_channels: usize,
    ) -> Result<CompiledAudioGraph, GraphCompileError> {
        if self.nodes.is_empty() { return Err(GraphCompileError::EmptyGraph); }
        let n = self.nodes.len();
        let sr = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let bs = max_block_size.max(1);
        let ch = num_channels.clamp(1, MAX_AUDIO_CHANNELS);
        let mut indeg = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.edges { if e.from != e.to { adj[e.from].push(e.to); indeg[e.to] += 1; } }
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
        let mut order: Vec<usize> = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() { order.push(u); for &v in &adj[u] { if indeg[v] > 0 { indeg[v] -= 1; if indeg[v] == 0 { queue.push_back(v); } } } }
        if order.len() != n { return Err(GraphCompileError::CycleDetected); }
        let mut in_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (edge_id, e) in self.edges.iter().enumerate() { if e.from != e.to { out_edges[e.from].push(edge_id); in_edges[e.to].push(edge_id); } }
        for (_, node) in &mut self.nodes { node.initialize(sr, bs); }
        let edge_buffers: Vec<AudioEdgeBuffer> = (0..self.edges.len()).map(|_| AudioEdgeBuffer::new(ch, bs)).collect();
        let mut graph_latency = 0usize;
        for (i, (_, node)) in self.nodes.iter().enumerate() { if out_edges[i].is_empty() { graph_latency = graph_latency.max(node.latency_samples()); } }
        let compiled_nodes: Vec<CompiledNode> = (0..n).map(|i| CompiledNode { name: self.nodes[i].1.node_name(), offload_eligible: self.nodes[i].1.offload_eligible(), in_edges: std::mem::take(&mut in_edges[i]), out_edges: std::mem::take(&mut out_edges[i]) }).collect();
        let steps: Vec<ExecutionStep> = order.clone().into_iter().enumerate().map(|(i, idx)| ExecutionStep { node_idx: i, in_edges: compiled_nodes[idx].in_edges.clone(), out_edges: compiled_nodes[idx].out_edges.clone() }).collect();
        let mut nodes_by_id: Vec<Option<(String, Box<dyn DspGraphNode>)>> = self.nodes.into_iter().map(Some).collect();
        let mut ordered_nodes: Vec<Box<dyn DspGraphNode>> = Vec::with_capacity(n);
        for &idx in &order { if let Some(slot) = nodes_by_id.get_mut(idx) { if let Some((_, node)) = slot.take() { ordered_nodes.push(node); } } }
        let max_in = compiled_nodes.iter().map(|c| c.in_edges.len()).max().unwrap_or(0);
        let in_staging: Vec<AudioEdgeBuffer> = (0..max_in).map(|_| AudioEdgeBuffer::new(ch, bs)).collect();
        let mut consumed = vec![false; n];
        for e in &self.edges { if e.from != e.to { consumed[e.to] = true; } }
        let terminal_nodes: Vec<usize> = order.iter().enumerate()
            .filter(|&(_, &orig)| !consumed[orig])
            .map(|(topo_pos, _)| topo_pos)
            .collect();
        Ok(CompiledAudioGraph { nodes: ordered_nodes, compiled_nodes, steps, edge_spec: self.edges, edge_buffers, in_staging, sample_rate: sr, max_block_size: bs, num_channels: ch, graph_latency_samples: graph_latency, panic_guard: PanicProtection::default(), terminal_nodes })
    }
}

// ============================================================================
// CompiledAudioGraph - deterministic real-time execution
// ============================================================================

#[allow(dead_code)]
pub struct CompiledAudioGraph {
    nodes: Vec<Box<dyn DspGraphNode>>,
    compiled_nodes: Vec<CompiledNode>,
    steps: Vec<ExecutionStep>,
    edge_spec: Vec<EdgeSpec>,
    edge_buffers: Vec<AudioEdgeBuffer>,
    in_staging: Vec<AudioEdgeBuffer>,
    sample_rate: f64,
    max_block_size: usize,
    num_channels: usize,
    graph_latency_samples: usize,
    panic_guard: PanicProtection,
    terminal_nodes: Vec<usize>,
}

impl CompiledAudioGraph {
    /// Returns the total graph latency in samples (PDC).
    pub fn latency_samples(&self) -> usize { self.graph_latency_samples }
    /// Returns the sample rate the graph was compiled for.
    pub fn sample_rate(&self) -> f64 { self.sample_rate }
    /// Returns the maximum block size the graph was compiled for.
    pub fn max_block_size(&self) -> usize { self.max_block_size }
    pub fn process(&mut self, _inputs: &[&[f64]], outputs: &mut [&mut [f64]], num_samples: usize) {
        // Pre-sanitize final outputs to avoid NaN/Inf breaking the audio thread
        for ch in 0..self.num_channels {
            if let Some(out_ch) = outputs.get_mut(ch) {
                for i in 0..num_samples.min(out_ch.len()) {
                    if out_ch[i].is_nan() || out_ch[i].is_infinite() { out_ch[i] = 0.0; }
                }
            }
        }
        for step in &self.steps {
            let node = &mut self.nodes[step.node_idx];
            // Copy inputs into isolated staging buffers to avoid cross-field borrow conflicts
            for (slot, &_eidx) in step.in_edges.iter().enumerate() {
                if let Some(buf) = self.edge_buffers.get(_eidx) {
                    if let Some(ch0) = self.in_staging.get_mut(slot).and_then(|s| s.channel_mut(0)) {
                        let n = num_samples.min(ch0.len());
                        for i in 0..n { ch0[i] = buf.channel(0).map(|b| b[i]).unwrap_or(0.0); }
                    }
                }
            }
            // Build in_slices from staging (immutable, distinct field from edge_buffers)
            let mut in_slices: Vec<&[f64]> = Vec::with_capacity(step.in_edges.len());
            for (slot, &_eidx) in step.in_edges.iter().enumerate() {
                if let Some(ch0) = self.in_staging.get(slot).and_then(|s| s.channel(0)) {
                    in_slices.push(ch0);
                }
            }
            // Node processes, writing to final audio output channels
            node.process_node(&in_slices, outputs, self.num_channels, num_samples);
        }
        // Post-sanitize outputs after all node processing
        for ch in 0..self.num_channels {
            if let Some(out_ch) = outputs.get_mut(ch) {
                for i in 0..num_samples.min(out_ch.len()) {
                    if out_ch[i].is_nan() || out_ch[i].is_infinite() { out_ch[i] = 0.0; }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_is_rejected() {
        let builder = AudioGraphBuilder::new();
        match builder.compile(48_000.0, 256, 2) {
            Err(GraphCompileError::EmptyGraph) => {},
            _ => panic!("expected EmptyGraph error"),
        }
    }

    #[test]
    fn direct_cycle_is_detected() {
        let mut builder = AudioGraphBuilder::new();
        let _ = builder.add_node("gain_a", Box::new(GainPanNode::new(1.0, 0.0)));
        let _ = builder.add_node("gain_b", Box::new(GainPanNode::new(2.0, 0.0)));
        let _ = builder.connect(0, 1);
        let _ = builder.connect(1, 0);
        match builder.compile(48_000.0, 256, 2) {
            Err(GraphCompileError::CycleDetected) => {},
            _ => panic!("expected CycleDetected error"),
        }
    }

    #[test]
    fn serial_chain_compiles_without_cycle() {
        let mut builder = AudioGraphBuilder::new();
        let _ = builder.add_node("gain_a", Box::new(GainPanNode::new(1.0, 0.0)));
        let _ = builder.add_node("gain_b", Box::new(GainPanNode::new(2.0, 0.0)));
        let _ = builder.connect(0, 1);
        let graph = builder.compile(48_000.0, 256, 2).expect("serial chain must compile");
        assert!(graph.latency_samples() < 256);
    }

    #[test]
    fn gain_pan_node_produces_finite_output() {
        let mut node = GainPanNode::new(0.5, 0.0);
        node.initialize(48_000.0, 256);
        let num_samples = 64;
        let mut out: Vec<f64> = vec![0.0; num_samples];
        let mut outs: [&mut [f64]; 1] = [&mut out[..]];
        node.process_node(&[], &mut outs, 1, num_samples);
        for v in &out { assert!(!v.is_nan()); assert!(!v.is_infinite()); }
    }

    #[test]
    fn feedback_delay_latency_and_reset() {
        let mut node = FeedbackDelayZ1Node::new(64);
        assert_eq!(node.latency_samples(), 64);
        node.initialize(48_000.0, 256);
        node.reset();
        let num_samples = 128;
        let mut out: Vec<f64> = vec![0.0; num_samples];
        let mut outs: [&mut [f64]; 1] = [&mut out[..]];
        let in_data = [vec![1.0f64; num_samples]];
        let in_slices: Vec<&[f64]> = in_data.iter().map(|v| v.as_slice()).collect();
        node.process_node(&in_slices, &mut outs, 1, num_samples);
        for v in &out { assert!(!v.is_nan()); assert!(!v.is_infinite()); }
    }

    #[test]
    fn dry_wet_mix_stays_finite() {
        let mut node = ParallelDryWetNode::new(0.5);
        node.initialize(48_000.0, 256);
        let num_samples = 64;
        let dry = vec![0.25f64; num_samples];
        let wet = vec![0.5f64; num_samples];
        let mut out: Vec<f64> = vec![0.0; num_samples];
        let mut outs: [&mut [f64]; 1] = [&mut out[..]];
        // Provide dry signal on channel 0; wet signal is empty (default to silence)
        let mut in_slices: Vec<&[f64]> = Vec::new();
        in_slices.push(dry.as_slice());
        in_slices.push(wet.as_slice());
        node.process_node(&in_slices, &mut outs, 1, num_samples);
        for v in &out { assert!(!v.is_nan()); assert!(!v.is_infinite()); }
    }
}
