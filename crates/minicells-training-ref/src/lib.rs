#![no_std]

//! The authoritative FP32 Echo training step.
//!
//! This crate intentionally has no host, filesystem, protocol, or allocator
//! dependencies.  Native and guest runners call this same implementation.

#[cfg(test)]
extern crate std;

pub type F32 = f32;
pub const PARAMETER_COUNT: usize = 4476;
pub const VOCAB_SIZE: usize = 44;
pub const NUM_CELLS: usize = 64;
pub const MAX_SEQ_LEN: usize = 32;
pub const EMBEDDING_DIM: usize = 8;
pub const HIDDEN_DIM: usize = 16;
pub const RADIUS: usize = 2;
pub const ITERATIONS: usize = 4;
pub const MLP_WIDTH: usize = 32;
pub const UPDATE_INPUT_DIM: usize = 88;
pub const LOGICAL_BATCH_SIZE: usize = 256;
/// Concurrent-training v1 freezes eight samples per independent leaf.
pub const PARALLEL_SHARD_SIZE: usize = 8;
pub const PARALLEL_LEAF_COUNT: usize = LOGICAL_BATCH_SIZE / PARALLEL_SHARD_SIZE;

pub const EMBEDDING_OFFSET: usize = 0;
pub const UPDATE_IN_WEIGHT_OFFSET: usize = 352;
pub const UPDATE_IN_BIAS_OFFSET: usize = 3168;
pub const UPDATE_OUT_WEIGHT_OFFSET: usize = 3200;
pub const UPDATE_OUT_BIAS_OFFSET: usize = 3712;
pub const OUTPUT_WEIGHT_OFFSET: usize = 3728;
pub const OUTPUT_BIAS_OFFSET: usize = 4432;

#[derive(Clone, Copy)]
pub struct TrainingState {
    pub weights: [F32; PARAMETER_COUNT],
    pub adam_m: [F32; PARAMETER_COUNT],
    pub adam_v: [F32; PARAMETER_COUNT],
    pub step: u64,
}

impl TrainingState {
    pub const fn from_weights(weights: [F32; PARAMETER_COUNT]) -> Self {
        Self {
            weights,
            adam_m: [0.0; PARAMETER_COUNT],
            adam_v: [0.0; PARAMETER_COUNT],
            step: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TrainingBatch {
    pub ids: [[u8; NUM_CELLS]; LOGICAL_BATCH_SIZE],
    pub lengths: [u8; LOGICAL_BATCH_SIZE],
    pub size: u16,
}

/// Reusable scratch storage for one logical training step.  Keeping this
/// workspace outside the call stack is required by the small PVM guest stack;
/// it does not change the order or precision of any arithmetic operation.
pub struct TrainingWorkspace {
    pub gradient: [F32; PARAMETER_COUNT],
    pub states: [[[F32; HIDDEN_DIM]; NUM_CELLS]; ITERATIONS + 1],
    pub hidden_pre: [[[F32; MLP_WIDTH]; NUM_CELLS]; ITERATIONS],
    pub logits: [[F32; VOCAB_SIZE]; MAX_SEQ_LEN],
    pub dstate: [[F32; HIDDEN_DIM]; NUM_CELLS],
    pub dprev: [[F32; HIDDEN_DIM]; NUM_CELLS],
}

/// Exact FP32 accumulation state for one logical optimizer step.  Gradients
/// are accumulated directly in sample order; normalization, clipping and
/// AdamW are applied only by `finalize_adamw_step`.
pub struct GradientAccumulator {
    pub gradient: [F32; PARAMETER_COUNT],
    pub loss_sum: F32,
    pub token_count: u32,
}

/// Raw, unnormalised gradient produced by one independent parallel leaf.
///
/// A leaf is evaluated against the same frozen `TrainingState`; it never
/// clips, normalises, or updates optimizer state.  The explicit metadata is
/// carried through reduction so malformed or mixed leaves can be rejected by
/// the application-level coordinator.
#[derive(Clone, Copy)]
pub struct PartialGradientV1 {
    pub gradient: [F32; PARAMETER_COUNT],
    pub loss_sum: F32,
    pub token_count: u32,
    pub processed_samples: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelJobStatusV1 { LeavesReady, LeavesRunning, RootReady, Finalizing, Complete, Failed }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelJobErrorV1 {
    WrongJob,
    WrongModel,
    WrongOptimizer,
    WrongBatch,
    WrongStep,
    WrongLeaf,
    WrongRange,
    DuplicateLeaf,
    ConflictingLeaf,
    MissingLeaf,
    StaleRoot,
    InvalidStatus,
}

/// Application-level DAG state for one parallel logical batch.  Leaf records
/// are immutable and keyed by leaf index; completion order is intentionally
/// not represented as a sequential `next_shard_index`.
#[derive(Clone, Copy)]
pub struct ParallelTrainingJobV1 {
    pub job_id: [u8; 32],
    pub generation: u64,
    pub optimizer_step: u64,
    pub model_commitment: [u8; 32],
    pub optimizer_commitment: [u8; 32],
    pub batch_commitment: [u8; 32],
    pub logical_batch_size: u16,
    pub shard_size: u16,
    pub leaf_count: u16,
    pub status: ParallelJobStatusV1,
    pub leaf_commitments: [Option<[u8; 32]>; PARALLEL_LEAF_COUNT],
}

impl ParallelTrainingJobV1 {
    pub const fn new(
        job_id: [u8; 32], generation: u64, optimizer_step: u64,
        model_commitment: [u8; 32], optimizer_commitment: [u8; 32], batch_commitment: [u8; 32],
    ) -> Self {
        Self { job_id, generation, optimizer_step, model_commitment, optimizer_commitment,
            batch_commitment, logical_batch_size: LOGICAL_BATCH_SIZE as u16,
            shard_size: PARALLEL_SHARD_SIZE as u16, leaf_count: PARALLEL_LEAF_COUNT as u16,
            status: ParallelJobStatusV1::LeavesReady, leaf_commitments: [None; PARALLEL_LEAF_COUNT] }
    }

    pub fn accept_leaf(
        &mut self, job_id: [u8; 32], generation: u64, optimizer_step: u64,
        model_commitment: [u8; 32], optimizer_commitment: [u8; 32], batch_commitment: [u8; 32],
        leaf_index: u16, sample_start: u16, sample_end: u16, commitment: [u8; 32],
    ) -> Result<(), ParallelJobErrorV1> {
        if !matches!(self.status, ParallelJobStatusV1::LeavesReady | ParallelJobStatusV1::LeavesRunning) { return Err(ParallelJobErrorV1::InvalidStatus); }
        if job_id != self.job_id { return Err(ParallelJobErrorV1::WrongJob); }
        if generation != self.generation || optimizer_step != self.optimizer_step { return Err(ParallelJobErrorV1::WrongStep); }
        if model_commitment != self.model_commitment { return Err(ParallelJobErrorV1::WrongModel); }
        if optimizer_commitment != self.optimizer_commitment { return Err(ParallelJobErrorV1::WrongOptimizer); }
        if batch_commitment != self.batch_commitment { return Err(ParallelJobErrorV1::WrongBatch); }
        let index = leaf_index as usize;
        if index >= PARALLEL_LEAF_COUNT { return Err(ParallelJobErrorV1::WrongLeaf); }
        if sample_start != leaf_index * self.shard_size || sample_end != sample_start + self.shard_size { return Err(ParallelJobErrorV1::WrongRange); }
        if let Some(existing) = self.leaf_commitments[index] {
            return if existing == commitment { Err(ParallelJobErrorV1::DuplicateLeaf) } else { Err(ParallelJobErrorV1::ConflictingLeaf) };
        }
        self.leaf_commitments[index] = Some(commitment);
        self.status = ParallelJobStatusV1::LeavesRunning;
        Ok(())
    }

    pub fn mark_root_ready(&mut self) -> Result<(), ParallelJobErrorV1> {
        if !matches!(self.status, ParallelJobStatusV1::LeavesRunning | ParallelJobStatusV1::LeavesReady) { return Err(ParallelJobErrorV1::InvalidStatus); }
        if self.leaf_commitments.iter().any(|item| item.is_none()) { return Err(ParallelJobErrorV1::MissingLeaf); }
        self.status = ParallelJobStatusV1::RootReady;
        Ok(())
    }

    pub fn begin_finalize(&mut self, current_model_commitment: [u8; 32]) -> Result<(), ParallelJobErrorV1> {
        if self.status != ParallelJobStatusV1::RootReady { return Err(ParallelJobErrorV1::InvalidStatus); }
        if current_model_commitment != self.model_commitment { self.status = ParallelJobStatusV1::Failed; return Err(ParallelJobErrorV1::StaleRoot); }
        self.status = ParallelJobStatusV1::Finalizing;
        Ok(())
    }

    pub fn complete(&mut self) -> Result<(), ParallelJobErrorV1> {
        if self.status != ParallelJobStatusV1::Finalizing { return Err(ParallelJobErrorV1::InvalidStatus); }
        self.status = ParallelJobStatusV1::Complete;
        Ok(())
    }
}

impl PartialGradientV1 {
    pub const fn new() -> Self {
        Self {
            gradient: [0.0; PARAMETER_COUNT],
            loss_sum: 0.0,
            token_count: 0,
            processed_samples: 0,
        }
    }
}

impl Default for PartialGradientV1 {
    fn default() -> Self { Self::new() }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingPhaseV1 { Idle, Accumulating, FinalizeReady }

#[derive(Clone, Copy)]
pub struct GradientAccumulatorStateV1 {
    pub gradient: [F32; PARAMETER_COUNT],
    pub loss_sum: F32,
    pub token_count: u32,
    pub processed_samples: u16,
}

impl GradientAccumulatorStateV1 {
    pub const fn empty() -> Self { Self { gradient: [0.0; PARAMETER_COUNT], loss_sum: 0.0, token_count: 0, processed_samples: 0 } }
}

#[derive(Clone, Copy)]
pub struct TrainingRoundStateV1 {
    pub version: u16,
    pub generation: u64,
    pub optimizer_step: u64,
    pub phase: TrainingPhaseV1,
    pub logical_batch_size: u16,
    pub shard_size: u16,
    pub next_shard_index: u16,
    pub batch_commitment: [u8; 32],
    pub model_commitment: [u8; 32],
    pub optimizer_commitment: [u8; 32],
    pub accumulator: GradientAccumulatorStateV1,
}

impl TrainingRoundStateV1 {
    pub const fn new(generation: u64, optimizer_step: u64, batch_commitment: [u8; 32], model_commitment: [u8; 32], optimizer_commitment: [u8; 32]) -> Self {
        Self { version: 1, generation, optimizer_step, phase: TrainingPhaseV1::Accumulating, logical_batch_size: 256, shard_size: 8, next_shard_index: 0, batch_commitment, model_commitment, optimizer_commitment, accumulator: GradientAccumulatorStateV1::empty() }
    }

    pub fn accept_mca(&mut self, generation: u64, optimizer_step: u64, batch_commitment: [u8; 32], model_commitment: [u8; 32], shard_index: u16, sample_start: u16, sample_end: u16, candidate: GradientAccumulatorStateV1) -> Result<(), RoundError> {
        if self.phase != TrainingPhaseV1::Accumulating { return Err(RoundError::InvalidPhase); }
        if generation != self.generation || optimizer_step != self.optimizer_step { return Err(RoundError::StaleStep); }
        if batch_commitment != self.batch_commitment { return Err(RoundError::WrongBatch); }
        if model_commitment != self.model_commitment { return Err(RoundError::WrongModel); }
        if shard_index != self.next_shard_index || sample_start != shard_index * self.shard_size || sample_end != sample_start + self.shard_size { return Err(RoundError::OutOfOrderShard); }
        if candidate.processed_samples != sample_end { return Err(RoundError::AccumulatorMismatch); }
        self.accumulator = candidate;
        self.next_shard_index += 1;
        if self.next_shard_index == self.logical_batch_size / self.shard_size { self.phase = TrainingPhaseV1::FinalizeReady; }
        Ok(())
    }

    pub fn finalize(&mut self, state: &mut TrainingState) -> Result<TrainStepReport, RoundError> {
        if self.phase != TrainingPhaseV1::FinalizeReady || self.next_shard_index != 32 || self.accumulator.processed_samples != 256 { return Err(RoundError::InvalidPhase); }
        let mut accumulator = GradientAccumulator { gradient: self.accumulator.gradient, loss_sum: self.accumulator.loss_sum, token_count: self.accumulator.token_count };
        let report = finalize_adamw_step(state, &mut accumulator);
        self.accumulator = GradientAccumulatorStateV1::empty(); self.next_shard_index = 0; self.phase = TrainingPhaseV1::Idle;
        Ok(report)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoundError { InvalidPhase, StaleStep, WrongBatch, WrongModel, OutOfOrderShard, AccumulatorMismatch }

impl GradientAccumulator {
    pub const fn new() -> Self {
        Self { gradient: [0.0; PARAMETER_COUNT], loss_sum: 0.0, token_count: 0 }
    }
    pub fn reset(&mut self) {
        self.gradient = [0.0; PARAMETER_COUNT];
        self.loss_sum = 0.0;
        self.token_count = 0;
    }
}

impl Default for GradientAccumulator {
    fn default() -> Self { Self::new() }
}

impl TrainingWorkspace {
    pub const fn new() -> Self {
        Self {
            gradient: [0.0; PARAMETER_COUNT],
            states: [[[0.0; HIDDEN_DIM]; NUM_CELLS]; ITERATIONS + 1],
            hidden_pre: [[[0.0; MLP_WIDTH]; NUM_CELLS]; ITERATIONS],
            logits: [[0.0; VOCAB_SIZE]; MAX_SEQ_LEN],
            dstate: [[0.0; HIDDEN_DIM]; NUM_CELLS],
            dprev: [[0.0; HIDDEN_DIM]; NUM_CELLS],
        }
    }

    fn reset_sample(&mut self) {
        self.states = [[[0.0; HIDDEN_DIM]; NUM_CELLS]; ITERATIONS + 1];
        self.hidden_pre = [[[0.0; MLP_WIDTH]; NUM_CELLS]; ITERATIONS];
        self.logits = [[0.0; VOCAB_SIZE]; MAX_SEQ_LEN];
        self.dstate = [[0.0; HIDDEN_DIM]; NUM_CELLS];
        self.dprev = [[0.0; HIDDEN_DIM]; NUM_CELLS];
    }
}

impl Default for TrainingWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingBatch {
    pub const fn empty() -> Self {
        Self {
            ids: [[0; NUM_CELLS]; LOGICAL_BATCH_SIZE],
            lengths: [0; LOGICAL_BATCH_SIZE],
            size: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrainStepReport {
    pub loss: F32,
    pub token_count: u32,
    pub grad_norm: F32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EvaluationReport {
    pub loss: F32,
    pub token_count: u32,
    pub correct_tokens: u32,
    pub token_accuracy: F32,
    pub exact_sequence_accuracy: F32,
}

#[inline(always)]
fn expf(x: F32) -> F32 {
    // Deterministic range-reduced exp approximation.  The range is sufficient
    // for the stable softmax (inputs are shifted by their row maximum).
    let x = if x < -16.0 {
        -16.0
    } else if x > 16.0 {
        16.0
    } else {
        x
    };
    let scaled = x * 1.4426950408889634;
    let k = if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    };
    let r = x - (k as F32) * 0.6931471805599453;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.04166667 + r * 0.008333333))));
    p * pow2i(k)
}

#[inline(always)]
fn pow2i(n: i32) -> F32 {
    if n < -126 {
        return 0.0;
    }
    if n > 127 {
        return F32::MAX;
    }
    F32::from_bits(((n + 127) as u32) << 23)
}

#[inline(always)]
fn logf(x: F32) -> F32 {
    let x = if x < 1.0e-30 { 1.0e-30 } else { x };
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = F32::from_bits((bits & 0x7f_ffff) | 0x3f80_0000);
    let y = (mantissa - 1.0) / (mantissa + 1.0);
    let y2 = y * y;
    let series = y * (2.0 + y2 * (0.6666667 + y2 * (0.4 + y2 * (0.2857143 + y2 * 0.2222222))));
    series + (exponent as F32) * 0.6931471805599453
}

#[inline(always)]
fn sqrtf(x: F32) -> F32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = F32::from_bits((x.to_bits() >> 1) + 0x1fc0_0000);
    for _ in 0..6 {
        y = 0.5 * (y + x / y);
    }
    y
}

#[inline(always)]
fn weight(
    weights: &[F32; PARAMETER_COUNT],
    offset: usize,
    row: usize,
    col: usize,
    width: usize,
) -> F32 {
    weights[offset + row * width + col]
}

#[inline(always)]
fn grad_add(grad: &mut [F32; PARAMETER_COUNT], index: usize, value: F32) {
    grad[index] += value;
}

fn forward_backward(
    weights: &[F32; PARAMETER_COUNT],
    ids: &[u8; NUM_CELLS],
    length: usize,
    workspace: &mut TrainingWorkspace,
    gradient: &mut [F32; PARAMETER_COUNT],
    backward: bool,
) -> (F32, u32) {
    workspace.reset_sample();
    let states = &mut workspace.states;
    let hidden_pre = &mut workspace.hidden_pre;
    let logits = &mut workspace.logits;
    for iteration in 0..ITERATIONS {
        for cell in 0..NUM_CELLS {
            let mut input = [0.0; UPDATE_INPUT_DIM];
            for relative in -(RADIUS as isize)..=(RADIUS as isize) {
                let neighbour = cell as isize + relative;
                if neighbour >= 0 && neighbour < NUM_CELLS as isize {
                    let source = neighbour as usize;
                    let cursor = ((relative + RADIUS as isize) as usize) * HIDDEN_DIM;
                    input[cursor..cursor + HIDDEN_DIM].copy_from_slice(&states[iteration][source]);
                }
            }
            let embedding = EMBEDDING_OFFSET + ids[cell] as usize * EMBEDDING_DIM;
            input[80..88].copy_from_slice(&weights[embedding..embedding + EMBEDDING_DIM]);
            for row in 0..MLP_WIDTH {
                let mut value = weights[UPDATE_IN_BIAS_OFFSET + row];
                for col in 0..UPDATE_INPUT_DIM {
                    value += weight(weights, UPDATE_IN_WEIGHT_OFFSET, row, col, UPDATE_INPUT_DIM)
                        * input[col];
                }
                hidden_pre[iteration][cell][row] = value;
            }
            for row in 0..HIDDEN_DIM {
                let mut value = weights[UPDATE_OUT_BIAS_OFFSET + row];
                for col in 0..MLP_WIDTH {
                    value += weight(weights, UPDATE_OUT_WEIGHT_OFFSET, row, col, MLP_WIDTH)
                        * relu(hidden_pre[iteration][cell][col]);
                }
                let unclamped = states[iteration][cell][row] + value;
                states[iteration + 1][cell][row] = clamp_state(unclamped);
            }
        }
    }
    for cell in 0..length {
        for row in 0..VOCAB_SIZE {
            let mut value = weights[OUTPUT_BIAS_OFFSET + row];
            for col in 0..HIDDEN_DIM {
                value += weight(weights, OUTPUT_WEIGHT_OFFSET, row, col, HIDDEN_DIM)
                    * states[ITERATIONS][cell][col];
            }
            logits[cell][row] = value;
        }
    }
    let dstate = &mut workspace.dstate;
    let mut loss = 0.0;
    let mut correct = 0;
    for cell in 0..length {
        let mut maximum = logits[cell][0];
        for row in 1..VOCAB_SIZE {
            if logits[cell][row] > maximum {
                maximum = logits[cell][row];
            }
        }
        let mut denominator = 0.0;
        for row in 0..VOCAB_SIZE {
            denominator += expf(logits[cell][row] - maximum);
        }
        let target = ids[cell] as usize;
        loss += logf(denominator) + maximum - logits[cell][target];
        let mut prediction = 0;
        let mut best = logits[cell][0];
        for row in 0..VOCAB_SIZE {
            let probability = expf(logits[cell][row] - maximum) / denominator;
            let dlogit = probability - if row == target { 1.0 } else { 0.0 };
            for col in 0..HIDDEN_DIM {
                grad_add(
                    gradient,
                    OUTPUT_WEIGHT_OFFSET + row * HIDDEN_DIM + col,
                    dlogit * states[ITERATIONS][cell][col],
                );
                dstate[cell][col] +=
                    weight(weights, OUTPUT_WEIGHT_OFFSET, row, col, HIDDEN_DIM) * dlogit;
            }
            grad_add(gradient, OUTPUT_BIAS_OFFSET + row, dlogit);
            if logits[cell][row] > best {
                best = logits[cell][row];
                prediction = row;
            }
        }
        if prediction == target {
            correct += 1;
        }
    }
    if !backward {
        return (loss, correct);
    }
    for iteration in (0..ITERATIONS).rev() {
        let dprev = &mut workspace.dprev;
        *dprev = [[0.0; HIDDEN_DIM]; NUM_CELLS];
        for cell in 0..NUM_CELLS {
            let mut dinput = [0.0; UPDATE_INPUT_DIM];
            let mut ddelta = [0.0; HIDDEN_DIM];
            for row in 0..HIDDEN_DIM {
                let unclamped = states[iteration][cell][row] + {
                    let mut value = weights[UPDATE_OUT_BIAS_OFFSET + row];
                    for col in 0..MLP_WIDTH {
                        value += weight(weights, UPDATE_OUT_WEIGHT_OFFSET, row, col, MLP_WIDTH)
                            * relu(hidden_pre[iteration][cell][col]);
                    }
                    value
                };
                let d = if unclamped >= -1.0 && unclamped <= 1.0 {
                    dstate[cell][row]
                } else {
                    0.0
                };
                dprev[cell][row] += d;
                ddelta[row] = d;
                grad_add(gradient, UPDATE_OUT_BIAS_OFFSET + row, d);
                for col in 0..MLP_WIDTH {
                    let h = relu(hidden_pre[iteration][cell][col]);
                    grad_add(
                        gradient,
                        UPDATE_OUT_WEIGHT_OFFSET + row * MLP_WIDTH + col,
                        d * h,
                    );
                }
            }
            let mut dhidden = [0.0; MLP_WIDTH];
            for col in 0..MLP_WIDTH {
                for row in 0..HIDDEN_DIM {
                    dhidden[col] += weight(weights, UPDATE_OUT_WEIGHT_OFFSET, row, col, MLP_WIDTH)
                        * ddelta[row];
                }
                if hidden_pre[iteration][cell][col] <= 0.0 {
                    dhidden[col] = 0.0;
                }
                grad_add(
                    gradient,
                    UPDATE_IN_BIAS_OFFSET + col,
                    dhidden[col],
                );
                for input in 0..UPDATE_INPUT_DIM {
                    let value = input_value(weights, &states[iteration], ids[cell], cell, input);
                    grad_add(
                        gradient,
                        UPDATE_IN_WEIGHT_OFFSET + col * UPDATE_INPUT_DIM + input,
                        dhidden[col] * value,
                    );
                    dinput[input] += weight(
                        weights,
                        UPDATE_IN_WEIGHT_OFFSET,
                        col,
                        input,
                        UPDATE_INPUT_DIM,
                    ) * dhidden[col];
                }
            }
            for relative in -(RADIUS as isize)..=(RADIUS as isize) {
                let neighbour = cell as isize + relative;
                if neighbour >= 0 && neighbour < NUM_CELLS as isize {
                    let cursor = ((relative + RADIUS as isize) as usize) * HIDDEN_DIM;
                    for row in 0..HIDDEN_DIM {
                        dprev[neighbour as usize][row] += dinput[cursor + row];
                    }
                }
            }
            // PyTorch's `nn.Embedding(..., padding_idx=0)` never accumulates
            // a gradient for the padding row, even though padded embeddings
            // participate in the cellular forward pass.
            if ids[cell] != 0 {
                let embedding = EMBEDDING_OFFSET + ids[cell] as usize * EMBEDDING_DIM;
                for row in 0..EMBEDDING_DIM {
                    grad_add(gradient, embedding + row, dinput[80 + row]);
                }
            }
        }
        *dstate = *dprev;
    }
    (loss, correct)
}

/// Diagnostic-only probes used by the PVM memory/stack investigation.  They
/// intentionally reuse the production forward/backward implementation.
pub fn diagnostic_sample_forward(
    state: &TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
) -> TrainStepReport {
    let length = (batch.lengths[0] as usize).min(MAX_SEQ_LEN);
    let gradient = &mut workspace.gradient as *mut [F32; PARAMETER_COUNT];
    let (loss, _) = unsafe { forward_backward(&state.weights, &batch.ids[0], length, workspace, &mut *gradient, false) };
    TrainStepReport {
        loss,
        token_count: length as u32,
        grad_norm: 0.0,
    }
}

pub fn diagnostic_sample_backward(
    state: &TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
) -> TrainStepReport {
    let length = (batch.lengths[0] as usize).min(MAX_SEQ_LEN);
    let gradient = &mut workspace.gradient as *mut [F32; PARAMETER_COUNT];
    let (loss, _) = unsafe { forward_backward(&state.weights, &batch.ids[0], length, workspace, &mut *gradient, true) };
    TrainStepReport {
        loss,
        token_count: length as u32,
        grad_norm: 0.0,
    }
}

#[inline(always)]
fn input_value(
    weights: &[F32; PARAMETER_COUNT],
    state: &[[F32; HIDDEN_DIM]; NUM_CELLS],
    id: u8,
    cell: usize,
    input: usize,
) -> F32 {
    if input >= 80 {
        return weights[EMBEDDING_OFFSET + id as usize * EMBEDDING_DIM + input - 80];
    }
    let relative = input / HIDDEN_DIM;
    let neighbour = cell as isize + relative as isize - RADIUS as isize;
    if neighbour < 0 || neighbour >= NUM_CELLS as isize {
        0.0
    } else {
        state[neighbour as usize][input % HIDDEN_DIM]
    }
}

#[inline(always)]
fn relu(x: F32) -> F32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}
#[inline(always)]
fn clamp_state(x: F32) -> F32 {
    if x < -1.0 {
        -1.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

pub fn train_step(state: &mut TrainingState, batch: &TrainingBatch) -> TrainStepReport {
    let mut workspace = TrainingWorkspace::new();
    train_step_with_workspace(state, batch, &mut workspace)
}

/// Execute one canonical optimizer step and copy the post-normalization,
/// post-global-clipping gradient into `gradient_out`.  The extra output is
/// used only by the Native fidelity gate; production callers can use the
/// smaller `train_step` wrapper above.
pub fn train_step_with_gradient(
    state: &mut TrainingState,
    batch: &TrainingBatch,
    gradient_out: &mut [F32; PARAMETER_COUNT],
) -> TrainStepReport {
    let mut workspace = TrainingWorkspace::new();
    let mut accumulator = GradientAccumulator::new();
    accumulate_batch_gradients(state, batch, &mut workspace, &mut accumulator);
    let report = finalize_adamw_step(state, &mut accumulator);
    *gradient_out = accumulator.gradient;
    report
}

pub fn train_step_with_workspace(
    state: &mut TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
) -> TrainStepReport {
    let mut accumulator = GradientAccumulator::new();
    train_step_with_accumulator(state, batch, workspace, &mut accumulator)
}

/// Execute a step using caller-owned accumulator storage (required by the
/// small PVM stack).
pub fn train_step_with_accumulator(
    state: &mut TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
    accumulator: &mut GradientAccumulator,
) -> TrainStepReport {
    accumulator.reset();
    accumulate_batch_gradients(state, batch, workspace, accumulator);
    finalize_adamw_step(state, accumulator)
}

/// Accumulate unnormalized per-token gradients in deterministic sample order.
pub fn accumulate_batch_gradients(
    state: &TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
    accumulator: &mut GradientAccumulator,
) {
    accumulate_batch_gradients_into(
        state,
        batch,
        workspace,
        &mut accumulator.gradient,
        &mut accumulator.loss_sum,
        &mut accumulator.token_count,
    );
}

/// Stack-bounded accumulation primitive used by independent PVM leaves.
/// Callers provide the destination fields directly so computing a leaf does
/// not materialize a second 18 KiB `GradientAccumulator` frame.
pub fn accumulate_batch_gradients_into(
    state: &TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
    gradient: &mut [F32; PARAMETER_COUNT],
    loss_sum: &mut F32,
    token_count: &mut u32,
) {
    for sample in 0..(batch.size as usize).min(LOGICAL_BATCH_SIZE) {
        let length = (batch.lengths[sample] as usize).min(MAX_SEQ_LEN);
        let (sample_loss, _) =
            forward_backward(&state.weights, &batch.ids[sample], length, workspace, gradient, true);
        *loss_sum += sample_loss;
        *token_count += length as u32;
    }
}

/// Compute one raw gradient leaf against frozen model/optimizer state.
/// `batch.size` must be the configured parallel shard size (the final API
/// rejects malformed sizes at its coordinator boundary); this core routine is
/// deliberately allocation-free and reuses the canonical accumulator path.
pub fn compute_gradient_leaf(
    state: &TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
    out: &mut PartialGradientV1,
) {
    out.gradient = [0.0; PARAMETER_COUNT];
    out.loss_sum = 0.0;
    out.token_count = 0;
    out.processed_samples = batch.size;
    accumulate_batch_gradients_into(
        state,
        batch,
        workspace,
        &mut out.gradient,
        &mut out.loss_sum,
        &mut out.token_count,
    );
}

/// Deterministically merge two raw leaves.  The caller chooses the tree shape;
/// this primitive never depends on completion order or shared mutable sums.
pub fn merge_partial_gradients(
    left: &PartialGradientV1,
    right: &PartialGradientV1,
    out: &mut PartialGradientV1,
) {
    for index in 0..PARAMETER_COUNT {
        out.gradient[index] = left.gradient[index] + right.gradient[index];
    }
    out.loss_sum = left.loss_sum + right.loss_sum;
    out.token_count = left.token_count + right.token_count;
    out.processed_samples = left.processed_samples + right.processed_samples;
}

/// Fixed five-level balanced reduction for exactly 32 leaves.  Leaves may be
/// produced in any order by an executor, but must be placed by index before
/// calling this function.  `scratch` is caller-owned to keep PVM stack use
/// bounded and to make the reduction storage explicit.
pub fn reduce_32_leaves(
    leaves: &[PartialGradientV1; PARALLEL_LEAF_COUNT],
    scratch: &mut [PartialGradientV1; PARALLEL_LEAF_COUNT],
) -> PartialGradientV1 {
    reduce_partial_gradients(leaves, scratch).expect("the 32-leaf tree is non-empty")
}

/// In-place tree reduction for memory-constrained guests.  The left slot is
/// overwritten by each pairwise merge, so only one 32-leaf arena is needed.
pub fn reduce_32_leaves_in_place(
    leaves: &mut [PartialGradientV1; PARALLEL_LEAF_COUNT],
) -> PartialGradientV1 {
    reduce_32_leaves_in_place_ref(leaves);
    leaves[0]
}

/// Stack-bounded reference form for guests that already own the leaf arena.
/// It avoids copying the 18 KiB root record onto the PVM call stack.
pub fn reduce_32_leaves_in_place_ref(
    leaves: &mut [PartialGradientV1; PARALLEL_LEAF_COUNT],
) -> &PartialGradientV1 {
    let mut stride = 1usize;
    while stride < PARALLEL_LEAF_COUNT {
        let pair_span = stride * 2;
        let mut left_index = 0usize;
        while left_index < PARALLEL_LEAF_COUNT {
            let right_index = left_index + stride;
            let (before_right, at_right) = leaves.split_at_mut(right_index);
            let left = &mut before_right[left_index];
            let right = &at_right[0];
            for parameter in 0..PARAMETER_COUNT {
                left.gradient[parameter] = left.gradient[parameter] + right.gradient[parameter];
            }
            left.loss_sum = left.loss_sum + right.loss_sum;
            left.token_count = left.token_count + right.token_count;
            left.processed_samples = left.processed_samples + right.processed_samples;
            left_index += pair_span;
        }
        stride *= 2;
    }
    &leaves[0]
}

/// Slice form of the fixed pairwise reduction, useful to executors that keep
/// leaf storage in a heap-backed arena.  The caller still supplies leaves in
/// canonical index order; completion order is never observed here.
pub fn reduce_partial_gradients(
    leaves: &[PartialGradientV1],
    scratch: &mut [PartialGradientV1],
) -> Option<PartialGradientV1> {
    if leaves.is_empty() || scratch.len() < leaves.len() {
        return None;
    }
    for index in 0..leaves.len() {
        scratch[index] = leaves[index].clone();
    }
    let mut width = leaves.len();
    while width > 1 {
        for index in 0..(width / 2) {
            let left = scratch[index * 2].clone();
            let right = scratch[index * 2 + 1].clone();
            merge_partial_gradients(&left, &right, &mut scratch[index]);
        }
        width /= 2;
    }
    Some(scratch[0].clone())
}

/// Execute the canonical tree32 logical step.  This helper schedules leaves
/// in index order for a deterministic Native reference; production executors
/// may compute the same leaves concurrently and then call the same reduction
/// and finalize primitives.
pub fn train_step_tree32(
    state: &mut TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
    leaves: &mut [PartialGradientV1; PARALLEL_LEAF_COUNT],
    scratch: &mut [PartialGradientV1; PARALLEL_LEAF_COUNT],
) -> TrainStepReport {
    debug_assert_eq!(batch.size as usize, LOGICAL_BATCH_SIZE);
    for leaf_index in 0..PARALLEL_LEAF_COUNT {
        let start = leaf_index * PARALLEL_SHARD_SIZE;
        let mut shard = TrainingBatch::empty();
        shard.size = PARALLEL_SHARD_SIZE as u16;
        for row in 0..PARALLEL_SHARD_SIZE {
            shard.ids[row] = batch.ids[start + row];
            shard.lengths[row] = batch.lengths[start + row];
        }
        compute_gradient_leaf(state, &shard, workspace, &mut leaves[leaf_index]);
    }
    let root = reduce_32_leaves(leaves, scratch);
    let mut accumulator = GradientAccumulator {
        gradient: root.gradient,
        loss_sum: root.loss_sum,
        token_count: root.token_count,
    };
    finalize_adamw_step(state, &mut accumulator)
}

/// Normalize once, clip once and apply one AdamW update.
pub fn finalize_adamw_step(
    state: &mut TrainingState,
    accumulator: &mut GradientAccumulator,
) -> TrainStepReport {
    let tokens = accumulator.token_count;
    if tokens == 0 {
        return TrainStepReport::default();
    }
    let inverse_tokens = 1.0 / tokens as F32;
    let mut norm_squared = 0.0;
    for value in &mut accumulator.gradient {
        *value *= inverse_tokens;
        norm_squared += *value * *value;
    }
    let norm = sqrtf(norm_squared);
    let scale = if norm > 1.0 { 1.0 / norm } else { 1.0 };
    for value in &mut accumulator.gradient {
        *value *= scale;
    }
    state.step += 1;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let lr = 0.001;
    let eps = 1.0e-8;
    let bias1 = 1.0 - powf(beta1, state.step);
    let bias2 = 1.0 - powf(beta2, state.step);
    for index in 0..PARAMETER_COUNT {
        state.adam_m[index] =
            beta1 * state.adam_m[index] + (1.0 - beta1) * accumulator.gradient[index];
        state.adam_v[index] = beta2 * state.adam_v[index]
            + (1.0 - beta2) * accumulator.gradient[index] * accumulator.gradient[index];
        let mhat = state.adam_m[index] / bias1;
        let vhat = state.adam_v[index] / bias2;
        state.weights[index] -= lr * mhat / (sqrtf(vhat) + eps);
    }
    TrainStepReport {
        loss: accumulator.loss_sum * inverse_tokens,
        token_count: tokens,
        grad_norm: norm,
    }
}

/// Evaluate a fixed batch without mutating optimizer state.  The same forward
/// and CE implementation is used as in `train_step`; gradients are discarded.
pub fn evaluate_batch(state: &TrainingState, batch: &TrainingBatch) -> TrainStepReport {
    let report = evaluate_batch_report(state, batch);
    TrainStepReport {
        loss: report.loss,
        token_count: report.token_count,
        grad_norm: 0.0,
    }
}

pub fn evaluate_batch_report(state: &TrainingState, batch: &TrainingBatch) -> EvaluationReport {
    let mut workspace = TrainingWorkspace::new();
    let mut loss = 0.0;
    let mut tokens = 0u32;
    let mut correct_tokens = 0u32;
    let mut exact_sequences = 0u32;
    let sample_count = (batch.size as usize).min(LOGICAL_BATCH_SIZE);
    for sample in 0..(batch.size as usize).min(LOGICAL_BATCH_SIZE) {
        let length = (batch.lengths[sample] as usize).min(MAX_SEQ_LEN);
        let gradient = &mut workspace.gradient as *mut [F32; PARAMETER_COUNT];
        let (sample_loss, sample_correct) = unsafe {
            forward_backward(&state.weights, &batch.ids[sample], length, &mut workspace, &mut *gradient, true)
        };
        loss += sample_loss;
        tokens += length as u32;
        correct_tokens += sample_correct;
        if sample_correct == length as u32 {
            exact_sequences += 1;
        }
    }
    EvaluationReport {
        loss: if tokens == 0 {
            0.0
        } else {
            loss / tokens as F32
        },
        token_count: tokens,
        correct_tokens,
        token_accuracy: if tokens == 0 {
            0.0
        } else {
            correct_tokens as F32 / tokens as F32
        },
        exact_sequence_accuracy: if sample_count == 0 {
            0.0
        } else {
            exact_sequences as F32 / sample_count as F32
        },
    }
}

#[inline(always)]
fn powf(base: F32, exponent: u64) -> F32 {
    let mut result = 1.0;
    for _ in 0..exponent {
        result *= base;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adamw_step_advances_and_clips() {
        let mut state = TrainingState::from_weights([0.0; PARAMETER_COUNT]);
        let mut batch = TrainingBatch::empty();
        batch.size = 1;
        batch.lengths[0] = 1;
        batch.ids[0][0] = 1;
        let report = train_step(&mut state, &batch);
        assert_eq!(state.step, 1);
        assert!(report.loss.is_finite());
        assert!(report.grad_norm.is_finite());
    }
    #[test]
    fn microbatch_shape_is_logical_batch() {
        assert_eq!(LOGICAL_BATCH_SIZE, 256);
    }

    #[test]
    fn padding_embedding_row_is_frozen() {
        let mut state = TrainingState::from_weights([0.0; PARAMETER_COUNT]);
        let mut batch = TrainingBatch::empty();
        batch.size = 1;
        batch.lengths[0] = 1;
        batch.ids[0][0] = 0;
        let mut gradient = [0.0; PARAMETER_COUNT];
        train_step_with_gradient(&mut state, &batch, &mut gradient);
        assert!(gradient[..EMBEDDING_DIM].iter().all(|value| *value == 0.0));
        assert!(state.weights[..EMBEDDING_DIM]
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn evaluation_reports_token_and_exact_accuracy() {
        let state = TrainingState::from_weights([0.0; PARAMETER_COUNT]);
        let mut batch = TrainingBatch::empty();
        batch.size = 1;
        batch.lengths[0] = 1;
        batch.ids[0][0] = 0;
        let report = evaluate_batch_report(&state, &batch);
        assert_eq!(report.token_count, 1);
        assert_eq!(report.correct_tokens, 1);
        assert_eq!(report.token_accuracy, 1.0);
        assert_eq!(report.exact_sequence_accuracy, 1.0);
    }

    #[test]
    fn accumulator_shard_does_not_mutate_optimizer_state() {
        let state = TrainingState::from_weights([0.01; PARAMETER_COUNT]);
        let before = state;
        let mut batch = TrainingBatch::empty(); batch.size = 1; batch.lengths[0] = 1; batch.ids[0][0] = 1;
        let mut workspace = TrainingWorkspace::new(); let mut acc = GradientAccumulator::new();
        accumulate_batch_gradients(&state, &batch, &mut workspace, &mut acc);
        assert_eq!(state.weights, before.weights); assert_eq!(state.adam_m, before.adam_m);
        assert_eq!(state.adam_v, before.adam_v); assert_eq!(state.step, before.step);
        assert_eq!(acc.token_count, 1); assert!(acc.loss_sum.is_finite());
    }

    #[test]
    fn sequential_accumulation_is_bit_exact() {
        let weights = [0.01; PARAMETER_COUNT];
        let mut full = TrainingBatch::empty(); full.size = 2;
        for row in 0..2 { full.lengths[row] = 1; full.ids[row][0] = (row + 1) as u8; }
        let mut ws = TrainingWorkspace::new(); let mut one = GradientAccumulator::new();
        accumulate_batch_gradients(&TrainingState::from_weights(weights), &full, &mut ws, &mut one);
        let mut a = GradientAccumulator::new(); let mut ws2 = TrainingWorkspace::new();
        let mut first = full; first.size = 1;
        accumulate_batch_gradients(&TrainingState::from_weights(weights), &first, &mut ws2, &mut a);
        let mut second = TrainingBatch::empty(); second.size = 1; second.ids[0] = full.ids[1]; second.lengths[0] = full.lengths[1];
        accumulate_batch_gradients(&TrainingState::from_weights(weights), &second, &mut ws2, &mut a);
        assert_eq!(one.gradient, a.gradient); assert_eq!(one.loss_sum.to_bits(), a.loss_sum.to_bits());
        assert_eq!(one.token_count, a.token_count);
        let mut mono = TrainingState::from_weights(weights); let mut chunk = TrainingState::from_weights(weights);
        let mut g = GradientAccumulator::new(); let mut w = TrainingWorkspace::new();
        accumulate_batch_gradients(&mono, &full, &mut w, &mut g); finalize_adamw_step(&mut mono, &mut g);
        let mut g2 = GradientAccumulator::new(); accumulate_batch_gradients(&chunk, &first, &mut w, &mut g2); accumulate_batch_gradients(&chunk, &second, &mut w, &mut g2); finalize_adamw_step(&mut chunk, &mut g2);
        assert_eq!(mono.weights, chunk.weights); assert_eq!(mono.adam_m, chunk.adam_m); assert_eq!(mono.adam_v, chunk.adam_v); assert_eq!(mono.step, chunk.step);
    }

    #[test]
    fn tree32_reduction_is_completion_order_independent() {
        let mut batch = TrainingBatch::empty();
        batch.size = LOGICAL_BATCH_SIZE as u16;
        for row in 0..LOGICAL_BATCH_SIZE {
            batch.lengths[row] = 1;
            batch.ids[row][0] = (row % VOCAB_SIZE) as u8;
        }
        let initial = TrainingState::from_weights([0.01; PARAMETER_COUNT]);
        let mut state_a = initial;
        let mut state_b = initial;
        let mut ws_a = TrainingWorkspace::new();
        let mut ws_b = TrainingWorkspace::new();
        let mut leaves_a = std::vec::Vec::from_iter(
            (0..PARALLEL_LEAF_COUNT).map(|_| PartialGradientV1::new()),
        );
        let mut leaves_b = std::vec::Vec::from_iter(
            (0..PARALLEL_LEAF_COUNT).map(|_| PartialGradientV1::new()),
        );
        let mut scratch_a = std::vec::Vec::from_iter(
            (0..PARALLEL_LEAF_COUNT).map(|_| PartialGradientV1::new()),
        );
        let mut scratch_b = std::vec::Vec::from_iter(
            (0..PARALLEL_LEAF_COUNT).map(|_| PartialGradientV1::new()),
        );
        // Simulate arbitrary completion order while retaining index-addressed
        // output slots.  The tree must be identical to the canonical order.
        for leaf_index in (0..PARALLEL_LEAF_COUNT).rev() {
            let start = leaf_index * PARALLEL_SHARD_SIZE;
            let mut shard = TrainingBatch::empty();
            shard.size = PARALLEL_SHARD_SIZE as u16;
            for row in 0..PARALLEL_SHARD_SIZE {
                shard.ids[row] = batch.ids[start + row];
                shard.lengths[row] = batch.lengths[start + row];
            }
            compute_gradient_leaf(&state_a, &shard, &mut ws_a, &mut leaves_a[leaf_index]);
        }
        for leaf_index in 0..PARALLEL_LEAF_COUNT {
            let start = leaf_index * PARALLEL_SHARD_SIZE;
            let mut shard = TrainingBatch::empty();
            shard.size = PARALLEL_SHARD_SIZE as u16;
            for row in 0..PARALLEL_SHARD_SIZE {
                shard.ids[row] = batch.ids[start + row];
                shard.lengths[row] = batch.lengths[start + row];
            }
            compute_gradient_leaf(&state_b, &shard, &mut ws_b, &mut leaves_b[leaf_index]);
        }
        let report_a = finalize_adamw_step(&mut state_a, &mut {
            let root = reduce_partial_gradients(&leaves_a, &mut scratch_a).unwrap();
            GradientAccumulator { gradient: root.gradient, loss_sum: root.loss_sum, token_count: root.token_count }
        });
        let report_b = finalize_adamw_step(&mut state_b, &mut {
            let root = reduce_partial_gradients(&leaves_b, &mut scratch_b).unwrap();
            GradientAccumulator { gradient: root.gradient, loss_sum: root.loss_sum, token_count: root.token_count }
        });
        assert_eq!(report_a.loss.to_bits(), report_b.loss.to_bits());
        assert_eq!(report_a.grad_norm.to_bits(), report_b.grad_norm.to_bits());
        assert_eq!(report_a.token_count, report_b.token_count);
        assert_eq!(state_a.weights, state_b.weights);
        assert_eq!(state_a.adam_m, state_b.adam_m);
        assert_eq!(state_a.adam_v, state_b.adam_v);
        assert_eq!(state_a.step, state_b.step);
    }

    #[test]
    fn in_place_tree32_matches_canonical_synthetic_metadata() {
        let mut leaves = [const { PartialGradientV1::new() }; PARALLEL_LEAF_COUNT];
        for (index, leaf) in leaves.iter_mut().enumerate() {
            leaf.token_count = (index + 1) as u32;
            leaf.processed_samples = 8;
            leaf.loss_sum = (index as f32) * 0.25 + 0.125;
            for parameter in 0..PARAMETER_COUNT {
                leaf.gradient[parameter] = (index as f32 + 1.0) * 0.001
                    + parameter as f32 * 0.000001;
            }
        }
        let mut scratch = [const { PartialGradientV1::new() }; PARALLEL_LEAF_COUNT];
        let expected = reduce_partial_gradients(&leaves, &mut scratch).unwrap();
        let actual = reduce_32_leaves_in_place_ref(&mut leaves);
        assert_eq!(actual.gradient, expected.gradient);
        assert_eq!(actual.loss_sum.to_bits(), expected.loss_sum.to_bits());
        assert_eq!(actual.token_count, 528);
        assert_eq!(actual.token_count, expected.token_count);
        assert_eq!(actual.processed_samples, 256);
        assert_eq!(actual.processed_samples, expected.processed_samples);
    }

    #[test]
    fn in_place_tree32_matches_canonical_native_leaves() {
        let mut batch = TrainingBatch::empty();
        batch.size = LOGICAL_BATCH_SIZE as u16;
        for row in 0..LOGICAL_BATCH_SIZE {
            batch.lengths[row] = 1;
            batch.ids[row][0] = (row % VOCAB_SIZE) as u8;
        }
        let state = TrainingState::from_weights([0.01; PARAMETER_COUNT]);
        let mut leaves = [const { PartialGradientV1::new() }; PARALLEL_LEAF_COUNT];
        let mut workspace = TrainingWorkspace::new();
        for leaf_index in 0..PARALLEL_LEAF_COUNT {
            let mut shard = TrainingBatch::empty();
            shard.size = PARALLEL_SHARD_SIZE as u16;
            let start = leaf_index * PARALLEL_SHARD_SIZE;
            for row in 0..PARALLEL_SHARD_SIZE {
                shard.ids[row] = batch.ids[start + row];
                shard.lengths[row] = batch.lengths[start + row];
            }
            compute_gradient_leaf(&state, &shard, &mut workspace, &mut leaves[leaf_index]);
        }
        let mut scratch = [const { PartialGradientV1::new() }; PARALLEL_LEAF_COUNT];
        let expected = reduce_partial_gradients(&leaves, &mut scratch).unwrap();
        let actual = reduce_32_leaves_in_place_ref(&mut leaves);
        assert_eq!(actual.gradient, expected.gradient);
        assert_eq!(actual.loss_sum.to_bits(), expected.loss_sum.to_bits());
        assert_eq!(actual.token_count, expected.token_count);
        assert_eq!(actual.processed_samples, expected.processed_samples);
    }

    #[test]
    fn round_state_enforces_order_and_finalizes_once() {
        let commitment = [7u8; 32]; let model = [8u8; 32]; let optimizer = [9u8; 32];
        let mut round = TrainingRoundStateV1::new(0, 0, commitment, model, optimizer);
        let bad = GradientAccumulatorStateV1::empty();
        assert_eq!(round.accept_mca(0, 0, commitment, model, 1, 8, 16, bad), Err(RoundError::OutOfOrderShard));
        for index in 0..32u16 {
            let mut candidate = round.accumulator; candidate.processed_samples = (index + 1) * 8; candidate.token_count = candidate.processed_samples as u32;
            assert_eq!(round.accept_mca(0, 0, commitment, model, index, index * 8, index * 8 + 8, candidate), Ok(()));
        }
        assert_eq!(round.phase, TrainingPhaseV1::FinalizeReady);
        let mut state = TrainingState::from_weights([0.0; PARAMETER_COUNT]);
        assert!(round.finalize(&mut state).is_ok()); assert_eq!(state.step, 1); assert_eq!(round.phase, TrainingPhaseV1::Idle);
        assert_eq!(round.finalize(&mut state), Err(RoundError::InvalidPhase));
    }

    #[test]
    fn parallel_job_accepts_out_of_order_immutable_leaves_and_rejects_mixing() {
        let mut job = ParallelTrainingJobV1::new([1; 32], 4, 7, [2; 32], [3; 32], [4; 32]);
        assert_eq!(job.mark_root_ready(), Err(ParallelJobErrorV1::MissingLeaf));
        assert_eq!(job.accept_leaf([9; 32], 4, 7, [2; 32], [3; 32], [4; 32], 0, 0, 8, [5; 32]), Err(ParallelJobErrorV1::WrongJob));
        assert_eq!(job.accept_leaf([1; 32], 4, 7, [2; 32], [3; 32], [4; 32], 3, 24, 32, [8; 32]), Ok(()));
        assert_eq!(job.accept_leaf([1; 32], 4, 7, [2; 32], [3; 32], [4; 32], 3, 24, 32, [8; 32]), Err(ParallelJobErrorV1::DuplicateLeaf));
        assert_eq!(job.accept_leaf([1; 32], 4, 7, [2; 32], [3; 32], [4; 32], 3, 24, 32, [9; 32]), Err(ParallelJobErrorV1::ConflictingLeaf));
        for index in 0..PARALLEL_LEAF_COUNT as u16 {
            if index == 3 { continue; }
            assert_eq!(job.accept_leaf([1; 32], 4, 7, [2; 32], [3; 32], [4; 32], index, index * 8, index * 8 + 8, [index as u8; 32]), Ok(()));
        }
        assert_eq!(job.mark_root_ready(), Ok(()));
        assert_eq!(job.begin_finalize([2; 32]), Ok(()));
        assert_eq!(job.complete(), Ok(()));
        assert_eq!(job.complete(), Err(ParallelJobErrorV1::InvalidStatus));
    }
}
