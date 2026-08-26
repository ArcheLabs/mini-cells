#![no_std]

//! The authoritative FP32 Echo training step.
//!
//! This crate intentionally has no host, filesystem, protocol, or allocator
//! dependencies.  Native and guest runners call this same implementation.

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
                    &mut workspace.gradient,
                    OUTPUT_WEIGHT_OFFSET + row * HIDDEN_DIM + col,
                    dlogit * states[ITERATIONS][cell][col],
                );
                dstate[cell][col] +=
                    weight(weights, OUTPUT_WEIGHT_OFFSET, row, col, HIDDEN_DIM) * dlogit;
            }
            grad_add(&mut workspace.gradient, OUTPUT_BIAS_OFFSET + row, dlogit);
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
                grad_add(&mut workspace.gradient, UPDATE_OUT_BIAS_OFFSET + row, d);
                for col in 0..MLP_WIDTH {
                    let h = relu(hidden_pre[iteration][cell][col]);
                    grad_add(
                        &mut workspace.gradient,
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
                    &mut workspace.gradient,
                    UPDATE_IN_BIAS_OFFSET + col,
                    dhidden[col],
                );
                for input in 0..UPDATE_INPUT_DIM {
                    let value = input_value(weights, &states[iteration], ids[cell], cell, input);
                    grad_add(
                        &mut workspace.gradient,
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
                    grad_add(&mut workspace.gradient, embedding + row, dinput[80 + row]);
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
    let (loss, _) = forward_backward(&state.weights, &batch.ids[0], length, workspace, false);
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
    let (loss, _) = forward_backward(&state.weights, &batch.ids[0], length, workspace, true);
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
    let report = train_step_with_workspace(state, batch, &mut workspace);
    *gradient_out = workspace.gradient;
    report
}

pub fn train_step_with_workspace(
    state: &mut TrainingState,
    batch: &TrainingBatch,
    workspace: &mut TrainingWorkspace,
) -> TrainStepReport {
    workspace.gradient = [0.0; PARAMETER_COUNT];
    let mut loss = 0.0;
    let mut tokens = 0u32;
    for sample in 0..(batch.size as usize).min(LOGICAL_BATCH_SIZE) {
        let length = (batch.lengths[sample] as usize).min(MAX_SEQ_LEN);
        let (sample_loss, _) =
            forward_backward(&state.weights, &batch.ids[sample], length, workspace, true);
        loss += sample_loss;
        tokens += length as u32;
    }
    if tokens == 0 {
        return TrainStepReport::default();
    }
    let inverse_tokens = 1.0 / tokens as F32;
    let mut norm_squared = 0.0;
    for value in &mut workspace.gradient {
        *value *= inverse_tokens;
        norm_squared += *value * *value;
    }
    let norm = sqrtf(norm_squared);
    let scale = if norm > 1.0 { 1.0 / norm } else { 1.0 };
    for value in &mut workspace.gradient {
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
            beta1 * state.adam_m[index] + (1.0 - beta1) * workspace.gradient[index];
        state.adam_v[index] = beta2 * state.adam_v[index]
            + (1.0 - beta2) * workspace.gradient[index] * workspace.gradient[index];
        let mhat = state.adam_m[index] / bias1;
        let vhat = state.adam_v[index] / bias2;
        state.weights[index] -= lr * mhat / (sqrtf(vhat) + eps);
    }
    TrainStepReport {
        loss: loss * inverse_tokens,
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
        let (sample_loss, sample_correct) = forward_backward(
            &state.weights,
            &batch.ids[sample],
            length,
            &mut workspace,
            true,
        );
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
}
