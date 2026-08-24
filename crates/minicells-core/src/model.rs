use blake2b_simd::Params;

use crate::fixed::{clamp_parameter, clamp_state, round_q16_16_to_q8_8};

pub const NUM_CELLS: usize = 64;
pub const MAX_SEQ_LEN: usize = 32;
pub const VOCAB_SIZE: usize = 44;
pub const EMBEDDING_DIM: usize = 8;
pub const HIDDEN_DIM: usize = 16;
pub const RADIUS: usize = 2;
pub const ITERATIONS: usize = 4;
pub const MLP_WIDTH: usize = 32;
pub const UPDATE_INPUT_DIM: usize = 88;
pub const PARAMETER_COUNT: usize = 4476;
pub const MODEL_BYTES: usize = PARAMETER_COUNT * 2;

pub const EMBEDDING_OFFSET: usize = 0;
pub const UPDATE_IN_WEIGHT_OFFSET: usize = 352;
pub const UPDATE_IN_BIAS_OFFSET: usize = 3168;
pub const UPDATE_OUT_WEIGHT_OFFSET: usize = 3200;
pub const UPDATE_OUT_BIAS_OFFSET: usize = 3712;
pub const OUTPUT_WEIGHT_OFFSET: usize = 3728;
pub const OUTPUT_BIAS_OFFSET: usize = 4432;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedModel {
    pub parameters: [i16; PARAMETER_COUNT],
}

impl Default for PackedModel {
    fn default() -> Self {
        Self {
            parameters: [0; PARAMETER_COUNT],
        }
    }
}

impl PackedModel {
    pub const fn from_parameters(parameters: [i16; PARAMETER_COUNT]) -> Self {
        Self { parameters }
    }

    pub fn encode_into(&self, output: &mut [u8]) -> Result<usize, ()> {
        if output.len() < MODEL_BYTES {
            return Err(());
        }
        for (index, value) in self.parameters.iter().enumerate() {
            output[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
        }
        Ok(MODEL_BYTES)
    }

    pub fn decode_from(input: &[u8]) -> Result<Self, ()> {
        let mut model = Self::default();
        model.decode_into(input)?;
        Ok(model)
    }

    pub fn decode_into(&mut self, input: &[u8]) -> Result<(), ()> {
        if input.len() != MODEL_BYTES {
            return Err(());
        }
        for (index, pair) in input.chunks_exact(2).enumerate() {
            let value = i16::from_le_bytes([pair[0], pair[1]]);
            if !(-2048..=2048).contains(&value) {
                return Err(());
            }
            self.parameters[index] = value;
        }
        Ok(())
    }

    pub fn bounded_add(&mut self, index: usize, delta: i32) {
        self.parameters[index] = clamp_parameter(self.parameters[index] as i32 + delta);
    }
}

pub struct Scratch {
    state_a: [[i16; HIDDEN_DIM]; NUM_CELLS],
    state_b: [[i16; HIDDEN_DIM]; NUM_CELLS],
}

impl Scratch {
    pub const fn new() -> Self {
        Self {
            state_a: [[0; HIDDEN_DIM]; NUM_CELLS],
            state_b: [[0; HIDDEN_DIM]; NUM_CELLS],
        }
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn linear(
    model: &PackedModel,
    weight_offset: usize,
    bias_offset: usize,
    inputs: &[i16],
    row: usize,
) -> i32 {
    let mut accumulator = (model.parameters[bias_offset + row] as i64) * 256;
    let start = weight_offset + row * inputs.len();
    for (column, input) in inputs.iter().enumerate() {
        accumulator =
            accumulator.saturating_add((model.parameters[start + column] as i64) * (*input as i64));
    }
    round_q16_16_to_q8_8(accumulator)
}

pub fn forward(
    model: &PackedModel,
    input_ids: &[u8; NUM_CELLS],
    logical_len: usize,
    scratch: &mut Scratch,
    predictions: &mut [u8; NUM_CELLS],
    logits_out: Option<&mut [[i32; VOCAB_SIZE]; MAX_SEQ_LEN]>,
) -> Result<(), ()> {
    if logical_len > MAX_SEQ_LEN || input_ids.iter().any(|id| *id as usize >= VOCAB_SIZE) {
        return Err(());
    }
    scratch.state_a.fill([0; HIDDEN_DIM]);
    scratch.state_b.fill([0; HIDDEN_DIM]);
    for iteration in 0..ITERATIONS {
        let homogeneous_start = logical_len
            .saturating_add(RADIUS * iteration)
            .min(NUM_CELLS);
        let homogeneous_end = NUM_CELLS.saturating_sub(RADIUS * iteration);
        let mut homogeneous_value: Option<[i16; HIDDEN_DIM]> = None;
        for cell in 0..NUM_CELLS {
            if cell >= homogeneous_start && cell < homogeneous_end {
                if let Some(value) = homogeneous_value {
                    scratch.state_b[cell] = value;
                    continue;
                }
            }
            let mut input = [0i16; UPDATE_INPUT_DIM];
            let mut cursor = 0;
            for relative in -(RADIUS as isize)..=(RADIUS as isize) {
                let neighbor = cell as isize + relative;
                if (0..NUM_CELLS as isize).contains(&neighbor) {
                    input[cursor..cursor + HIDDEN_DIM]
                        .copy_from_slice(&scratch.state_a[neighbor as usize]);
                }
                cursor += HIDDEN_DIM;
            }
            let embedding_start = EMBEDDING_OFFSET + input_ids[cell] as usize * EMBEDDING_DIM;
            input[cursor..cursor + EMBEDDING_DIM].copy_from_slice(
                &model.parameters[embedding_start..embedding_start + EMBEDDING_DIM],
            );
            let mut hidden = [0i16; MLP_WIDTH];
            for row in 0..MLP_WIDTH {
                hidden[row] = linear(
                    model,
                    UPDATE_IN_WEIGHT_OFFSET,
                    UPDATE_IN_BIAS_OFFSET,
                    &input,
                    row,
                )
                .max(0)
                .min(i16::MAX as i32) as i16;
            }
            for row in 0..HIDDEN_DIM {
                let delta = linear(
                    model,
                    UPDATE_OUT_WEIGHT_OFFSET,
                    UPDATE_OUT_BIAS_OFFSET,
                    &hidden,
                    row,
                );
                scratch.state_b[cell][row] = clamp_state(scratch.state_a[cell][row] as i32 + delta);
            }
            if cell >= homogeneous_start && cell < homogeneous_end {
                homogeneous_value = Some(scratch.state_b[cell]);
            }
        }
        core::mem::swap(&mut scratch.state_a, &mut scratch.state_b);
    }
    let mut logits_out = logits_out;
    predictions.fill(0);
    for cell in 0..logical_len {
        let mut best_id = 0u8;
        let mut best_logit = i32::MIN;
        for row in 0..VOCAB_SIZE {
            let logit = linear(
                model,
                OUTPUT_WEIGHT_OFFSET,
                OUTPUT_BIAS_OFFSET,
                &scratch.state_a[cell],
                row,
            );
            if cell < logical_len {
                if let Some(ref mut logits) = logits_out {
                    logits[cell][row] = logit;
                }
            }
            if logit > best_logit {
                best_logit = logit;
                best_id = row as u8;
            }
        }
        predictions[cell] = best_id;
    }
    Ok(())
}

pub fn model_hash(model: &PackedModel) -> [u8; 32] {
    let mut state = Params::new().hash_length(32).to_state();
    state.update(b"mini-cells:model:v1");
    let mut chunk = [0u8; 128];
    for values in model.parameters.chunks(64) {
        for (index, value) in values.iter().enumerate() {
            chunk[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
        }
        state.update(&chunk[..values.len() * 2]);
    }
    let hash = state.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(hash.as_bytes());
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_is_exact() {
        assert_eq!(OUTPUT_BIAS_OFFSET + 44, PARAMETER_COUNT);
        assert_eq!(MODEL_BYTES, 8952);
    }
    #[test]
    fn codec_and_hash_are_deterministic() {
        let mut model = PackedModel::default();
        model.parameters[0] = -16;
        model.parameters[4475] = 16;
        let mut bytes = [0u8; MODEL_BYTES];
        assert_eq!(model.encode_into(&mut bytes), Ok(MODEL_BYTES));
        assert_eq!(PackedModel::decode_from(&bytes).unwrap(), model);
        assert_eq!(model_hash(&model), model_hash(&model));
    }
    #[test]
    fn forward_is_deterministic() {
        let model = PackedModel::default();
        let input = [0u8; 64];
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        forward(&model, &input, 1, &mut Scratch::new(), &mut a, None).unwrap();
        forward(&model, &input, 1, &mut Scratch::new(), &mut b, None).unwrap();
        assert_eq!(a, b);
    }
}
