use blake2b_simd::Params;

use crate::{
    model::{forward, PackedModel, Scratch, MAX_SEQ_LEN, NUM_CELLS, VOCAB_SIZE},
    vocab::{encode_byte, SYMBOLS},
};

pub const MAX_BATCH: usize = 4;
const WORDS: [&[u8]; 10] = [
    b"mini", b"cells", b"jam", b"hello", b"world", b"learn", b"echo", b"small", b"local", b"neural",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EchoBatch {
    pub ids: [[u8; NUM_CELLS]; MAX_BATCH],
    pub lengths: [u8; MAX_BATCH],
    pub size: u8,
}
impl Default for EchoBatch {
    fn default() -> Self {
        Self {
            ids: [[0; NUM_CELLS]; MAX_BATCH],
            lengths: [0; MAX_BATCH],
            size: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Evaluation {
    pub loss: i64,
    pub correct_tokens: u32,
    pub total_tokens: u32,
    pub digest: [u8; 32],
}

#[derive(Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn range(&mut self, upper: usize) -> usize {
        (self.next() % upper as u64) as usize
    }
}

fn domain_seed(domain: &[u8], parent_hash: &[u8; 32], generation: u64) -> u64 {
    let mut state = Params::new().hash_length(32).to_state();
    state.update(domain);
    state.update(parent_hash);
    state.update(&generation.to_le_bytes());
    let hash = state.finalize();
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap())
}

pub fn canonical_batch(parent_hash: &[u8; 32], generation: u64, size: u8) -> Result<EchoBatch, ()> {
    if size == 0 || size as usize > MAX_BATCH {
        return Err(());
    }
    let mut rng = SplitMix64::new(domain_seed(b"mini-cells:batch:v1", parent_hash, generation));
    let mut batch = EchoBatch::default();
    batch.size = size;
    for sample in 0..size as usize {
        let length = 1 + rng.range(MAX_SEQ_LEN);
        batch.lengths[sample] = length as u8;
        if rng.range(10) < 7 {
            for pos in 0..length {
                batch.ids[sample][pos] = (1 + rng.range(SYMBOLS.len())) as u8;
            }
        } else {
            let mut pos = 0;
            while pos < length {
                if pos > 0 {
                    batch.ids[sample][pos] = encode_byte(b' ').unwrap();
                    pos += 1;
                    if pos >= length {
                        break;
                    }
                }
                let word = WORDS[rng.range(WORDS.len())];
                for value in word {
                    if pos >= length {
                        break;
                    }
                    batch.ids[sample][pos] = encode_byte(*value).unwrap();
                    pos += 1;
                }
            }
        }
    }
    Ok(batch)
}

pub fn batch_digest(batch: &EchoBatch) -> [u8; 32] {
    let mut state = Params::new().hash_length(32).to_state();
    state.update(b"mini-cells:batch-digest:v1");
    state.update(&[batch.size]);
    for index in 0..batch.size as usize {
        state.update(&[batch.lengths[index]]);
        state.update(&batch.ids[index][..batch.lengths[index] as usize]);
    }
    let hash = state.finalize();
    let mut out = [0; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}

pub fn evaluate_batch(
    model: &PackedModel,
    batch: &EchoBatch,
    margin: i32,
    scratch: &mut Scratch,
) -> Result<Evaluation, ()> {
    let mut predictions = [0; NUM_CELLS];
    let mut logits = [[0; VOCAB_SIZE]; MAX_SEQ_LEN];
    evaluate_batch_with_buffers(model, batch, margin, scratch, &mut predictions, &mut logits)
}

pub fn evaluate_batch_with_buffers(
    model: &PackedModel,
    batch: &EchoBatch,
    margin: i32,
    scratch: &mut Scratch,
    predictions: &mut [u8; NUM_CELLS],
    logits: &mut [[i32; VOCAB_SIZE]; MAX_SEQ_LEN],
) -> Result<Evaluation, ()> {
    let mut result = Evaluation::default();
    let mut digest = Params::new().hash_length(32).to_state();
    digest.update(b"mini-cells:eval:v1");
    for sample in 0..batch.size as usize {
        let length = batch.lengths[sample] as usize;
        predictions.fill(0);
        logits.fill([0; VOCAB_SIZE]);
        forward(
            model,
            &batch.ids[sample],
            length,
            scratch,
            predictions,
            Some(logits),
        )?;
        for position in 0..length {
            let target = batch.ids[sample][position] as usize;
            let target_logit = logits[position][target];
            let mut other = i32::MIN;
            for (id, value) in logits[position].iter().enumerate() {
                if id != target {
                    other = other.max(*value);
                }
            }
            result.loss = result
                .loss
                .saturating_add((margin - (target_logit - other)).max(0) as i64);
            result.total_tokens += 1;
            if predictions[position] as usize == target {
                result.correct_tokens += 1;
            }
        }
        digest.update(&predictions[..length]);
    }
    result.digest.copy_from_slice(digest.finalize().as_bytes());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prng_and_batch_are_deterministic() {
        let h = [7; 32];
        let a = canonical_batch(&h, 0, 4).unwrap();
        assert_eq!(a, canonical_batch(&h, 0, 4).unwrap());
        assert_ne!(a, canonical_batch(&h, 1, 4).unwrap());
        let mut r = SplitMix64::new(1);
        assert_eq!(r.next(), 10451216379200822465);
    }
    #[test]
    fn margin_loss_is_bounded() {
        let b = canonical_batch(&[0; 32], 0, 1).unwrap();
        let e = evaluate_batch(&PackedModel::default(), &b, 256, &mut Scratch::new()).unwrap();
        assert_eq!(e.loss, e.total_tokens as i64 * 256);
    }
}
