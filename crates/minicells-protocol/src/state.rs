use crate::{
    codec::{Reader, Writer},
    runtime_config, Error,
};
pub const META_MAGIC: [u8; 4] = *b"MCV1";
pub const META_ENCODED_LEN: usize = 128;
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetaV1 {
    pub format_version: u16,
    pub model_version: u16,
    pub optimizer_version: u16,
    pub capability_id: u16,
    pub generation: u64,
    pub model_hash: [u8; 32],
    pub current_eval_loss: i64,
    pub current_correct: u32,
    pub current_tokens: u32,
    pub perturbation_q: i16,
    pub update_step_q: i16,
    pub train_batch_size: u16,
    pub margin_q: u16,
    pub genesis_seed: u64,
    pub batch_seed_domain: u64,
    pub history_head: u8,
    pub initialized: u8,
    pub paused: u8,
    pub modality_id: u8,
    pub successful_updates: u64,
    pub zero_diff_updates: u64,
    pub stale_results: u64,
    pub duplicate_results: u64,
}
impl MetaV1 {
    pub fn new(hash: [u8; 32]) -> Self {
        Self {
            format_version: 1,
            model_version: runtime_config::MODEL_FORMAT as u16,
            optimizer_version: runtime_config::OPTIMIZER_VERSION as u16,
            capability_id: runtime_config::CAPABILITY_ID as u16,
            generation: 0,
            model_hash: hash,
            current_eval_loss: 0,
            current_correct: 0,
            current_tokens: 0,
            perturbation_q: runtime_config::PERTURBATION_Q as i16,
            update_step_q: runtime_config::UPDATE_STEP_Q as i16,
            train_batch_size: runtime_config::TRAIN_BATCH_SIZE as u16,
            margin_q: runtime_config::MARGIN_Q as u16,
            genesis_seed: runtime_config::GENESIS_SEED as u64,
            batch_seed_domain: runtime_config::BATCH_SEED_DOMAIN as u64,
            history_head: 0,
            initialized: 1,
            paused: 0,
            modality_id: runtime_config::MODALITY_ID as u8,
            successful_updates: 0,
            zero_diff_updates: 0,
            stale_results: 0,
            duplicate_results: 0,
        }
    }
    pub fn encode_into(&self, o: &mut [u8]) -> Result<usize, Error> {
        let mut w = Writer::new(o);
        w.bytes(&META_MAGIC)?;
        w.u16(self.format_version)?;
        w.u16(self.model_version)?;
        w.u16(self.optimizer_version)?;
        w.u16(self.capability_id)?;
        w.u64(self.generation)?;
        w.bytes(&self.model_hash)?;
        w.i64(self.current_eval_loss)?;
        w.u32(self.current_correct)?;
        w.u32(self.current_tokens)?;
        w.i16(self.perturbation_q)?;
        w.i16(self.update_step_q)?;
        w.u16(self.train_batch_size)?;
        w.u16(self.margin_q)?;
        w.u64(self.genesis_seed)?;
        w.u64(self.batch_seed_domain)?;
        w.u8(self.history_head)?;
        w.u8(self.initialized)?;
        w.u8(self.paused)?;
        w.u8(self.modality_id)?;
        w.u64(self.successful_updates)?;
        w.u64(self.zero_diff_updates)?;
        w.u64(self.stale_results)?;
        w.u64(self.duplicate_results)?;
        Ok(w.len())
    }
    pub fn decode(i: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(i);
        if r.bytes::<4>()? != META_MAGIC {
            return Err(Error::Invalid);
        }
        let value = Self {
            format_version: r.u16()?,
            model_version: r.u16()?,
            optimizer_version: r.u16()?,
            capability_id: r.u16()?,
            generation: r.u64()?,
            model_hash: r.bytes()?,
            current_eval_loss: r.i64()?,
            current_correct: r.u32()?,
            current_tokens: r.u32()?,
            perturbation_q: r.i16()?,
            update_step_q: r.i16()?,
            train_batch_size: r.u16()?,
            margin_q: r.u16()?,
            genesis_seed: r.u64()?,
            batch_seed_domain: r.u64()?,
            history_head: r.u8()?,
            initialized: r.u8()?,
            paused: r.u8()?,
            modality_id: r.u8()?,
            successful_updates: r.u64()?,
            zero_diff_updates: r.u64()?,
            stale_results: r.u64()?,
            duplicate_results: r.u64()?,
        };
        if !r.done() || value.format_version != 1 {
            return Err(Error::Invalid);
        }
        Ok(value)
    }
}

pub fn validate_canonical_training_config(meta: &MetaV1) -> Result<(), Error> {
    if meta.model_version != runtime_config::MODEL_FORMAT as u16
        || meta.optimizer_version != runtime_config::OPTIMIZER_VERSION as u16
        || meta.capability_id != runtime_config::CAPABILITY_ID as u16
        || meta.modality_id != runtime_config::MODALITY_ID as u8
        || meta.margin_q != runtime_config::MARGIN_Q as u16
        || meta.perturbation_q != runtime_config::PERTURBATION_Q as i16
        || meta.update_step_q != runtime_config::UPDATE_STEP_Q as i16
        || meta.train_batch_size != runtime_config::TRAIN_BATCH_SIZE as u16
        || meta.genesis_seed != runtime_config::GENESIS_SEED as u64
        || meta.batch_seed_domain != runtime_config::BATCH_SEED_DOMAIN as u64
    {
        return Err(Error::Invalid);
    }
    Ok(())
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingV1 {
    pub generation: u64,
    pub parent_hash: [u8; 32],
    pub side: i8,
    pub loss: i64,
    pub correct: u32,
    pub tokens: u32,
    pub digest: [u8; 32],
}
impl PendingV1 {
    pub const LEN: usize = 89;
    pub fn encode_into(&self, o: &mut [u8]) -> Result<usize, Error> {
        let mut w = Writer::new(o);
        w.u64(self.generation)?;
        w.bytes(&self.parent_hash)?;
        w.i8(self.side)?;
        w.i64(self.loss)?;
        w.u32(self.correct)?;
        w.u32(self.tokens)?;
        w.bytes(&self.digest)?;
        Ok(w.len())
    }
    pub fn decode(i: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(i);
        let x = Self {
            generation: r.u64()?,
            parent_hash: r.bytes()?,
            side: r.i8()?,
            loss: r.i64()?,
            correct: r.u32()?,
            tokens: r.u32()?,
            digest: r.bytes()?,
        };
        if !r.done() || ![-1, 1].contains(&x.side) {
            Err(Error::Invalid)
        } else {
            Ok(x)
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HistoryV1 {
    pub generation: u64,
    pub parent_hash: [u8; 32],
    pub model_hash: [u8; 32],
    pub plus_loss: i64,
    pub minus_loss: i64,
    pub plus_correct: u32,
    pub minus_correct: u32,
    pub tokens: u32,
    pub updated: u8,
}
impl HistoryV1 {
    pub const LEN: usize = 105;
    pub fn encode_into(&self, o: &mut [u8]) -> Result<usize, Error> {
        let mut w = Writer::new(o);
        w.u64(self.generation)?;
        w.bytes(&self.parent_hash)?;
        w.bytes(&self.model_hash)?;
        w.i64(self.plus_loss)?;
        w.i64(self.minus_loss)?;
        w.u32(self.plus_correct)?;
        w.u32(self.minus_correct)?;
        w.u32(self.tokens)?;
        w.u8(self.updated)?;
        Ok(w.len())
    }
    pub fn decode(i: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(i);
        let x = Self {
            generation: r.u64()?,
            parent_hash: r.bytes()?,
            model_hash: r.bytes()?,
            plus_loss: r.i64()?,
            minus_loss: r.i64()?,
            plus_correct: r.u32()?,
            minus_correct: r.u32()?,
            tokens: r.u32()?,
            updated: r.u8()?,
        };
        if r.done() {
            Ok(x)
        } else {
            Err(Error::Invalid)
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InferenceV1 {
    pub request_id: u64,
    pub generation: u64,
    pub model_hash: [u8; 32],
    pub input_len: u8,
    pub output_len: u8,
    pub input: [u8; 32],
    pub output: [u8; 32],
    pub matching_tokens: u8,
}
impl InferenceV1 {
    pub const LEN: usize = 115;
    pub fn encode_into(&self, o: &mut [u8]) -> Result<usize, Error> {
        let mut w = Writer::new(o);
        w.u64(self.request_id)?;
        w.u64(self.generation)?;
        w.bytes(&self.model_hash)?;
        w.u8(self.input_len)?;
        w.u8(self.output_len)?;
        w.bytes(&self.input)?;
        w.bytes(&self.output)?;
        w.u8(self.matching_tokens)?;
        Ok(w.len())
    }
    pub fn decode(i: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(i);
        let x = Self {
            request_id: r.u64()?,
            generation: r.u64()?,
            model_hash: r.bytes()?,
            input_len: r.u8()?,
            output_len: r.u8()?,
            input: r.bytes()?,
            output: r.bytes()?,
            matching_tokens: r.u8()?,
        };
        if r.done() && x.input_len <= 32 && x.output_len <= 32 {
            Ok(x)
        } else {
            Err(Error::Invalid)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn meta_codec() {
        let x = MetaV1::new([9; 32]);
        let mut b = [0; META_ENCODED_LEN];
        let n = x.encode_into(&mut b).unwrap();
        assert_eq!(n, META_ENCODED_LEN);
        assert_eq!(MetaV1::decode(&b), Ok(x));
    }

    #[test]
    fn forged_training_configuration_is_rejected() {
        let mut x = MetaV1::new([9; 32]);
        assert!(validate_canonical_training_config(&x).is_ok());
        x.margin_q = x.margin_q.saturating_add(1);
        assert!(validate_canonical_training_config(&x).is_err());
    }
}
