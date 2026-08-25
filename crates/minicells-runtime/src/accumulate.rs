use crate::{ensure_initialized, Host, HostError};
use minicells_core::{
    model::{model_hash, PackedModel, MODEL_BYTES},
    optimizer::{apply_update, UpdateDecision},
};
use minicells_protocol::{
    keys::{self, inference_key, inference_slot},
    HistoryV1, InferenceV1, MetaV1, PendingV1, RefineResult, ResultBody, META_ENCODED_LEN,
    STATUS_OK,
};

const MAX_RESULT_BYTES: usize = 160;
pub struct AccumulateWorkspace {
    model: PackedModel,
    model_bytes: [u8; MODEL_BYTES],
}
impl AccumulateWorkspace {
    pub const fn new() -> Self {
        Self {
            model: PackedModel::from_parameters([0; minicells_core::model::PARAMETER_COUNT]),
            model_bytes: [0; MODEL_BYTES],
        }
    }
}
impl Default for AccumulateWorkspace {
    fn default() -> Self {
        Self::new()
    }
}
fn write_meta<H: Host>(host: &mut H, meta: &MetaV1) -> Result<(), HostError> {
    let mut b = [0u8; META_ENCODED_LEN];
    let n = meta.encode_into(&mut b).map_err(|_| HostError::Failure)?;
    host.storage_write(keys::META, &b[..n])
}
fn read_pending<H: Host>(host: &H, key: &[u8]) -> Result<Option<PendingV1>, HostError> {
    let mut b = [0u8; PendingV1::LEN];
    match host.storage_read(key, &mut b)? {
        Some(n) => PendingV1::decode(&b[..n])
            .map(Some)
            .map_err(|_| HostError::Failure),
        None => Ok(None),
    }
}
fn write_pending<H: Host>(host: &mut H, key: &[u8], p: &PendingV1) -> Result<(), HostError> {
    let mut b = [0u8; PendingV1::LEN];
    let n = p.encode_into(&mut b).map_err(|_| HostError::Failure)?;
    host.storage_write(key, &b[..n])
}
fn read_model<H: Host>(
    host: &H,
    model: &mut PackedModel,
    b: &mut [u8; MODEL_BYTES],
) -> Result<(), HostError> {
    b.fill(0);
    let n = host
        .storage_read(keys::MODEL, b)?
        .ok_or(HostError::Missing)?;
    model.decode_into(&b[..n]).map_err(|_| HostError::Failure)
}
fn persist_inference<H: Host>(
    host: &mut H,
    result: &RefineResult,
    input_len: u8,
    output_len: u8,
    input: [u8; 32],
    output: [u8; 32],
    matching_tokens: u8,
) -> Result<(), HostError> {
    let record = InferenceV1 {
        request_id: result.request_id,
        generation: result.generation,
        model_hash: result.model_hash,
        input_len,
        output_len,
        input,
        output,
        matching_tokens,
    };
    let mut b = [0u8; InferenceV1::LEN];
    let n = record.encode_into(&mut b).map_err(|_| HostError::Failure)?;
    host.storage_write(&inference_key(inference_slot(result.request_id)), &b[..n])
}

#[inline(never)]
fn handle_training<H: Host>(
    host: &mut H,
    meta: &mut MetaV1,
    result: &RefineResult,
    side: i8,
    base_loss: i64,
    base_correct_tokens: u32,
    loss: i64,
    correct_tokens: u32,
    total_tokens: u32,
    eval_digest: [u8; 32],
    workspace: &mut AccumulateWorkspace,
) -> Result<(), HostError> {
    if result.generation != meta.generation || result.model_hash != meta.model_hash {
        meta.stale_results = meta.stale_results.saturating_add(1);
        return Ok(());
    }
    let key = if side == 1 {
        keys::PENDING_PLUS
    } else if side == -1 {
        keys::PENDING_MINUS
    } else {
        return Ok(());
    };
    if read_pending(host, key)?.is_some() {
        meta.duplicate_results = meta.duplicate_results.saturating_add(1);
        return Ok(());
    }
    write_pending(
        host,
        key,
        &PendingV1 {
            generation: result.generation,
            parent_hash: result.model_hash,
            side,
            loss,
            correct: correct_tokens,
            tokens: total_tokens,
            digest: eval_digest,
        },
    )?;
    let (Some(plus), Some(minus)) = (
        read_pending(host, keys::PENDING_PLUS)?,
        read_pending(host, keys::PENDING_MINUS)?,
    ) else {
        return Ok(());
    };
    apply_pair(
        host,
        meta,
        plus,
        minus,
        base_loss,
        base_correct_tokens,
        total_tokens,
        workspace,
    )
}

#[inline(never)]
fn apply_pair<H: Host>(
    host: &mut H,
    meta: &mut MetaV1,
    plus: PendingV1,
    minus: PendingV1,
    base_loss: i64,
    base_correct_tokens: u32,
    current_result_tokens: u32,
    workspace: &mut AccumulateWorkspace,
) -> Result<(), HostError> {
    if plus.generation != meta.generation
        || minus.generation != meta.generation
        || plus.parent_hash != meta.model_hash
        || minus.parent_hash != meta.model_hash
        || plus.tokens != minus.tokens
        || current_result_tokens != plus.tokens
    {
        return Err(HostError::Failure);
    }
    read_model(host, &mut workspace.model, &mut workspace.model_bytes)?;
    let parent = meta.model_hash;
    let decision = apply_update(
        &mut workspace.model,
        &parent,
        meta.generation,
        base_loss,
        plus.loss,
        minus.loss,
        meta.update_step_q,
    );
    let next_hash = model_hash(&workspace.model);
    workspace
        .model
        .encode_into(&mut workspace.model_bytes)
        .map_err(|_| HostError::Failure)?;
    host.storage_write(keys::MODEL, &workspace.model_bytes)?;
    meta.generation = meta.generation.saturating_add(1);
    meta.model_hash = next_hash;
    let (retained_loss, retained_correct) = match decision {
        UpdateDecision::Plus => (plus.loss, plus.correct),
        UpdateDecision::Minus => (minus.loss, minus.correct),
        UpdateDecision::Keep => (base_loss, base_correct_tokens),
    };
    meta.current_eval_loss = retained_loss;
    meta.current_correct = retained_correct;
    meta.current_tokens = plus.tokens;
    meta.successful_updates = meta
        .successful_updates
        .saturating_add((decision != UpdateDecision::Keep) as u64);
    meta.zero_diff_updates = meta
        .zero_diff_updates
        .saturating_add((decision == UpdateDecision::Keep) as u64);
    let history = HistoryV1 {
        generation: meta.generation,
        parent_hash: parent,
        model_hash: next_hash,
        plus_loss: plus.loss,
        minus_loss: minus.loss,
        plus_correct: plus.correct,
        minus_correct: minus.correct,
        tokens: plus.tokens,
        updated: (decision != UpdateDecision::Keep) as u8,
    };
    let mut b = [0u8; HistoryV1::LEN];
    let n = history
        .encode_into(&mut b)
        .map_err(|_| HostError::Failure)?;
    host.storage_write(&keys::history_key(meta.history_head), &b[..n])?;
    meta.history_head = (meta.history_head + 1) % 64;
    host.storage_delete(keys::PENDING_PLUS)?;
    host.storage_delete(keys::PENDING_MINUS)?;
    Ok(())
}

pub fn accumulate<H: Host>(
    host: &mut H,
    workspace: &mut AccumulateWorkspace,
) -> Result<(), HostError> {
    let mut meta = ensure_initialized(host)?;
    let count = host.result_count();
    let mut result_bytes = [0u8; MAX_RESULT_BYTES];
    for index in 0..count {
        let size = match host.result(index, &mut result_bytes) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let result = match RefineResult::decode(&result_bytes[..size]) {
            Ok(x) if x.status == STATUS_OK => x,
            _ => continue,
        };
        match result.body.clone() {
            ResultBody::Inference {
                input_len,
                output_len,
                input,
                output,
                matching_tokens,
            } => {
                if result.generation == meta.generation && result.model_hash == meta.model_hash {
                    persist_inference(
                        host,
                        &result,
                        input_len,
                        output_len,
                        input,
                        output,
                        matching_tokens,
                    )?;
                } else {
                    meta.stale_results = meta.stale_results.saturating_add(1);
                }
            }
            ResultBody::Training {
                side,
                base_loss,
                base_correct_tokens,
                base_eval_digest: _,
                loss,
                correct_tokens,
                total_tokens,
                eval_digest,
            } => handle_training(
                host,
                &mut meta,
                &result,
                side,
                base_loss,
                base_correct_tokens,
                loss,
                correct_tokens,
                total_tokens,
                eval_digest,
                workspace,
            )?,
            ResultBody::Status => {}
        }
    }
    write_meta(host, &meta)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use minicells_protocol::{Op, ResultBody, STATUS_OK};
    use std::{collections::BTreeMap, vec};

    #[derive(Default)]
    struct TestHost {
        storage: BTreeMap<std::vec::Vec<u8>, std::vec::Vec<u8>>,
        results: std::vec::Vec<std::vec::Vec<u8>>,
    }
    impl Host for TestHost {
        fn payload(&self, _: &mut [u8]) -> Result<usize, HostError> {
            Err(HostError::Missing)
        }
        fn result_count(&self) -> usize {
            self.results.len()
        }
        fn result(&self, index: usize, output: &mut [u8]) -> Result<usize, HostError> {
            let value = self.results.get(index).ok_or(HostError::Missing)?;
            if output.len() < value.len() {
                return Err(HostError::BufferTooSmall);
            }
            output[..value.len()].copy_from_slice(value);
            Ok(value.len())
        }
        fn storage_read(&self, key: &[u8], output: &mut [u8]) -> Result<Option<usize>, HostError> {
            let Some(value) = self.storage.get(key) else {
                return Ok(None);
            };
            if output.len() < value.len() {
                return Err(HostError::BufferTooSmall);
            }
            output[..value.len()].copy_from_slice(value);
            Ok(Some(value.len()))
        }
        fn storage_write(&mut self, key: &[u8], value: &[u8]) -> Result<(), HostError> {
            self.storage.insert(key.to_vec(), value.to_vec());
            Ok(())
        }
        fn storage_delete(&mut self, key: &[u8]) -> Result<(), HostError> {
            self.storage.remove(key);
            Ok(())
        }
        fn yield_value(&mut self, _: &[u8]) {}
    }

    fn result(op: Op, side: i8, base_loss: i64, loss: i64) -> std::vec::Vec<u8> {
        let value = RefineResult {
            op,
            status: STATUS_OK,
            request_id: if side == 1 { 1 } else { 2 },
            generation: 0,
            model_hash: crate::genesis::GENESIS_MODEL_HASH,
            body: ResultBody::Training {
                side,
                base_loss,
                base_correct_tokens: 7,
                base_eval_digest: [1; 32],
                loss,
                correct_tokens: 8,
                total_tokens: 16,
                eval_digest: [2; 32],
            },
        };
        let mut out = [0; 160];
        let n = value.encode_into(&mut out).unwrap();
        out[..n].to_vec()
    }

    #[test]
    fn accepted_pair_changes_model_and_records_candidate() {
        let mut host = TestHost::default();
        ensure_initialized(&mut host).unwrap();
        let parent = host.storage.get(keys::MODEL).unwrap().clone();
        host.results = vec![result(Op::TrainPlus, 1, 100, 80)];
        accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
        host.results = vec![result(Op::TrainMinus, -1, 100, 120)];
        accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
        let meta = MetaV1::decode(host.storage.get(keys::META).unwrap()).unwrap();
        assert_eq!(meta.generation, 1);
        assert_ne!(host.storage.get(keys::MODEL).unwrap(), &parent);
        assert_eq!(meta.successful_updates, 1);
        assert_eq!(meta.current_eval_loss, 80);
        assert_eq!(host.storage.get(keys::PENDING_PLUS), None);
        assert_eq!(host.storage.get(keys::PENDING_MINUS), None);
        let history =
            HistoryV1::decode(host.storage.get(keys::history_key(0).as_slice()).unwrap()).unwrap();
        assert_eq!(history.updated, 1);
    }

    #[test]
    fn rejected_pair_keeps_model_and_records_base() {
        let mut host = TestHost::default();
        ensure_initialized(&mut host).unwrap();
        let parent = host.storage.get(keys::MODEL).unwrap().clone();
        host.results = vec![result(Op::TrainPlus, 1, 100, 110)];
        accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
        host.results = vec![result(Op::TrainMinus, -1, 100, 120)];
        accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
        let meta = MetaV1::decode(host.storage.get(keys::META).unwrap()).unwrap();
        assert_eq!(meta.generation, 1);
        assert_eq!(host.storage.get(keys::MODEL).unwrap(), &parent);
        assert_eq!(meta.successful_updates, 0);
        assert_eq!(meta.zero_diff_updates, 1);
        assert_eq!(meta.current_eval_loss, 100);
        let history =
            HistoryV1::decode(host.storage.get(keys::history_key(0).as_slice()).unwrap()).unwrap();
        assert_eq!(history.updated, 0);
    }
}
