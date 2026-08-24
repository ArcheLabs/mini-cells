use crate::{ensure_initialized, Host, HostError};
use minicells_core::{
    model::{model_hash, PackedModel, MODEL_BYTES},
    optimizer::apply_update,
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
    apply_pair(host, meta, plus, minus, workspace)
}

#[inline(never)]
fn apply_pair<H: Host>(
    host: &mut H,
    meta: &mut MetaV1,
    plus: PendingV1,
    minus: PendingV1,
    workspace: &mut AccumulateWorkspace,
) -> Result<(), HostError> {
    read_model(host, &mut workspace.model, &mut workspace.model_bytes)?;
    let parent = meta.model_hash;
    let updated = apply_update(
        &mut workspace.model,
        &parent,
        meta.generation,
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
    meta.current_eval_loss = plus.loss.min(minus.loss);
    meta.current_correct = if plus.loss <= minus.loss {
        plus.correct
    } else {
        minus.correct
    };
    meta.current_tokens = plus.tokens;
    meta.successful_updates = meta.successful_updates.saturating_add(updated as u64);
    meta.zero_diff_updates = meta.zero_diff_updates.saturating_add((!updated) as u64);
    let history = HistoryV1 {
        generation: meta.generation,
        parent_hash: parent,
        model_hash: next_hash,
        plus_loss: plus.loss,
        minus_loss: minus.loss,
        plus_correct: plus.correct,
        minus_correct: minus.correct,
        tokens: plus.tokens,
        updated: updated as u8,
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
                loss,
                correct_tokens,
                total_tokens,
                eval_digest,
            } => handle_training(
                host,
                &mut meta,
                &result,
                side,
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
