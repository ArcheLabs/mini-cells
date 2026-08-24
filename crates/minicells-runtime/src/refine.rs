use crate::{
    genesis::{read_meta_or_genesis, read_model_into_or_genesis},
    Host, HostError,
};
use minicells_core::{
    batch::{canonical_batch, evaluate_batch_with_buffers},
    model::{forward, model_hash, PackedModel, MAX_SEQ_LEN, MODEL_BYTES, NUM_CELLS, VOCAB_SIZE},
    optimizer::candidate_into,
    vocab::{decode_id, encode_text},
    Scratch,
};
use minicells_protocol::{
    Op, RefineResult, ResultBody, WorkBody, WorkPayload, STATUS_OK, USE_CURRENT_GENERATION,
};

pub const MAX_WORK_BYTES: usize = 96;
pub const MAX_RESULT_BYTES: usize = 160;

pub struct RefineWorkspace {
    model: PackedModel,
    evaluated_model: PackedModel,
    scratch: Scratch,
    model_bytes: [u8; MODEL_BYTES],
    predictions: [u8; NUM_CELLS],
    logits: [[i32; VOCAB_SIZE]; MAX_SEQ_LEN],
}

impl RefineWorkspace {
    pub const fn new() -> Self {
        Self {
            model: PackedModel::from_parameters([0; minicells_core::model::PARAMETER_COUNT]),
            evaluated_model: PackedModel::from_parameters(
                [0; minicells_core::model::PARAMETER_COUNT],
            ),
            scratch: Scratch::new(),
            model_bytes: [0; MODEL_BYTES],
            predictions: [0; NUM_CELLS],
            logits: [[0; VOCAB_SIZE]; MAX_SEQ_LEN],
        }
    }
}

impl Default for RefineWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

pub fn refine<H: Host>(
    host: &mut H,
    workspace: &mut RefineWorkspace,
    output: &mut [u8],
) -> Result<usize, HostError> {
    let mut payload = [0u8; MAX_WORK_BYTES];
    let size = host.payload(&mut payload)?;
    let work = WorkPayload::decode(&payload[..size]).map_err(|_| HostError::Failure)?;
    let meta = read_meta_or_genesis(host)?;
    let result = match work.body {
        WorkBody::Infer {
            expected_generation,
            text_len,
            text,
        } => refine_inference(
            host,
            workspace,
            work.request_id,
            meta,
            expected_generation,
            text_len,
            text,
        )?,
        WorkBody::Train {
            generation,
            parent_model_hash,
        } => refine_training(
            host,
            workspace,
            work.op,
            work.request_id,
            meta,
            generation,
            parent_model_hash,
        )?,
        WorkBody::StatusProbe => RefineResult {
            op: Op::StatusProbe,
            status: STATUS_OK,
            request_id: work.request_id,
            generation: meta.generation,
            model_hash: meta.model_hash,
            body: ResultBody::Status,
        },
    };
    result
        .encode_into(output)
        .map_err(|_| HostError::BufferTooSmall)
}

#[inline(never)]
fn refine_inference<H: Host>(
    host: &H,
    workspace: &mut RefineWorkspace,
    request_id: u64,
    meta: minicells_protocol::MetaV1,
    expected_generation: u64,
    text_len: u8,
    text: [u8; 32],
) -> Result<RefineResult, HostError> {
    if expected_generation != USE_CURRENT_GENERATION && expected_generation != meta.generation {
        return Err(HostError::Failure);
    }
    read_model_into_or_genesis(host, &mut workspace.model, &mut workspace.model_bytes)?;
    if model_hash(&workspace.model) != meta.model_hash {
        return Err(HostError::Failure);
    }
    let mut ids = [0u8; NUM_CELLS];
    encode_text(&text[..text_len as usize], &mut ids).map_err(|_| HostError::Failure)?;
    let mut predictions = [0u8; NUM_CELLS];
    forward(
        &workspace.model,
        &ids,
        text_len as usize,
        &mut workspace.scratch,
        &mut predictions,
        None,
    )
    .map_err(|_| HostError::Failure)?;
    let mut rendered = [0u8; 32];
    let mut matching = 0;
    for i in 0..text_len as usize {
        rendered[i] = decode_id(predictions[i]).unwrap_or(0);
        if rendered[i] == text[i] {
            matching += 1
        }
    }
    Ok(RefineResult {
        op: Op::Infer,
        status: STATUS_OK,
        request_id,
        generation: meta.generation,
        model_hash: meta.model_hash,
        body: ResultBody::Inference {
            input_len: text_len,
            output_len: text_len,
            input: text,
            output: rendered,
            matching_tokens: matching,
        },
    })
}

#[inline(never)]
fn refine_training<H: Host>(
    host: &H,
    workspace: &mut RefineWorkspace,
    op: Op,
    request_id: u64,
    meta: minicells_protocol::MetaV1,
    generation: u64,
    parent_model_hash: [u8; 32],
) -> Result<RefineResult, HostError> {
    read_model_into_or_genesis(host, &mut workspace.model, &mut workspace.model_bytes)?;
    if model_hash(&workspace.model) != meta.model_hash {
        return Err(HostError::Failure);
    }
    if generation != meta.generation || parent_model_hash != meta.model_hash {
        return Err(HostError::Failure);
    }
    let side = if op == Op::TrainPlus {
        1
    } else if op == Op::TrainMinus {
        -1
    } else {
        return Err(HostError::Failure);
    };
    candidate_into(
        &workspace.model,
        &mut workspace.evaluated_model,
        &meta.model_hash,
        meta.generation,
        side,
        meta.perturbation_q,
    );
    let batch_size = u8::try_from(meta.train_batch_size).map_err(|_| HostError::Failure)?;
    let batch = canonical_batch(&meta.model_hash, meta.generation, batch_size)
        .map_err(|_| HostError::Failure)?;
    let evaluation = evaluate_batch_with_buffers(
        &workspace.evaluated_model,
        &batch,
        256,
        &mut workspace.scratch,
        &mut workspace.predictions,
        &mut workspace.logits,
    )
    .map_err(|_| HostError::Failure)?;
    Ok(RefineResult {
        op,
        status: STATUS_OK,
        request_id,
        generation: meta.generation,
        model_hash: meta.model_hash,
        body: ResultBody::Training {
            side,
            loss: evaluation.loss,
            correct_tokens: evaluation.correct_tokens,
            total_tokens: evaluation.total_tokens,
            eval_digest: evaluation.digest,
        },
    })
}
