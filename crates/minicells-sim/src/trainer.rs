//! Persistent, deterministic native training harness.
//!
//! This module deliberately drives the same `minicells-runtime` refine and
//! accumulate entry points used by the guest.  It is therefore a useful local
//! research harness, rather than a second implementation of the optimizer.

use crate::{execute_refine, memory_host::MemoryHost, meta};
use minicells_core::{
    batch::{batch_digest, canonical_batch},
    model::{model_hash, MODEL_BYTES},
};
use minicells_dataset::Dataset;
use minicells_protocol::{keys, MetaV1, Op, WorkBody, WorkPayload};
use minicells_runtime::{accumulate, AccumulateWorkspace};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::Instant,
};

pub const LOCAL_PROBE_DOMAIN: &[u8] = b"mini-cells:local-probe:v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProbeMetrics {
    pub generation: u64,
    pub model_hash: String,
    pub total_loss: i64,
    pub correct_tokens: u64,
    pub total_tokens: u64,
    pub token_accuracy: f64,
    pub loss_per_token: f64,
}

fn probe_parent_hash() -> [u8; 32] {
    let mut state = blake2b_simd::Params::new().hash_length(32).to_state();
    state.update(LOCAL_PROBE_DOMAIN);
    let mut out = [0; 32];
    out.copy_from_slice(state.finalize().as_bytes());
    out
}

pub fn evaluate_fixed_probe(
    model: &minicells_core::PackedModel,
    generation: u64,
) -> Result<ProbeMetrics, TrainerError> {
    let parent = probe_parent_hash();
    let mut total_loss = 0i64;
    let mut correct_tokens = 0u64;
    let mut total_tokens = 0u64;
    for batch_generation in 0..32u64 {
        let batch =
            canonical_batch(&parent, batch_generation, 4).map_err(|_| TrainerError::Runtime)?;
        let evaluation = minicells_core::batch::evaluate_batch(
            model,
            &batch,
            minicells_protocol::runtime_config::MARGIN_Q as i32,
            &mut minicells_core::Scratch::new(),
        )
        .map_err(|_| TrainerError::Runtime)?;
        total_loss = total_loss.saturating_add(evaluation.loss);
        correct_tokens += evaluation.correct_tokens as u64;
        total_tokens += evaluation.total_tokens as u64;
    }
    Ok(ProbeMetrics {
        generation,
        model_hash: format!("0x{}", hex::encode(model_hash(model))),
        total_loss,
        correct_tokens,
        total_tokens,
        token_accuracy: if total_tokens == 0 {
            0.0
        } else {
            correct_tokens as f64 / total_tokens as f64
        },
        loss_per_token: if total_tokens == 0 {
            0.0
        } else {
            total_loss as f64 / total_tokens as f64
        },
    })
}

pub fn load_checkpoint_model(
    root: &Path,
    generation: u64,
) -> Result<minicells_core::PackedModel, TrainerError> {
    let path = root
        .join("checkpoints")
        .join(format!("generation-{generation:06}"))
        .join("model.bin");
    let bytes = fs::read(path)?;
    minicells_core::PackedModel::decode_from(&bytes)
        .map_err(|_| TrainerError::Checkpoint("model decode failed".into()))
}

#[derive(Debug, thiserror::Error)]
pub enum TrainerError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid checkpoint: {0}")]
    Checkpoint(String),
    #[error("runtime failure while executing native training")]
    Runtime,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenerationMetrics {
    pub backend: String,
    /// The completed generation.  Generation zero is the genesis checkpoint;
    /// the first training transition is recorded as generation one.
    pub generation: u64,
    pub parent_model_hash: String,
    pub next_model_hash: String,
    pub batch_digest: String,
    pub base_loss: i64,
    pub base_correct_tokens: u32,
    pub plus_loss: i64,
    pub plus_correct_tokens: u32,
    pub minus_loss: i64,
    pub minus_correct_tokens: u32,
    pub retained_loss: i64,
    pub retained_correct_tokens: u32,
    pub total_tokens: u32,
    pub decision: String,
    pub updated: bool,
    pub wall_clock_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema: String,
    pub backend: String,
    pub dataset_root: String,
    pub target_generation: u64,
    pub checkpoint_every: u64,
    pub protocol: String,
    pub optimizer: String,
    pub initial_loss: i64,
    pub initial_correct_tokens: u32,
    pub initial_total_tokens: u32,
    pub final_generation: u64,
    pub final_model_hash: String,
    pub final_correct_tokens: u32,
    pub final_total_tokens: u32,
}

/// A complete local state, including the same canonical host storage the
/// runtime sees.  `MemoryHost` is intentionally kept private to callers so a
/// checkpoint can never accidentally omit pending results or metadata.
pub struct NativeTrainer {
    pub host: MemoryHost,
    next_request_id: u64,
    pub dataset_root: String,
    pub dataset: Option<Dataset>,
}

impl NativeTrainer {
    pub fn new(dataset_root: impl Into<String>) -> Result<Self, TrainerError> {
        let mut host = MemoryHost::default();
        accumulate(&mut host, &mut AccumulateWorkspace::new())
            .map_err(|_| TrainerError::Runtime)?;
        Ok(Self {
            host,
            next_request_id: 1,
            dataset_root: dataset_root.into(),
            dataset: None,
        })
    }

    pub fn with_dataset(mut self, dataset: Dataset) -> Self {
        self.dataset_root = dataset.dataset_root.clone();
        self.dataset = Some(dataset);
        self
    }

    pub fn generation(&self) -> u64 {
        meta(&self.host).generation
    }

    pub fn model_hash(&self) -> [u8; 32] {
        meta(&self.host).model_hash
    }

    pub fn initial_evaluation(&self) -> Result<(i64, u32, u32), TrainerError> {
        let genesis = minicells_core::genesis::genesis_model();
        let batch = if let Some(dataset) = &self.dataset {
            dataset
                .batch(minicells_runtime::genesis::GENESIS_MODEL_HASH, 0, 4)
                .map_err(|_| TrainerError::Runtime)?
                .0
        } else {
            canonical_batch(&minicells_runtime::genesis::GENESIS_MODEL_HASH, 0, 4)
                .map_err(|_| TrainerError::Runtime)?
        };
        let evaluation = minicells_core::batch::evaluate_batch(
            &genesis,
            &batch,
            meta(&self.host).margin_q as i32,
            &mut minicells_core::Scratch::new(),
        )
        .map_err(|_| TrainerError::Runtime)?;
        Ok((
            evaluation.loss,
            evaluation.correct_tokens,
            evaluation.total_tokens,
        ))
    }

    pub fn run_generation(&mut self) -> Result<GenerationMetrics, TrainerError> {
        if self.dataset.is_some() {
            return self.run_dataset_generation();
        }
        let before = meta(&self.host);
        let parent = before.model_hash;
        let generation = before.generation;
        let batch = canonical_batch(&parent, generation, before.train_batch_size as u8)
            .map_err(|_| TrainerError::Runtime)?;
        let digest = batch_digest(&batch);
        let start = Instant::now();
        let plus = execute_refine(
            &mut self.host,
            WorkPayload {
                op: Op::TrainPlus,
                flags: 0,
                request_id: self.next_request_id,
                body: WorkBody::Train {
                    generation,
                    parent_model_hash: parent,
                },
            },
        );
        self.next_request_id = self.next_request_id.saturating_add(1);
        self.host.results = vec![plus];
        accumulate(&mut self.host, &mut AccumulateWorkspace::new())
            .map_err(|_| TrainerError::Runtime)?;
        let minus = execute_refine(
            &mut self.host,
            WorkPayload {
                op: Op::TrainMinus,
                flags: 0,
                request_id: self.next_request_id,
                body: WorkBody::Train {
                    generation,
                    parent_model_hash: parent,
                },
            },
        );
        self.next_request_id = self.next_request_id.saturating_add(1);
        self.host.results = vec![minus.clone()];
        accumulate(&mut self.host, &mut AccumulateWorkspace::new())
            .map_err(|_| TrainerError::Runtime)?;
        let after = meta(&self.host);
        let history = self
            .host
            .storage
            .get(keys::history_key(generation as u8).as_slice())
            .ok_or(TrainerError::Runtime)
            .and_then(|b| {
                minicells_protocol::HistoryV1::decode(b).map_err(|_| TrainerError::Runtime)
            })?;
        let minus_result =
            minicells_protocol::RefineResult::decode(&minus).map_err(|_| TrainerError::Runtime)?;
        let (base_loss, base_correct_tokens) = match minus_result.body {
            minicells_protocol::ResultBody::Training {
                base_loss,
                base_correct_tokens,
                ..
            } => (base_loss, base_correct_tokens),
            _ => return Err(TrainerError::Runtime),
        };
        let decision = if history.updated != 0 {
            if history.plus_loss < base_loss && history.plus_loss < history.minus_loss {
                "plus"
            } else {
                "minus"
            }
        } else {
            "keep"
        };
        Ok(GenerationMetrics {
            backend: "native".into(),
            generation: after.generation,
            parent_model_hash: format!("0x{}", hex::encode(parent)),
            next_model_hash: format!("0x{}", hex::encode(after.model_hash)),
            batch_digest: format!("0x{}", hex::encode(digest)),
            base_loss,
            base_correct_tokens,
            plus_loss: history.plus_loss,
            plus_correct_tokens: history.plus_correct,
            minus_loss: history.minus_loss,
            minus_correct_tokens: history.minus_correct,
            retained_loss: after.current_eval_loss,
            retained_correct_tokens: after.current_correct,
            total_tokens: history.tokens,
            decision: decision.into(),
            updated: history.updated != 0,
            wall_clock_ms: start.elapsed().as_millis(),
        })
    }

    fn run_dataset_generation(&mut self) -> Result<GenerationMetrics, TrainerError> {
        let dataset = self.dataset.as_ref().ok_or(TrainerError::Runtime)?;
        let before = meta(&self.host);
        let parent = before.model_hash;
        let generation = before.generation;
        let (batch, selection) = dataset
            .batch(parent, generation, before.train_batch_size as usize)
            .map_err(|_| TrainerError::Runtime)?;
        let start = Instant::now();
        let mut model_bytes = [0u8; MODEL_BYTES];
        let model_raw = self
            .host
            .storage
            .get(keys::MODEL)
            .ok_or(TrainerError::Runtime)?;
        model_bytes.copy_from_slice(model_raw);
        let model = minicells_core::PackedModel::decode_from(&model_bytes)
            .map_err(|_| TrainerError::Runtime)?;
        let plus_model = minicells_core::optimizer::candidate(
            &model,
            &parent,
            generation,
            1,
            before.perturbation_q,
        );
        let minus_model = minicells_core::optimizer::candidate(
            &model,
            &parent,
            generation,
            -1,
            before.perturbation_q,
        );
        let plus = minicells_core::batch::evaluate_batch(
            &plus_model,
            &batch,
            before.margin_q as i32,
            &mut minicells_core::Scratch::new(),
        )
        .map_err(|_| TrainerError::Runtime)?;
        let minus = minicells_core::batch::evaluate_batch(
            &minus_model,
            &batch,
            before.margin_q as i32,
            &mut minicells_core::Scratch::new(),
        )
        .map_err(|_| TrainerError::Runtime)?;
        let mut next = model.clone();
        let decision = minicells_core::optimizer::apply_update(
            &mut next,
            &parent,
            generation,
            before.current_eval_loss,
            plus.loss,
            minus.loss,
            before.update_step_q,
        );
        let next_hash = model_hash(&next);
        next.encode_into(&mut model_bytes)
            .map_err(|_| TrainerError::Runtime)?;
        self.host
            .storage
            .insert(keys::MODEL.to_vec(), model_bytes.to_vec());
        let mut after = before.clone();
        after.generation = after.generation.saturating_add(1);
        after.model_hash = next_hash;
        let (retained_loss, retained_correct) = match decision {
            minicells_core::optimizer::UpdateDecision::Plus => (plus.loss, plus.correct_tokens),
            minicells_core::optimizer::UpdateDecision::Minus => (minus.loss, minus.correct_tokens),
            minicells_core::optimizer::UpdateDecision::Keep => {
                (before.current_eval_loss, before.current_correct)
            }
        };
        after.current_eval_loss = retained_loss;
        after.current_correct = retained_correct;
        after.current_tokens = plus.total_tokens;
        after.successful_updates = after
            .successful_updates
            .saturating_add((decision != minicells_core::optimizer::UpdateDecision::Keep) as u64);
        after.zero_diff_updates = after
            .zero_diff_updates
            .saturating_add((decision == minicells_core::optimizer::UpdateDecision::Keep) as u64);
        let history = minicells_protocol::HistoryV1 {
            generation: after.generation,
            parent_hash: parent,
            model_hash: next_hash,
            plus_loss: plus.loss,
            minus_loss: minus.loss,
            plus_correct: plus.correct_tokens,
            minus_correct: minus.correct_tokens,
            tokens: plus.total_tokens,
            updated: (decision != minicells_core::optimizer::UpdateDecision::Keep) as u8,
        };
        let mut history_bytes = [0u8; minicells_protocol::HistoryV1::LEN];
        let history_len = history
            .encode_into(&mut history_bytes)
            .map_err(|_| TrainerError::Runtime)?;
        self.host.storage.insert(
            keys::history_key(after.history_head).to_vec(),
            history_bytes[..history_len].to_vec(),
        );
        after.history_head = (after.history_head + 1) % 64;
        let mut meta_bytes = [0u8; minicells_protocol::META_ENCODED_LEN];
        let meta_len = after
            .encode_into(&mut meta_bytes)
            .map_err(|_| TrainerError::Runtime)?;
        self.host
            .storage
            .insert(keys::META.to_vec(), meta_bytes[..meta_len].to_vec());
        let decision_name = match decision {
            minicells_core::optimizer::UpdateDecision::Plus => "plus",
            minicells_core::optimizer::UpdateDecision::Minus => "minus",
            minicells_core::optimizer::UpdateDecision::Keep => "keep",
        };
        Ok(GenerationMetrics {
            backend: "native-dataset".into(),
            generation: after.generation,
            parent_model_hash: format!("0x{}", hex::encode(parent)),
            next_model_hash: format!("0x{}", hex::encode(next_hash)),
            batch_digest: selection.digest,
            base_loss: before.current_eval_loss,
            base_correct_tokens: before.current_correct,
            plus_loss: plus.loss,
            plus_correct_tokens: plus.correct_tokens,
            minus_loss: minus.loss,
            minus_correct_tokens: minus.correct_tokens,
            retained_loss: after.current_eval_loss,
            retained_correct_tokens: after.current_correct,
            total_tokens: plus.total_tokens,
            decision: decision_name.into(),
            updated: decision != minicells_core::optimizer::UpdateDecision::Keep,
            wall_clock_ms: start.elapsed().as_millis(),
        })
    }

    fn checkpoint(&self, root: &Path) -> Result<(), TrainerError> {
        let generation = self.generation();
        let dir = root
            .join("checkpoints")
            .join(format!("generation-{generation:06}"));
        fs::create_dir_all(&dir)?;
        let model = self
            .host
            .storage
            .get(keys::MODEL)
            .ok_or_else(|| TrainerError::Checkpoint("missing model".into()))?;
        let meta_bytes = self
            .host
            .storage
            .get(keys::META)
            .ok_or_else(|| TrainerError::Checkpoint("missing metadata".into()))?;
        fs::write(dir.join("model.bin"), model)?;
        fs::write(dir.join("meta.bin"), meta_bytes)?;
        let identity = serde_json::json!({
            "schema": "minicells.checkpoint.v1", "generation": generation,
            "dataset_root": self.dataset_root, "model_hash": format!("0x{}", hex::encode(self.model_hash())),
            "model_bytes": MODEL_BYTES, "protocol": "minicells-protocol-v1",
            "optimizer": "guarded-sign-spsa-v2"
        });
        fs::write(
            dir.join("checkpoint.json"),
            serde_json::to_vec_pretty(&identity)?,
        )?;
        Ok(())
    }

    pub fn load_checkpoint(
        root: &Path,
        dataset_root: impl Into<String>,
    ) -> Result<Self, TrainerError> {
        let dataset_root = dataset_root.into();
        let checkpoints = root.join("checkpoints");
        let mut candidates: Vec<PathBuf> = fs::read_dir(&checkpoints)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        candidates.sort();
        let dir = candidates
            .pop()
            .ok_or_else(|| TrainerError::Checkpoint("no checkpoint directories".into()))?;
        let identity: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("checkpoint.json"))?)?;
        if identity["dataset_root"].as_str() != Some(dataset_root.as_str()) {
            return Err(TrainerError::Checkpoint("dataset root mismatch".into()));
        }
        let model = fs::read(dir.join("model.bin"))?;
        if model.len() != MODEL_BYTES {
            return Err(TrainerError::Checkpoint("model size mismatch".into()));
        }
        let metadata = fs::read(dir.join("meta.bin"))?;
        let m = MetaV1::decode(&metadata)
            .map_err(|_| TrainerError::Checkpoint("metadata decode failed".into()))?;
        let decoded = minicells_core::PackedModel::decode_from(&model)
            .map_err(|_| TrainerError::Checkpoint("model decode failed".into()))?;
        if model_hash(&decoded) != m.model_hash
            || format!("0x{}", hex::encode(m.model_hash))
                != identity["model_hash"].as_str().unwrap_or_default()
        {
            return Err(TrainerError::Checkpoint("model hash mismatch".into()));
        }
        let mut host = MemoryHost::default();
        host.storage.insert(keys::MODEL.to_vec(), model);
        host.storage.insert(keys::META.to_vec(), metadata);
        Ok(Self {
            host,
            next_request_id: m.generation.saturating_mul(2).saturating_add(1),
            dataset_root,
            dataset: None,
        })
    }
}

pub fn run_persistent_native(
    root: &Path,
    target_generation: u64,
    checkpoint_every: u64,
    resume: bool,
    dataset_root: &str,
    dataset: Option<Dataset>,
) -> Result<Vec<GenerationMetrics>, TrainerError> {
    fs::create_dir_all(root.join("checkpoints"))?;
    let mut trainer = if resume {
        NativeTrainer::load_checkpoint(root, dataset_root)?
    } else {
        NativeTrainer::new(dataset_root)?
    };
    if let Some(dataset) = dataset {
        trainer = trainer.with_dataset(dataset);
    }
    let initial = if resume {
        fs::read(root.join("run.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<RunManifest>(&bytes).ok())
            .map(|manifest| {
                (
                    manifest.initial_loss,
                    manifest.initial_correct_tokens,
                    manifest.initial_total_tokens,
                )
            })
            .unwrap_or((0, 0, 0))
    } else {
        trainer.initial_evaluation()?
    };
    let manifest = RunManifest {
        schema: "minicells.run.v1".into(),
        backend: "native".into(),
        dataset_root: dataset_root.into(),
        target_generation,
        checkpoint_every,
        protocol: "minicells-protocol-v1".into(),
        optimizer: "guarded-sign-spsa-v2".into(),
        initial_loss: initial.0,
        initial_correct_tokens: initial.1,
        initial_total_tokens: initial.2,
        final_generation: trainer.generation(),
        final_model_hash: format!("0x{}", hex::encode(trainer.model_hash())),
        final_correct_tokens: 0,
        final_total_tokens: 0,
    };
    fs::write(root.join("run.json"), serde_json::to_vec_pretty(&manifest)?)?;
    if !resume {
        trainer.checkpoint(root)?;
    }
    let metrics_path = root.join("metrics.jsonl");
    let mut output = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(metrics_path)?;
    let mut metrics = Vec::new();
    while trainer.generation() < target_generation {
        let m = trainer.run_generation()?;
        writeln!(output, "{}", serde_json::to_string(&m)?)?;
        if checkpoint_every > 0 && m.generation % checkpoint_every == 0 {
            trainer.checkpoint(root)?;
        }
        metrics.push(m);
    }
    trainer.checkpoint(root)?;
    let mut final_manifest = manifest;
    final_manifest.final_generation = trainer.generation();
    final_manifest.final_model_hash = format!("0x{}", hex::encode(trainer.model_hash()));
    if let Some(last) = metrics.last() {
        final_manifest.final_correct_tokens = last.retained_correct_tokens;
        final_manifest.final_total_tokens = last.total_tokens;
    }
    fs::write(
        root.join("run.json"),
        serde_json::to_vec_pretty(&final_manifest)?,
    )?;
    output.flush()?;
    Ok(metrics)
}

/// Read metrics without requiring a running trainer; used by the research CLI
/// and by reproducibility scripts.
pub fn read_metrics(path: &Path) -> Result<Vec<GenerationMetrics>, TrainerError> {
    let file = fs::File::open(path)?;
    BufReader::new(file)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}
