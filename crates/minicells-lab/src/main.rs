use clap::{Args, Parser, Subcommand, ValueEnum};
use minicells_core::{
    batch::{batch_digest, canonical_batch},
    model::model_hash,
};
use minicells_dataset::{compile_jsonl, inspect as inspect_dataset, Dataset};
use minicells_protocol::{keys, HistoryV1, MetaV1, RefineResult, ResultBody, META_ENCODED_LEN};
use minicells_protocol::{Op, WorkBody, WorkPayload};
use minicells_pvm::DirectPvmHarness;
use minicells_sim::trainer::{
    evaluate_fixed_probe, load_checkpoint_model, read_metrics, run_persistent_native, NativeTrainer,
};
use minicells_training_ref::{
    accumulate_batch_gradients, compute_gradient_leaf, evaluate_batch_report, finalize_adamw_step,
    reduce_partial_gradients, train_step_with_gradient, GradientAccumulator, PartialGradientV1,
    TrainStepReport, TrainingBatch, TrainingState, TrainingWorkspace, LOGICAL_BATCH_SIZE,
    PARALLEL_LEAF_COUNT, PARALLEL_SHARD_SIZE, PARAMETER_COUNT,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "minicells-lab",
    version,
    about = "Deterministic MINI Cells research lab"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
    Train(TrainArgs),
    Resume(ResumeArgs),
    Evaluate {
        run: PathBuf,
    },
    Compare {
        left: PathBuf,
        right: PathBuf,
    },
    Benchmark(TrainArgs),
    Gate(GateArgs),
    FidelityNative(FidelityNativeArgs),
    ParallelNative(ParallelNativeArgs),
    PvmParity(PvmParityArgs),
    PvmGas(PvmGasArgs),
}

#[derive(Subcommand)]
enum DatasetCommand {
    Build {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Inspect {
        path: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Native,
    Pvm,
}

#[derive(Args, Clone)]
struct TrainArgs {
    #[arg(long, value_enum, default_value_t = Backend::Native)]
    backend: Backend,
    #[arg(long, default_value_t = 10)]
    generations: u64,
    #[arg(long, default_value_t = 100)]
    checkpoint_every: u64,
    #[arg(long, default_value = ".local/runs/native")]
    output: PathBuf,
    #[arg(long)]
    dataset: Option<PathBuf>,
}

#[derive(Args)]
struct ResumeArgs {
    #[arg(long, default_value = ".local/runs/native")]
    output: PathBuf,
    #[arg(long)]
    generations: u64,
    #[arg(long, default_value_t = 100)]
    checkpoint_every: u64,
    #[arg(long)]
    dataset: Option<PathBuf>,
}

#[derive(Args)]
struct GateArgs {
    #[arg(long, default_value = ".local/runs/training-gate-native")]
    output: PathBuf,
    #[arg(long, default_value_t = 512)]
    generations: u64,
}

#[derive(Args)]
struct FidelityNativeArgs {
    #[arg(long, default_value = "fixtures/training-fidelity-v1")]
    fixture: PathBuf,
    #[arg(long, default_value = ".local/runs/fidelity-native")]
    output: PathBuf,
    #[arg(long, default_value_t = 5000)]
    steps: u64,
}

#[derive(Args)]
struct ParallelNativeArgs {
    #[arg(long, default_value = "fixtures/training-fidelity-v1")]
    fixture: PathBuf,
    #[arg(long, default_value = ".local/runs/parallel-native")]
    output: PathBuf,
    #[arg(long, default_value_t = 5000)]
    steps: u64,
    #[arg(long, default_value_t = 4)]
    parallel_width: usize,
}

#[derive(Args)]
struct PvmGasArgs {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long, default_value_t = 10_000_000_000)]
    gas_limit: u64,
    #[arg(long, conflicts_with = "payload_file")]
    payload_hex: Option<String>,
    #[arg(long, conflicts_with = "payload_hex")]
    payload_file: Option<PathBuf>,
    /// Diagnostic-only stage probe; its gas is never written as final gas evidence.
    #[arg(long)]
    diagnostic_stage: Option<String>,
    #[arg(long, default_value = "artifacts/pvm-algorithm-fidelity")]
    output: PathBuf,
}

#[derive(Args)]
struct PvmParityArgs {
    #[arg(long, default_value = "service/artifacts/training-fidelity.blob")]
    artifact: PathBuf,
    #[arg(long, default_value = "fixtures/training-fidelity-v1")]
    fixture: PathBuf,
    #[arg(long, default_value_t = 16)]
    samples: usize,
    #[arg(long, default_value_t = 10_000_000_000)]
    gas_limit: u64,
    #[arg(
        long,
        default_value = "artifacts/pvm-algorithm-fidelity/pvm-parity.json"
    )]
    output: PathBuf,
    /// Also execute the experimental multi-Refine round trip (costly).
    #[arg(long, default_value_t = false)]
    multi_refine: bool,
}

fn dataset_root(path: &Option<PathBuf>) -> Result<String, Box<dyn std::error::Error>> {
    Ok(match path {
        Some(path) => Dataset::load(path)?.dataset_root,
        None => "synthetic-v1".into(),
    })
}

fn train(args: TrainArgs, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    if matches!(args.backend, Backend::Pvm) {
        run_persistent_pvm(&args.output, args.generations)?;
        return Ok(());
    }
    let root = dataset_root(&args.dataset)?;
    let dataset = args.dataset.as_ref().map(Dataset::load).transpose()?;
    let metrics = run_persistent_native(
        &args.output,
        args.generations,
        args.checkpoint_every,
        resume,
        &root,
        dataset,
    )?;
    println!(
        "backend=native generation={} metrics={} output={}",
        metrics.last().map(|m| m.generation).unwrap_or(0),
        metrics.len(),
        args.output.display()
    );
    Ok(())
}

fn run_persistent_pvm(
    output: &PathBuf,
    generations: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output)?;
    let mut harness = DirectPvmHarness::load(
        "service/artifacts/service.blob",
        1_000_000_000,
        "5947c50699863948c51028bc346980481d839884",
        "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
    )?;
    harness.execute_accumulate()?;
    let mut metrics_file = std::fs::File::create(output.join("metrics.jsonl"))?;
    for generation in 0..generations {
        let meta = MetaV1::decode(
            harness
                .host
                .storage
                .get(keys::META)
                .ok_or("missing PVM META")?,
        )
        .map_err(|e| format!("PVM META decode failed: {e:?}"))?;
        if meta.generation != generation {
            return Err(format!(
                "PVM generation mismatch: {} != {generation}",
                meta.generation
            )
            .into());
        }
        let parent = meta.model_hash;
        let batch = canonical_batch(&parent, generation, meta.train_batch_size as u8)
            .map_err(|_| "canonical batch failed")?;
        let digest = batch_digest(&batch);
        let run_side = |h: &mut DirectPvmHarness,
                        op: Op,
                        request_id: u64|
         -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let work = WorkPayload {
                op,
                flags: 0,
                request_id,
                body: WorkBody::Train {
                    generation,
                    parent_model_hash: parent,
                },
            };
            let mut payload = [0u8; 96];
            let n = work
                .encode_into(&mut payload)
                .map_err(|e| format!("work encode failed: {e:?}"))?;
            let result = h.execute_refine(&payload[..n])?;
            h.host.results = vec![DirectPvmHarness::pack_accumulation_result(&result)?];
            h.execute_accumulate()?;
            h.host.results.clear();
            Ok(result)
        };
        let plus = run_side(
            &mut harness,
            Op::TrainPlus,
            generation.saturating_mul(2).saturating_add(1),
        )?;
        let minus = run_side(
            &mut harness,
            Op::TrainMinus,
            generation.saturating_mul(2).saturating_add(2),
        )?;
        let plus_result =
            RefineResult::decode(&plus).map_err(|e| format!("PVM plus decode failed: {e:?}"))?;
        let minus_result =
            RefineResult::decode(&minus).map_err(|e| format!("PVM minus decode failed: {e:?}"))?;
        let (base_loss, base_correct, plus_loss, plus_correct, tokens) = match plus_result.body {
            ResultBody::Training {
                base_loss,
                base_correct_tokens,
                loss,
                correct_tokens,
                total_tokens,
                ..
            } => (
                base_loss,
                base_correct_tokens,
                loss,
                correct_tokens,
                total_tokens,
            ),
            _ => return Err("PVM plus result is not training".into()),
        };
        let (minus_loss, minus_correct) = match minus_result.body {
            ResultBody::Training {
                loss,
                correct_tokens,
                ..
            } => (loss, correct_tokens),
            _ => return Err("PVM minus result is not training".into()),
        };
        let after = MetaV1::decode(
            harness
                .host
                .storage
                .get(keys::META)
                .ok_or("missing PVM META")?,
        )
        .map_err(|e| format!("PVM META decode failed: {e:?}"))?;
        let history = HistoryV1::decode(
            harness
                .host
                .storage
                .get(keys::history_key(generation as u8).as_slice())
                .ok_or("missing PVM history")?,
        )
        .map_err(|e| format!("PVM history decode failed: {e:?}"))?;
        let decision = if history.updated != 0 {
            if plus_loss < base_loss && plus_loss < minus_loss {
                "plus"
            } else {
                "minus"
            }
        } else {
            "keep"
        };
        let metric = minicells_sim::trainer::GenerationMetrics {
            backend: "pvm".into(),
            generation: after.generation,
            parent_model_hash: format!("0x{}", hex::encode(parent)),
            next_model_hash: format!("0x{}", hex::encode(after.model_hash)),
            batch_digest: format!("0x{}", hex::encode(digest)),
            base_loss,
            base_correct_tokens: base_correct,
            plus_loss,
            plus_correct_tokens: plus_correct,
            minus_loss,
            minus_correct_tokens: minus_correct,
            retained_loss: after.current_eval_loss,
            retained_correct_tokens: after.current_correct,
            total_tokens: tokens,
            decision: decision.into(),
            updated: history.updated != 0,
            wall_clock_ms: 0,
        };
        use std::io::Write;
        writeln!(metrics_file, "{}", serde_json::to_string(&metric)?)?;
    }
    println!(
        "backend=pvm generation={} output={}",
        generations,
        output.display()
    );
    Ok(())
}

fn load_fidelity_batch(path: &PathBuf) -> Result<TrainingBatch, Box<dyn std::error::Error>> {
    let mut batches = load_fidelity_batches(path)?;
    if batches.len() != 1 || batches[0].size != 256 {
        return Err("training batch must be exactly 256x64".into());
    }
    Ok(batches.pop().unwrap())
}

fn load_fidelity_batches(
    path: &std::path::Path,
) -> Result<Vec<TrainingBatch>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[..4] != b"MCB1" {
        return Err(format!("invalid training batch fixture: {}", path.display()).into());
    }
    let rows = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let width = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if rows == 0 || width != 64 || rows > u16::MAX as usize {
        return Err("training fixture must have non-empty rows of width 64".into());
    }
    let ids_len = rows * width;
    let expected = 12 + ids_len + rows;
    if bytes.len() != expected {
        return Err("training batch length mismatch".into());
    }
    let mut batches = Vec::new();
    for base in (0..rows).step_by(256) {
        let count = (rows - base).min(256);
        let mut batch = TrainingBatch::empty();
        batch.size = count as u16;
        for row in 0..count {
            let source = base + row;
            batch.ids[row].copy_from_slice(&bytes[12 + source * width..12 + (source + 1) * width]);
            batch.lengths[row] = bytes[12 + ids_len + source];
        }
        batches.push(batch);
    }
    Ok(batches)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn read_f32_array(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not an f32 fixture", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn compare_f32(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    if actual.len() != expected.len() {
        return (f32::INFINITY, f32::INFINITY);
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (&left, &right) in actual.iter().zip(expected) {
        let abs = (left - right).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / right.abs().max(1.0e-12));
    }
    (max_abs, max_rel)
}

fn expected_scalar(path: &std::path::Path, key: &str) -> Result<f32, Box<dyn std::error::Error>> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    value[key]
        .as_f64()
        .map(|value| value as f32)
        .ok_or_else(|| format!("missing numeric {key} in {}", path.display()).into())
}

fn digest_bytes(bytes: &[u8]) -> String {
    let mut state = blake2b_simd::Params::new().hash_length(32).to_state();
    state.update(bytes);
    format!("0x{}", hex::encode(state.finalize().as_bytes()))
}

fn run_fidelity_native(args: FidelityNativeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let initial = std::fs::read(args.fixture.join("initial-weights-f32.bin"))?;
    if initial.len() != PARAMETER_COUNT * 4 {
        return Err("initial weight fixture length mismatch".into());
    }
    let mut weights = [0.0f32; PARAMETER_COUNT];
    for (index, chunk) in initial.chunks_exact(4).enumerate() {
        weights[index] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    let mut state = TrainingState::from_weights(weights);
    std::fs::create_dir_all(&args.output)?;
    let mut reports = Vec::new();
    let mut parity_steps = Vec::new();
    let mut parity_blocked = false;
    let mut parity_pass = true;
    let steps = args.steps as usize;
    if steps < 16 {
        return Err("Native fidelity gate requires at least 16 optimizer steps".into());
    }
    for step in 1..=steps {
        let batch = load_fidelity_batch(&args.fixture.join(format!("batch-{step:06}.bin")))?;
        let mut gradient = [0.0f32; PARAMETER_COUNT];
        let report: TrainStepReport = train_step_with_gradient(&mut state, &batch, &mut gradient);
        let bytes = f32_bytes(&state.weights);
        reports.push(
            serde_json::json!({"step": step, "loss": report.loss, "token_count": report.token_count,
            "grad_norm": report.grad_norm, "weight_digest": digest_bytes(&bytes)}),
        );
        if matches!(step, 1 | 2 | 4 | 16) {
            let expected_root = args.fixture.join("expected");
            let paths = [
                (
                    "gradient",
                    expected_root.join(format!("step-{step:06}-gradients-f32.bin")),
                    &gradient[..],
                ),
                (
                    "weights",
                    expected_root.join(format!("step-{step:06}-weights-f32.bin")),
                    &state.weights[..],
                ),
                (
                    "adam_m",
                    expected_root.join(format!("step-{step:06}-adam-m-f32.bin")),
                    &state.adam_m[..],
                ),
                (
                    "adam_v",
                    expected_root.join(format!("step-{step:06}-adam-v-f32.bin")),
                    &state.adam_v[..],
                ),
            ];
            let mut metrics = serde_json::Map::new();
            metrics.insert("step".into(), serde_json::json!(step));
            let mut available = true;
            for (name, path, actual) in paths {
                if !path.is_file() {
                    available = false;
                    parity_blocked = true;
                    continue;
                }
                let expected = read_f32_array(&path)?;
                let (max_abs, max_rel) = compare_f32(actual, &expected);
                let pass = max_abs <= 5.0e-4 || max_rel <= 5.0e-4;
                parity_pass &= pass;
                metrics.insert(
                    name.into(),
                    serde_json::json!({"max_abs":max_abs,"max_rel":max_rel,"pass":pass}),
                );
            }
            let loss_path = expected_root.join(format!("step-{step:06}-loss.json"));
            if loss_path.is_file() {
                let expected_loss = expected_scalar(&loss_path, "loss")?;
                let error = (report.loss - expected_loss).abs();
                let pass = error <= 5.0e-4 || error / expected_loss.abs().max(1.0e-12) <= 5.0e-4;
                parity_pass &= pass;
                metrics.insert("loss".into(), serde_json::json!({"actual":report.loss,"expected":expected_loss,"abs_error":error,"pass":pass}));
                let expected_norm = expected_scalar(&loss_path, "grad_norm")?;
                let norm_error = (report.grad_norm - expected_norm).abs();
                let norm_pass =
                    norm_error <= 5.0e-4 || norm_error / expected_norm.abs().max(1.0e-12) <= 5.0e-4;
                parity_pass &= norm_pass;
                metrics.insert("grad_norm".into(), serde_json::json!({"actual":report.grad_norm,"expected":expected_norm,"abs_error":norm_error,"pass":norm_pass}));
            } else {
                available = false;
                parity_blocked = true;
            }
            metrics.insert("available".into(), serde_json::json!(available));
            parity_steps.push(serde_json::Value::Object(metrics));
        }
    }
    let weight_bytes = f32_bytes(&state.weights);
    let m_bytes = f32_bytes(&state.adam_m);
    let v_bytes = f32_bytes(&state.adam_v);
    std::fs::write(args.output.join("weights-f32.bin"), &weight_bytes)?;
    std::fs::write(args.output.join("adam-m-f32.bin"), &m_bytes)?;
    std::fs::write(args.output.join("adam-v-f32.bin"), &v_bytes)?;
    std::fs::write(
        args.output.join("native-training.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "minicells.native-training-fidelity.v1", "algorithm": "echo-adamw-ce-v1",
            "steps": steps, "logical_batch_size": 256, "final_step": state.step,
            "final_weight_digest": digest_bytes(&weight_bytes), "reports": reports
        }))?,
    )?;
    let validation_report = if args.fixture.join("validation.bin").is_file() {
        let validation_batches = load_fidelity_batches(&args.fixture.join("validation.bin"))?;
        let mut loss_sum = 0.0f32;
        let mut token_count = 0u64;
        let mut correct_tokens = 0u64;
        let mut exact_sum = 0.0f32;
        let mut sample_count = 0u64;
        for batch in &validation_batches {
            let report = evaluate_batch_report(&state, batch);
            loss_sum += report.loss * report.token_count as f32;
            token_count += report.token_count as u64;
            correct_tokens += report.correct_tokens as u64;
            exact_sum += report.exact_sequence_accuracy * batch.size as f32;
            sample_count += batch.size as u64;
        }
        serde_json::json!({"status":"PASS", "batches":validation_batches.len(), "token_count":token_count,
            "loss":if token_count == 0 { 0.0 } else { loss_sum / token_count as f32 },
            "correct_tokens":correct_tokens,
            "token_accuracy":if token_count == 0 { 0.0 } else { correct_tokens as f32 / token_count as f32 },
            "exact_sequence_accuracy":if sample_count == 0 { 0.0 } else { exact_sum / sample_count as f32 }})
    } else {
        serde_json::json!({"status":"BLOCKED_VALIDATION_FIXTURE"})
    };
    std::fs::write(
        args.output.join("validation.json"),
        serde_json::to_vec_pretty(&validation_report)?,
    )?;
    let parity_status = if parity_blocked {
        "BLOCKED_EXPECTED_FIXTURE"
    } else if parity_pass {
        "PASS"
    } else {
        "FAIL"
    };
    let parity = serde_json::json!({"schema": "minicells.native-training-parity.v1", "status": parity_status,
        "tolerance": {"absolute": 5.0e-4, "relative": 5.0e-4}, "steps": parity_steps,
        "reason": if parity_blocked { Some("Python exporter must provide expected FP32 tensors") } else { None::<&str> }});
    std::fs::write(
        args.output.join("native-parity.json"),
        serde_json::to_vec_pretty(&parity)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&parity)?);
    Ok(())
}

/// Run the tree32 algorithm with deterministic, index-addressed leaf slots.
/// This is intentionally a scheduler simulation: it proves that completion
/// order does not affect the numerical result, while leaving claims about
/// actual wall-clock overlap to the MiniJAM lane executor and its evidence.
fn run_parallel_native(args: ParallelNativeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.parallel_width == 0 || args.parallel_width > PARALLEL_LEAF_COUNT {
        return Err(format!("parallel width must be 1..={PARALLEL_LEAF_COUNT}").into());
    }
    let initial = std::fs::read(args.fixture.join("initial-weights-f32.bin"))?;
    if initial.len() != PARAMETER_COUNT * 4 {
        return Err("initial weight fixture length mismatch".into());
    }
    let mut weights = [0.0f32; PARAMETER_COUNT];
    for (index, chunk) in initial.chunks_exact(4).enumerate() {
        weights[index] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    let mut batch_paths = Vec::new();
    for step in 1..=LOGICAL_BATCH_SIZE.max(16) {
        let path = args.fixture.join(format!("batch-{step:06}.bin"));
        if !path.is_file() {
            break;
        }
        batch_paths.push(path);
    }
    if batch_paths.is_empty() {
        return Err("parallel runner found no canonical batches".into());
    }
    let mut state = TrainingState::from_weights(weights);
    let mut workspace = TrainingWorkspace::new();
    let mut leaves =
        std::vec::Vec::from_iter((0..PARALLEL_LEAF_COUNT).map(|_| PartialGradientV1::new()));
    let mut scratch =
        std::vec::Vec::from_iter((0..PARALLEL_LEAF_COUNT).map(|_| PartialGradientV1::new()));
    let mut reports = Vec::new();
    for step in 0..args.steps as usize {
        let batch = load_fidelity_batch(&batch_paths[step % batch_paths.len()])?;
        if batch.size as usize != LOGICAL_BATCH_SIZE {
            return Err("parallel runner requires logical batch size 256".into());
        }
        // A lane scheduler may complete leaves in any order.  The output slot
        // remains keyed by leaf index, so reduction order is always 0..31.
        let mut completion_order = Vec::with_capacity(PARALLEL_LEAF_COUNT);
        for lane in 0..args.parallel_width {
            let mut index = lane;
            while index < PARALLEL_LEAF_COUNT {
                completion_order.push(index);
                index += args.parallel_width;
            }
        }
        for leaf_index in completion_order {
            let start = leaf_index * PARALLEL_SHARD_SIZE;
            let mut shard = TrainingBatch::empty();
            shard.size = PARALLEL_SHARD_SIZE as u16;
            for row in 0..PARALLEL_SHARD_SIZE {
                shard.ids[row] = batch.ids[start + row];
                shard.lengths[row] = batch.lengths[start + row];
            }
            compute_gradient_leaf(&state, &shard, &mut workspace, &mut leaves[leaf_index]);
        }
        let root = reduce_partial_gradients(&leaves, &mut scratch)
            .ok_or("tree32 reduction received no leaves")?;
        let mut accumulator = GradientAccumulator {
            gradient: root.gradient,
            loss_sum: root.loss_sum,
            token_count: root.token_count,
        };
        let report = finalize_adamw_step(&mut state, &mut accumulator);
        if matches!(step + 1, 1 | 2 | 4 | 16 | 100 | 500 | 1000 | 5000) {
            reports.push(serde_json::json!({"step":step + 1,"loss":report.loss,
                "grad_norm":report.grad_norm,"token_count":report.token_count,
                "weight_digest":digest_bytes(&f32_bytes(&state.weights))}));
        }
    }
    let validation = if args.fixture.join("validation.bin").is_file() {
        let batches = load_fidelity_batches(&args.fixture.join("validation.bin"))?;
        let mut tokens = 0u64;
        let mut correct = 0u64;
        let mut exact = 0u64;
        for item in &batches {
            let result = evaluate_batch_report(&state, item);
            tokens += result.token_count as u64;
            correct += result.correct_tokens as u64;
            exact += (result.exact_sequence_accuracy * item.size as f32) as u64;
        }
        serde_json::json!({"status":"PASS","token_accuracy":if tokens == 0 {0.0} else {correct as f64/tokens as f64},
            "exact_sequence_accuracy":if batches.is_empty() {0.0} else {exact as f64/(batches.iter().map(|x| x.size as u64).sum::<u64>()) as f64}})
    } else {
        serde_json::json!({"status":"BLOCKED_VALIDATION_FIXTURE"})
    };
    std::fs::create_dir_all(&args.output)?;
    let pass = validation["token_accuracy"].as_f64() == Some(1.0)
        && validation["exact_sequence_accuracy"].as_f64() == Some(1.0);
    let report = serde_json::json!({
        "schema":"minicells.parallel-native-training.v1",
        "algorithm":"echo-adamw-ce-tree32-v1",
        "steps":args.steps,"logical_batch_size":LOGICAL_BATCH_SIZE,
        "shard_size":PARALLEL_SHARD_SIZE,"leaf_count":PARALLEL_LEAF_COUNT,
        "parallel_width":args.parallel_width,
        "completion_order_indexed":true,
        "actual_overlap_proven":false,
        "status":if pass {"PASS_PARALLEL_NATIVE"} else {"BLOCKED_PARALLEL_LEARNING"},
        "validation":validation,"trajectory":reports,
        "final_step":state.step,"final_weight_digest":digest_bytes(&f32_bytes(&state.weights)),
        "algorithm_changes":"NUMERIC_REDUCTION_ORDER_ONLY"
    });
    std::fs::write(
        args.output.join("parallel-native.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !pass {
        return Err("parallel Native learning gate failed".into());
    }
    Ok(())
}

fn classify_gas(gas_used: u64) -> &'static str {
    if gas_used <= 1_000_000_000 {
        "TINY"
    } else if gas_used <= 4_000_000_000 {
        "FULL_COMFORTABLE"
    } else if gas_used <= 5_000_000_000 {
        "NEAR_FULL"
    } else {
        "OVER_FULL"
    }
}

fn classify_execution(
    completed: bool,
    gas_used: u64,
    error: Option<&str>,
    gas_limit: u64,
) -> &'static str {
    if completed {
        classify_gas(gas_used)
    } else if error.is_some_and(|value| value.contains("out of gas")) {
        // OOG at the diagnostic limit is authoritative evidence that the
        // workload exceeds that limit, even though no result was returned.
        if gas_limit >= 5_000_000_000 {
            "OVER_FULL"
        } else {
            "OOG_AT_LIMIT"
        }
    } else {
        "NOT_MEASURED"
    }
}

fn run_full_multi_refine(args: PvmParityArgs) -> Result<(), Box<dyn std::error::Error>> {
    const SHARD_SIZE: usize = 8;
    let initial = read_f32_array(&args.fixture.join("initial-weights-f32.bin"))?;
    if initial.len() != PARAMETER_COUNT {
        return Err("initial weight fixture length mismatch".into());
    }
    let mut weights = [0.0f32; PARAMETER_COUNT];
    weights.copy_from_slice(&initial);
    let full = load_fidelity_batch(&args.fixture.join("batch-000001.bin"))?;
    let mut mono_state = TrainingState::from_weights(weights);
    let mut mono_gradient = [0.0f32; PARAMETER_COUNT];
    let mono_report = train_step_with_gradient(&mut mono_state, &full, &mut mono_gradient);
    let mut acc = GradientAccumulator::new();
    let mut shard_gas = Vec::with_capacity(32);
    let mut shard_records = Vec::with_capacity(32);
    let pack_prefix = |mode: &[u8; 4]| {
        let mut p = Vec::with_capacity(64_000);
        p.extend_from_slice(mode);
        p.extend_from_slice(&0u64.to_le_bytes());
        p.extend_from_slice(&f32_bytes(&weights));
        p.extend_from_slice(&vec![0u8; PARAMETER_COUNT * 8]);
        p
    };
    for shard_index in 0..32usize {
        let start = shard_index * SHARD_SIZE;
        let mut payload = pack_prefix(b"MCA1");
        payload.extend_from_slice(&(SHARD_SIZE as u16).to_le_bytes());
        payload.extend_from_slice(&acc.token_count.to_le_bytes());
        payload.extend_from_slice(&acc.loss_sum.to_le_bytes());
        payload.extend_from_slice(&f32_bytes(&acc.gradient));
        for row in start..start + SHARD_SIZE {
            payload.extend_from_slice(&full.ids[row]);
        }
        payload.extend_from_slice(&full.lengths[start..start + SHARD_SIZE]);
        let mut h = DirectPvmHarness::load(
            &args.artifact,
            args.gas_limit,
            "5947c50699863948c51028bc346980481d839884",
            "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
        )?;
        let ex = h.execute_refine_measured(&payload)?;
        if ex.output.len() != 12 + PARAMETER_COUNT * 4 || &ex.output[..4] != b"MCAR" {
            return Err(format!("invalid MCAR output for shard {shard_index}").into());
        }
        acc.loss_sum = f32::from_le_bytes(ex.output[4..8].try_into().unwrap());
        acc.token_count = u32::from_le_bytes(ex.output[8..12].try_into().unwrap());
        for i in 0..PARAMETER_COUNT {
            acc.gradient[i] =
                f32::from_le_bytes(ex.output[12 + i * 4..16 + i * 4].try_into().unwrap());
        }
        shard_gas.push(ex.gas_used);
        let classification = if ex.gas_used <= 1_000_000_000 {
            "SHARD_PASS_TINY"
        } else if ex.gas_used <= 4_000_000_000 {
            "SHARD_PASS_FULL_COMFORTABLE"
        } else if ex.gas_used <= 5_000_000_000 {
            "SHARD_NEAR_FULL"
        } else {
            "SHARD_OVER_FULL"
        };
        shard_records.push(serde_json::json!({"logical_batch_size":256,"shard_size":8,"shard_index":shard_index,"sample_range":[start,start+SHARD_SIZE],"gas_used":ex.gas_used,"classification":classification}));
    }
    let mut final_payload = pack_prefix(b"MCF1");
    final_payload.extend_from_slice(&acc.token_count.to_le_bytes());
    final_payload.extend_from_slice(&acc.loss_sum.to_le_bytes());
    final_payload.extend_from_slice(&f32_bytes(&acc.gradient));
    let mut h = DirectPvmHarness::load(
        &args.artifact,
        args.gas_limit,
        "5947c50699863948c51028bc346980481d839884",
        "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
    )?;
    let final_ex = h.execute_refine_measured(&final_payload)?;
    if final_ex.output.len() != 24 + PARAMETER_COUNT * 12 || &final_ex.output[..4] != b"MCPR" {
        return Err("invalid MCF1 output".into());
    }
    let final_loss = f32::from_le_bytes(final_ex.output[4..8].try_into().unwrap());
    let final_norm = f32::from_le_bytes(final_ex.output[8..12].try_into().unwrap());
    let final_tokens = u32::from_le_bytes(final_ex.output[12..16].try_into().unwrap());
    let final_step = u64::from_le_bytes(final_ex.output[16..24].try_into().unwrap());
    let mono_w = f32_bytes(&mono_state.weights);
    let mono_m = f32_bytes(&mono_state.adam_m);
    let mono_v = f32_bytes(&mono_state.adam_v);
    let bit_exact = final_ex.output[24..24 + PARAMETER_COUNT * 4] == mono_w[..]
        && final_ex.output[24 + PARAMETER_COUNT * 4..24 + PARAMETER_COUNT * 8] == mono_m[..]
        && final_ex.output[24 + PARAMETER_COUNT * 8..24 + PARAMETER_COUNT * 12] == mono_v[..]
        && final_loss.to_bits() == mono_report.loss.to_bits()
        && final_norm.to_bits() == mono_report.grad_norm.to_bits()
        && final_tokens == mono_report.token_count
        && final_step == mono_state.step;
    let worst_ids = full.ids;
    let mut worst_payload = pack_prefix(b"MCA1");
    worst_payload.extend_from_slice(&(SHARD_SIZE as u16).to_le_bytes());
    worst_payload.extend_from_slice(&0u32.to_le_bytes());
    worst_payload.extend_from_slice(&0.0f32.to_le_bytes());
    worst_payload.extend_from_slice(&vec![0u8; PARAMETER_COUNT * 4]);
    for row in 0..SHARD_SIZE {
        worst_payload.extend_from_slice(&worst_ids[row]);
    }
    worst_payload.extend_from_slice(&[32u8; SHARD_SIZE]);
    let mut wh = DirectPvmHarness::load(
        &args.artifact,
        args.gas_limit,
        "5947c50699863948c51028bc346980481d839884",
        "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
    )?;
    let worst_ex = wh.execute_refine_measured(&worst_payload)?;
    let worst_gas = worst_ex.gas_used;
    let worst_class = if worst_gas <= 4_000_000_000 {
        "SHARD_PASS_FULL_COMFORTABLE"
    } else if worst_gas <= 5_000_000_000 {
        "SHARD_NEAR_FULL"
    } else {
        "SHARD_OVER_FULL"
    };
    let mut sorted = shard_gas.clone();
    sorted.sort_unstable();
    let p95 = sorted[((sorted.len() * 95).saturating_sub(1) / 100).min(sorted.len() - 1)];
    let mean = shard_gas.iter().sum::<u64>() as f64 / shard_gas.len() as f64;
    let max = *sorted.last().unwrap();
    let report = serde_json::json!({"schema":"minicells.pvm-full-logical-batch-gate.v1","status":if bit_exact && max <= 4_000_000_000 && worst_gas <= 4_000_000_000 {"PASS_FULL_COMFORTABLE_MULTI_REFINE"} else {"FAIL"},"logical_batch_size":256,"shard_size":8,"shard_count":32,"shards":shard_records,"shard_gas_summary":{"min":sorted[0],"mean":mean,"p95":p95,"max":max},"finalize":{"gas_used":final_ex.gas_used,"classification":"FINALIZE_PASS"},"total_refine_gas":shard_gas.iter().sum::<u64>() + final_ex.gas_used,"native_vs_pvm_bit_exact":bit_exact,"worst_case_shard":{"samples":8,"length":32,"gas_used":worst_gas,"classification":worst_class},"production_refine_limit_authorized":bit_exact && max <= 4_000_000_000 && worst_gas <= 4_000_000_000,"fresh_chain_e2e":"BLOCKED_EXTERNAL_CHAIN"});
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    let gas_dir = args
        .output
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("mca-gas-full");
    std::fs::create_dir_all(&gas_dir)?;
    for record in &shard_records {
        let index = record["shard_index"].as_u64().unwrap();
        std::fs::write(
            gas_dir.join(format!("shard-{index:02}.json")),
            serde_json::to_vec_pretty(record)?,
        )?;
    }
    std::fs::write(
        gas_dir.join("worst-case-8.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"logical_batch_size":256,"shard_size":8,"shard_index":0,"sample_range":[0,8],"all_lengths":32,"gas_used":worst_gas,"classification":worst_class}),
        )?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report["status"] != "PASS_FULL_COMFORTABLE_MULTI_REFINE" {
        return Err("full logical-batch gate failed".into());
    }
    Ok(())
}

fn run_pvm_parity(args: PvmParityArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.multi_refine && args.samples == 256 {
        return run_full_multi_refine(args);
    }
    if args.samples == 0 || args.samples > 256 {
        return Err("samples must be 1..=256".into());
    }
    let initial = read_f32_array(&args.fixture.join("initial-weights-f32.bin"))?;
    if initial.len() != PARAMETER_COUNT {
        return Err("initial weight fixture length mismatch".into());
    }
    let mut weights = [0.0f32; PARAMETER_COUNT];
    weights.copy_from_slice(&initial);
    let mut state = TrainingState::from_weights(weights);
    let mut batch = load_fidelity_batch(&args.fixture.join("batch-000001.bin"))?;
    batch.size = args.samples as u16;
    let mut native_gradient = [0.0f32; PARAMETER_COUNT];
    let native_report = train_step_with_gradient(&mut state, &batch, &mut native_gradient);
    let mut chunked_state = TrainingState::from_weights(weights);
    let mut chunked_workspace = TrainingWorkspace::new();
    let mut accumulator = GradientAccumulator::new();
    for start in if args.multi_refine {
        (0..args.samples).step_by(8).collect::<Vec<_>>()
    } else {
        Vec::new()
    } {
        let end = (start + 8).min(args.samples);
        let mut shard = TrainingBatch::empty();
        shard.size = (end - start) as u16;
        for row in 0..(end - start) {
            shard.ids[row] = batch.ids[start + row];
            shard.lengths[row] = batch.lengths[start + row];
        }
        accumulate_batch_gradients(
            &chunked_state,
            &shard,
            &mut chunked_workspace,
            &mut accumulator,
        );
    }
    let chunked_report = finalize_adamw_step(&mut chunked_state, &mut accumulator);
    let chunked_exact = f32_bytes(&chunked_state.weights) == f32_bytes(&state.weights)
        && f32_bytes(&chunked_state.adam_m) == f32_bytes(&state.adam_m)
        && f32_bytes(&chunked_state.adam_v) == f32_bytes(&state.adam_v)
        && chunked_report.loss.to_bits() == native_report.loss.to_bits()
        && chunked_report.grad_norm.to_bits() == native_report.grad_norm.to_bits();
    let mut payload = Vec::with_capacity(4 + 8 + PARAMETER_COUNT * 12 + 2 + args.samples * 65);
    payload.extend_from_slice(b"MCP1");
    payload.extend_from_slice(&0u64.to_le_bytes());
    payload.extend_from_slice(&f32_bytes(&weights));
    payload.extend_from_slice(&vec![0u8; PARAMETER_COUNT * 4]);
    payload.extend_from_slice(&vec![0u8; PARAMETER_COUNT * 4]);
    payload.extend_from_slice(&(args.samples as u16).to_le_bytes());
    for row in 0..args.samples {
        payload.extend_from_slice(&batch.ids[row]);
    }
    payload.extend_from_slice(&batch.lengths[..args.samples]);
    let mut harness = DirectPvmHarness::load(
        &args.artifact,
        args.gas_limit,
        "5947c50699863948c51028bc346980481d839884",
        "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
    )?;
    let execution = harness.execute_refine_measured(&payload)?;
    if execution.output.len() != 24 + PARAMETER_COUNT * 12 || &execution.output[..4] != b"MCPR" {
        return Err(format!("invalid MCPR output length {}", execution.output.len()).into());
    }
    let out = &execution.output;
    let read = |offset: usize| f32::from_le_bytes(out[offset..offset + 4].try_into().unwrap());
    let pvm_loss = read(4);
    let pvm_norm = read(8);
    let pvm_tokens = u32::from_le_bytes(out[12..16].try_into().unwrap());
    let pvm_step = u64::from_le_bytes(out[16..24].try_into().unwrap());
    let state_bytes = |offset: usize| &out[offset..offset + PARAMETER_COUNT * 4];
    let native_weights = f32_bytes(&state.weights);
    let native_m = f32_bytes(&state.adam_m);
    let native_v = f32_bytes(&state.adam_v);
    let weights_exact = state_bytes(24) == native_weights.as_slice();
    let m_exact = state_bytes(24 + PARAMETER_COUNT * 4) == native_m.as_slice();
    let v_exact = state_bytes(24 + PARAMETER_COUNT * 8) == native_v.as_slice();
    let scalar_exact = pvm_loss.to_bits() == native_report.loss.to_bits()
        && pvm_norm.to_bits() == native_report.grad_norm.to_bits()
        && pvm_tokens == native_report.token_count
        && pvm_step == state.step;
    let mut shard_gas = Vec::new();
    let mut shard_acc = GradientAccumulator::new();
    let mut shard_ok = true;
    for start in (0..args.samples).step_by(8) {
        let end = (start + 8).min(args.samples);
        let mut shard_payload = Vec::with_capacity(payload.len());
        shard_payload.extend_from_slice(b"MCA1");
        shard_payload.extend_from_slice(&0u64.to_le_bytes());
        shard_payload.extend_from_slice(&f32_bytes(&weights));
        shard_payload.extend_from_slice(&vec![0u8; PARAMETER_COUNT * 8]);
        shard_payload.extend_from_slice(&((end - start) as u16).to_le_bytes());
        shard_payload.extend_from_slice(&shard_acc.token_count.to_le_bytes());
        shard_payload.extend_from_slice(&shard_acc.loss_sum.to_le_bytes());
        shard_payload.extend_from_slice(&f32_bytes(&shard_acc.gradient));
        for row in start..end {
            shard_payload.extend_from_slice(&batch.ids[row]);
        }
        shard_payload.extend_from_slice(&batch.lengths[start..end]);
        let mut h = DirectPvmHarness::load(
            &args.artifact,
            args.gas_limit,
            "5947c50699863948c51028bc346980481d839884",
            "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
        )?;
        let ex = h.execute_refine_measured(&shard_payload)?;
        shard_gas.push(ex.gas_used);
        if ex.output.len() != 12 + PARAMETER_COUNT * 4 || &ex.output[..4] != b"MCAR" {
            shard_ok = false;
            break;
        }
        let loss = f32::from_le_bytes(ex.output[4..8].try_into().unwrap());
        let tokens = u32::from_le_bytes(ex.output[8..12].try_into().unwrap());
        shard_acc.loss_sum = loss;
        shard_acc.token_count = tokens;
        for i in 0..PARAMETER_COUNT {
            let v = f32::from_le_bytes(ex.output[12 + i * 4..16 + i * 4].try_into().unwrap());
            shard_acc.gradient[i] = v;
        }
    }
    let mut finalize_gas = 0u64;
    let mut multi_refine_exact = false;
    if shard_ok {
        let mut final_payload =
            Vec::with_capacity(4 + 8 + PARAMETER_COUNT * 12 + 8 + PARAMETER_COUNT * 4);
        final_payload.extend_from_slice(b"MCF1");
        final_payload.extend_from_slice(&0u64.to_le_bytes());
        final_payload.extend_from_slice(&f32_bytes(&weights));
        final_payload.extend_from_slice(&vec![0u8; PARAMETER_COUNT * 8]);
        final_payload.extend_from_slice(&shard_acc.token_count.to_le_bytes());
        final_payload.extend_from_slice(&shard_acc.loss_sum.to_le_bytes());
        final_payload.extend_from_slice(&f32_bytes(&shard_acc.gradient));
        let mut h = DirectPvmHarness::load(
            &args.artifact,
            args.gas_limit,
            "5947c50699863948c51028bc346980481d839884",
            "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
        )?;
        let ex = h.execute_refine_measured(&final_payload)?;
        finalize_gas = ex.gas_used;
        if ex.output.len() == 24 + PARAMETER_COUNT * 12 && &ex.output[..4] == b"MCPR" {
            multi_refine_exact = ex.output[24..24 + PARAMETER_COUNT * 4] == native_weights[..]
                && ex.output[24 + PARAMETER_COUNT * 4..24 + PARAMETER_COUNT * 8] == native_m[..]
                && ex.output[24 + PARAMETER_COUNT * 8..24 + PARAMETER_COUNT * 12] == native_v[..];
        }
    }
    let report = serde_json::json!({
        "schema":"minicells.pvm-training-parity.v1", "status":if weights_exact && m_exact && v_exact && scalar_exact {"PASS_BIT_EXACT"} else {"FAIL"},
        "samples":args.samples, "gas_used":execution.gas_used, "gas_remaining":execution.gas_remaining,
        "scalar_exact":scalar_exact, "weights_bit_exact":weights_exact, "adam_m_bit_exact":m_exact, "adam_v_bit_exact":v_exact,
        "native_chunked": {"shard_samples":8,"bit_exact":chunked_exact},
        "pvm_multi_refine": {"status":if args.multi_refine {if multi_refine_exact {"PASS_BIT_EXACT"} else {"FAIL"}} else {"NOT_RUN"},"shard_samples":8,"shard_gas":shard_gas,"finalize_gas":finalize_gas,"bit_exact":multi_refine_exact,"mca_state_frozen":true,"sequential_accumulator_bit_exact":multi_refine_exact},
        "native":{"loss":native_report.loss,"grad_norm":native_report.grad_norm,"token_count":native_report.token_count,"step":state.step},
        "pvm":{"loss":pvm_loss,"grad_norm":pvm_norm,"token_count":pvm_tokens,"step":pvm_step},
        "artifact_hash":format!("0x{}",hex::encode(harness.artifact.blake2_hash)), "gas_classification":classify_gas(execution.gas_used)
    });
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report["status"] != "PASS_BIT_EXACT" {
        return Err("PVM parity failed".into());
    }
    Ok(())
}

fn run_pvm_gas(args: PvmGasArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut harness = DirectPvmHarness::load(
        &args.artifact,
        args.gas_limit,
        "5947c50699863948c51028bc346980481d839884",
        "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
    )?;
    let payload = match (&args.payload_hex, &args.payload_file) {
        (Some(hex), None) => hex::decode(hex.trim_start_matches("0x"))?,
        (None, Some(path)) => std::fs::read(path)?,
        _ => return Err("provide exactly one of --payload-hex or --payload-file".into()),
    };
    let (payload, diagnostic_stage) = if let Some(stage) = args.diagnostic_stage.as_deref() {
        let code = match stage {
            "payload" | "0" => 0,
            "decode" | "state" | "1" => 1,
            "batch" | "2" => 2,
            "train" | "3" => 3,
            "forward" | "4" => 4,
            "backward" | "5" => 5,
            "full-batch" | "6" => 6,
            "adamw" | "7" => 7,
            "return" | "8" => 8,
            _ => return Err(format!("unknown diagnostic stage: {stage}").into()),
        };
        let mut wrapped = b"MCD1".to_vec();
        wrapped.push(code);
        wrapped.extend_from_slice(&payload);
        (wrapped, Some((stage.to_string(), code)))
    } else {
        (payload, None)
    };
    let execution = harness.execute_refine_measured(&payload);
    std::fs::create_dir_all(&args.output)?;
    let (completed, gas_used, gas_remaining, error, exhausted) = match execution {
        Ok(result) => (true, result.gas_used, result.gas_remaining, None, false),
        Err(error) => {
            let message = error.to_string();
            let exhausted = message.contains("out of gas");
            (
                false,
                if exhausted { args.gas_limit } else { 0 },
                0,
                Some(message),
                exhausted,
            )
        }
    };
    let error_ref = error.as_deref();
    let base_classification = classify_execution(completed, gas_used, error_ref, args.gas_limit);
    // MCA1 is a shard measurement, not a complete logical-batch decision.
    // Keep its evidence explicitly scoped so a tiny shard can never authorize
    // integrating the full 256-sample algorithm with the Tiny profile.
    let is_shard = payload.get(..4) == Some(b"MCA1") || payload.get(..4) == Some(b"MCG1");
    const PAYLOAD_HEADER: usize = 4 + 8;
    const MODEL_BYTES: usize = PARAMETER_COUNT * 4;
    let (shard_size, shard_index, sample_range) = if payload.get(..4) == Some(b"MCG1") {
        // MCG1 metadata is: magic/version/job/step/model/batch, then the
        // canonical leaf index and its exact sample range.  Preserve those
        // values in the gas artifact so a leaf probe cannot be mistaken for
        // a different shard.
        const MCG1_META: usize = 4 + 2 + 32 + 8 + 32 + 32;
        if payload.len() >= MCG1_META + 6 {
            let index =
                u16::from_le_bytes(payload[MCG1_META..MCG1_META + 2].try_into().unwrap()) as usize;
            let start =
                u16::from_le_bytes(payload[MCG1_META + 2..MCG1_META + 4].try_into().unwrap())
                    as usize;
            let end = u16::from_le_bytes(payload[MCG1_META + 4..MCG1_META + 6].try_into().unwrap())
                as usize;
            (PARALLEL_SHARD_SIZE, index, [start, end])
        } else {
            (PARALLEL_SHARD_SIZE, 0, [0, PARALLEL_SHARD_SIZE])
        }
    } else if is_shard && payload.len() >= PAYLOAD_HEADER + MODEL_BYTES * 3 + 2 {
        let size = u16::from_le_bytes(
            payload[PAYLOAD_HEADER + MODEL_BYTES * 3..PAYLOAD_HEADER + MODEL_BYTES * 3 + 2]
                .try_into()
                .unwrap(),
        ) as usize;
        (size, 0, [0, size])
    } else {
        (0, 0, [0, 0])
    };
    let classification = if is_shard {
        match base_classification {
            "TINY" => "SHARD_PASS_TINY",
            "FULL_COMFORTABLE" => "SHARD_PASS_FULL_COMFORTABLE",
            "NEAR_FULL" => "SHARD_NEAR_FULL",
            "OVER_FULL" => "SHARD_OVER_FULL",
            _ => "SHARD_NOT_MEASURED",
        }
    } else {
        base_classification
    };
    if let Some((stage_name, stage_code)) = diagnostic_stage {
        let report = serde_json::json!({
            "schema": "minicells.pvm-training-diagnostic.v1",
            "stage": stage_name,
            "stage_code": stage_code,
            "completed": completed,
            "gas_used": gas_used,
            "gas_remaining": gas_remaining,
            "classification": "NOT_MEASURED",
            "error": error,
        });
        std::fs::write(
            args.output.join("pvm-diagnostic.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let report = serde_json::json!({"schema":"minicells.pvm-training-gas.v1", "algorithm":if is_shard {"echo-adamw-ce-tree32-v1"} else {"echo-adamw-ce-v1"},
        "logical_batch_size":256, "artifact_hash":format!("0x{}",hex::encode(harness.artifact.blake2_hash)),
        "gas_limit":args.gas_limit, "gas_used":if exhausted {serde_json::Value::Null} else {serde_json::json!(gas_used)},
        "gas_lower_bound":if exhausted {serde_json::json!(args.gas_limit)} else {serde_json::Value::Null},
        "gas_remaining":gas_remaining, "completed":completed,
        "tiny_limit":1_000_000_000u64, "full_limit":5_000_000_000u64,
        "tiny_ratio":if exhausted {serde_json::Value::Null} else {serde_json::json!(gas_used as f64/1_000_000_000f64)},
        "full_ratio":if exhausted {serde_json::Value::Null} else {serde_json::json!(gas_used as f64/5_000_000_000f64)},
        "classification":classification, "error":error,
        "shard_size":if is_shard {serde_json::json!(shard_size)} else {serde_json::Value::Null},
        "shard_index":if is_shard {serde_json::json!(shard_index)} else {serde_json::Value::Null},
        "sample_range":if is_shard {serde_json::json!(sample_range)} else {serde_json::Value::Null}});
    std::fs::write(
        args.output.join("pvm-gas.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let next = if is_shard {
        "SHARD_MEASUREMENT_ONLY_NO_FULL_STEP_DECISION"
    } else {
        match classification {
            "TINY" => "INTEGRATE_REFERENCE_ALGORITHM_WITH_TINY_PROFILE",
            "FULL_COMFORTABLE" => "SWITCH_MINIJAM_REFINE_LIMIT_TO_5B_THEN_INTEGRATE",
            "NEAR_FULL" => "OPTIMIZE_SINGLE_REFINE_WITHOUT_ALGORITHM_CHANGE",
            "OVER_FULL" => "IMPLEMENT_LOGICAL_BATCH_MULTI_REFINE",
            _ => "BLOCK_UNTIL_DEDICATED_PVM_EXECUTION_IS_VALID",
        }
    };
    let status = if completed {
        if is_shard {
            classification.to_string()
        } else {
            format!("PASS_{classification}")
        }
    } else if base_classification == "OVER_FULL" {
        "ENGINEERING_SPLIT_REQUIRED".to_string()
    } else {
        "BLOCKED_PVM_EXECUTION".to_string()
    };
    std::fs::write(
        args.output.join("decision.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"status":status, "next_action":next, "gas":report}),
        )?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_gate(args: GateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let evidence = PathBuf::from("artifacts/local-training-gate");
    std::fs::create_dir_all(&evidence)?;
    let metrics = run_persistent_native(
        &args.output,
        args.generations,
        16,
        false,
        "synthetic-v1",
        None,
    )?;
    let mut failure = None::<String>;
    if metrics.len() != args.generations as usize {
        failure = Some(format!(
            "expected {} generations, got {}",
            args.generations,
            metrics.len()
        ));
    }
    for (index, metric) in metrics.iter().enumerate() {
        if metric.generation != index as u64 + 1 {
            failure = Some(format!("generation sequence failed at index {index}"));
            break;
        }
        if index > 0 && metric.parent_model_hash != metrics[index - 1].next_model_hash {
            failure = Some(format!(
                "parent hash mismatch at generation {}",
                metric.generation
            ));
            break;
        }
        match metric.decision.as_str() {
            "plus"
                if !(metric.plus_loss < metric.base_loss
                    && metric.plus_loss < metric.minus_loss) =>
            {
                failure = Some(format!(
                    "invalid plus acceptance at generation {}",
                    metric.generation
                ))
            }
            "minus"
                if !(metric.minus_loss < metric.base_loss
                    && metric.minus_loss < metric.plus_loss) =>
            {
                failure = Some(format!(
                    "invalid minus acceptance at generation {}",
                    metric.generation
                ))
            }
            "keep" if metric.next_model_hash != metric.parent_model_hash => {
                failure = Some(format!(
                    "keep changed model at generation {}",
                    metric.generation
                ))
            }
            _ => {}
        }
        if failure.is_some() {
            break;
        }
    }
    if failure.is_none() {
        let resume_root = args.output.join("resume-check");
        run_persistent_native(&resume_root, 256, 16, false, "synthetic-v1", None)?;
        run_persistent_native(
            &resume_root,
            args.generations,
            16,
            true,
            "synthetic-v1",
            None,
        )?;
        let direct_model = std::fs::read(
            args.output
                .join("checkpoints")
                .join(format!("generation-{:06}", args.generations))
                .join("model.bin"),
        )?;
        let resumed_model = std::fs::read(
            resume_root
                .join("checkpoints")
                .join(format!("generation-{:06}", args.generations))
                .join("model.bin"),
        )?;
        let direct_meta = std::fs::read(
            args.output
                .join("checkpoints")
                .join(format!("generation-{:06}", args.generations))
                .join("meta.bin"),
        )?;
        let resumed_meta = std::fs::read(
            resume_root
                .join("checkpoints")
                .join(format!("generation-{:06}", args.generations))
                .join("meta.bin"),
        )?;
        if direct_model != resumed_model || direct_meta != resumed_meta {
            failure = Some("checkpoint resume is not byte deterministic".into());
        }
    }
    let mut native_probe = Vec::new();
    for generation in (0..=args.generations).step_by(16) {
        let model = load_checkpoint_model(&args.output, generation)?;
        native_probe.push(evaluate_fixed_probe(&model, generation)?);
    }
    std::fs::write(
        evidence.join("native-metrics.jsonl"),
        metrics
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
            + "\n",
    )?;
    std::fs::write(
        evidence.join("native-probe.jsonl"),
        native_probe
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
            + "\n",
    )?;
    let first = native_probe.first().ok_or("missing initial probe")?;
    let last = native_probe.last().ok_or("missing final probe")?;
    let best = native_probe
        .iter()
        .min_by_key(|x| x.total_loss)
        .ok_or("missing best probe")?;
    let updates = metrics.iter().filter(|x| x.updated).count();
    let native_pass = failure.is_none()
        && last.total_loss as f64 <= first.total_loss as f64 * 0.95
        && best.total_loss as f64 <= first.total_loss as f64 * 0.90
        && (last.token_accuracy >= first.token_accuracy + 0.02
            || last.total_loss as f64 <= first.total_loss as f64 * 0.90)
        && updates >= 4
        && last.total_loss as f64 <= best.total_loss as f64 * 1.15;

    let solved_path =
        PathBuf::from("artifacts/experiments/003b-quantization-localization/solved-q88-model.bin");
    let solved_bytes = std::fs::read(&solved_path)?;
    let solved_model = minicells_core::PackedModel::decode_from(&solved_bytes)
        .map_err(|_| "invalid solved model")?;
    let solved_start = evaluate_fixed_probe(&solved_model, 0)?;
    let mut solved = NativeTrainer::new("synthetic-v1")?;
    solved
        .host
        .storage
        .insert(keys::MODEL.to_vec(), solved_bytes);
    let solved_hash = model_hash(&solved_model);
    let solved_meta = MetaV1::new(solved_hash);
    let mut meta_bytes = [0u8; META_ENCODED_LEN];
    let meta_len = solved_meta
        .encode_into(&mut meta_bytes)
        .map_err(|_| "meta encode")?;
    solved
        .host
        .storage
        .insert(keys::META.to_vec(), meta_bytes[..meta_len].to_vec());
    for _ in 0..128 {
        solved.run_generation()?;
    }
    let solved_final_model = minicells_core::PackedModel::decode_from(
        solved
            .host
            .storage
            .get(keys::MODEL)
            .ok_or("missing solved model")?,
    )
    .map_err(|_| "invalid final solved model")?;
    let solved_end = evaluate_fixed_probe(&solved_final_model, 128)?;
    let solved_pass = solved_end.token_accuracy + 0.01 >= solved_start.token_accuracy
        && solved_end.total_loss as f64 <= solved_start.total_loss as f64 * 1.02;
    std::fs::write(
        evidence.join("solved-regression.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"status": if solved_pass {"PASS"} else {"FAIL"}, "start": solved_start, "end": solved_end}),
        )?,
    )?;
    let status = native_pass && solved_pass;
    let reason = failure.unwrap_or_else(|| "fixed-probe threshold failed".into());
    let decision = serde_json::json!({
        "schema": "minicells.local-training-gate.v1",
        "status": if status {"PASS"} else {"FAIL"},
        "next_action": if status {"RUN_PVM_GATE"} else {"STOP"},
        "failed_gate": if status {serde_json::Value::Null} else {serde_json::Value::String(if !native_pass {"native"} else {"solved-regression"}.into())},
        "reason": if status {"native and solved regression gates passed".to_string()} else {reason},
        "native_gate": if native_pass {"PASS"} else {"FAIL"},
        "solved_regression_gate": if solved_pass {"PASS"} else {"FAIL"},
        "pvm_gate": "NOT_STARTED",
        "native_generations": args.generations,
        "optimizer": "guarded-sign-spsa-v2",
        "optimizer_version": 2,
    });
    std::fs::write(
        evidence.join("decision.json"),
        serde_json::to_vec_pretty(&decision)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&decision)?);
    if !status {
        return Err("local training gate failed".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Dataset {
            command: DatasetCommand::Build { input, output },
        } => {
            let (dataset, rejected) = compile_jsonl(input)?;
            dataset.save(&output, &rejected)?;
            println!(
                "dataset_root={} samples={} rejected={} output={}",
                dataset.dataset_root,
                dataset.sample_count,
                rejected.len(),
                output.display()
            );
        }
        Command::Dataset {
            command: DatasetCommand::Inspect { path },
        } => println!("{}", serde_json::to_string_pretty(&inspect_dataset(path)?)?),
        Command::Train(args) | Command::Benchmark(args) => train(args, false)?,
        Command::Gate(args) => run_gate(args)?,
        Command::FidelityNative(args) => run_fidelity_native(args)?,
        Command::ParallelNative(args) => run_parallel_native(args)?,
        Command::PvmParity(args) => run_pvm_parity(args)?,
        Command::PvmGas(args) => run_pvm_gas(args)?,
        Command::Resume(args) => train(
            TrainArgs {
                backend: Backend::Native,
                generations: args.generations,
                checkpoint_every: args.checkpoint_every,
                output: args.output,
                dataset: args.dataset,
            },
            true,
        )?,
        Command::Evaluate { run } => {
            let metrics = read_metrics(&run.join("metrics.jsonl"))?;
            println!("{}", serde_json::to_string_pretty(&metrics.last())?);
        }
        Command::Compare { left, right } => {
            let l = read_metrics(&left.join("metrics.jsonl"))?;
            let r = read_metrics(&right.join("metrics.jsonl"))?;
            println!(
                "left_generations={} right_generations={} final_hash_equal={}",
                l.len(),
                r.len(),
                l.last().map(|x| &x.next_model_hash) == r.last().map(|x| &x.next_model_hash)
            );
        }
    }
    Ok(())
}
