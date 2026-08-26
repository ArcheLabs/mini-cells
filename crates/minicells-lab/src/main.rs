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
    evaluate_batch, train_step_with_gradient, TrainStepReport, TrainingBatch, TrainingState,
    PARAMETER_COUNT,
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
struct PvmGasArgs {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long, default_value_t = 10_000_000_000)]
    gas_limit: u64,
    #[arg(long, conflicts_with = "payload_file")]
    payload_hex: Option<String>,
    #[arg(long, conflicts_with = "payload_hex")]
    payload_file: Option<PathBuf>,
    #[arg(long, default_value = "artifacts/pvm-algorithm-fidelity")]
    output: PathBuf,
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
    for step in 1..=steps {
        let batch = load_fidelity_batch(&args.fixture.join(format!("batch-{step:06}.bin")))?;
        let mut gradient = [0.0f32; PARAMETER_COUNT];
        let report: TrainStepReport = train_step_with_gradient(&mut state, &batch, &mut gradient);
        let bytes = f32_bytes(&state.weights);
        reports.push(
            serde_json::json!({"step": step, "loss": report.loss, "token_count": report.token_count,
            "grad_norm": report.grad_norm, "weight_digest": digest_bytes(&bytes)}),
        );
        if step <= 2 {
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
        for batch in &validation_batches {
            let report = evaluate_batch(&state, batch);
            loss_sum += report.loss * report.token_count as f32;
            token_count += report.token_count as u64;
        }
        serde_json::json!({"status":"PASS", "batches":validation_batches.len(), "token_count":token_count,
            "loss":if token_count == 0 { 0.0 } else { loss_sum / token_count as f32 }})
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
    let execution = harness.execute_refine_measured(&payload);
    std::fs::create_dir_all(&args.output)?;
    let (completed, gas_used, gas_remaining, error) = match execution {
        Ok(result) => (true, result.gas_used, result.gas_remaining, None),
        Err(error) => (false, 0, 0, Some(error.to_string())),
    };
    let error_ref = error.as_deref();
    let classification = classify_execution(completed, gas_used, error_ref, args.gas_limit);
    let report = serde_json::json!({"schema":"minicells.pvm-training-gas.v1", "algorithm":"echo-adamw-ce-v1",
        "logical_batch_size":256, "artifact_hash":format!("0x{}",hex::encode(harness.artifact.blake2_hash)),
        "gas_limit":args.gas_limit, "gas_used":gas_used, "gas_remaining":gas_remaining, "completed":completed,
        "tiny_limit":1_000_000_000u64, "full_limit":5_000_000_000u64,
        "tiny_ratio":gas_used as f64/1_000_000_000f64, "full_ratio":gas_used as f64/5_000_000_000f64,
        "classification":classification, "error":error});
    std::fs::write(
        args.output.join("pvm-gas.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let next = match classification {
        "TINY" => "INTEGRATE_REFERENCE_ALGORITHM_WITH_TINY_PROFILE",
        "FULL_COMFORTABLE" => "SWITCH_MINIJAM_REFINE_LIMIT_TO_5B_THEN_INTEGRATE",
        "NEAR_FULL" => "OPTIMIZE_SINGLE_REFINE_WITHOUT_ALGORITHM_CHANGE",
        "OVER_FULL" => "IMPLEMENT_LOGICAL_BATCH_MULTI_REFINE",
        _ => "BLOCK_UNTIL_DEDICATED_PVM_EXECUTION_IS_VALID",
    };
    let status = if completed {
        format!("PASS_{classification}")
    } else if classification == "OVER_FULL" {
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
