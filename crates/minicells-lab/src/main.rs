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
