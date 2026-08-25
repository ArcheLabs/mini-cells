use clap::{Args, Parser, Subcommand, ValueEnum};
use minicells_dataset::{compile_jsonl, inspect as inspect_dataset, Dataset};
use minicells_protocol::{Op, WorkBody, WorkPayload};
use minicells_pvm::DirectPvmHarness;
use minicells_sim::trainer::{read_metrics, run_persistent_native};
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

fn dataset_root(path: &Option<PathBuf>) -> Result<String, Box<dyn std::error::Error>> {
    Ok(match path {
        Some(path) => Dataset::load(path)?.dataset_root,
        None => "synthetic-v1".into(),
    })
}

fn train(args: TrainArgs, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    if matches!(args.backend, Backend::Pvm) {
        // Jambda's predecoder consumes the converted JAM service blob.  The
        // sibling `.pvm` file is the raw PolkaVM image and is not this API's
        // outer image format.
        let artifact = PathBuf::from("service/artifacts/service.blob");
        let mut harness = DirectPvmHarness::load(
            &artifact,
            10_000_000,
            "5947c50699863948c51028bc346980481d839884",
            "e52307a726868205a151e6917a0a70a79965a028",
        )?;
        let work = WorkPayload {
            op: Op::StatusProbe,
            flags: 0,
            request_id: 1,
            body: WorkBody::StatusProbe,
        };
        let mut payload = [0u8; 96];
        let payload_len = work
            .encode_into(&mut payload)
            .map_err(|error| format!("work payload encode failed: {error:?}"))?;
        let output = harness.execute_refine(&payload[..payload_len])?;
        println!(
            "backend=pvm output_bytes={} output_hex={}",
            output.len(),
            hex::encode(output)
        );
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
