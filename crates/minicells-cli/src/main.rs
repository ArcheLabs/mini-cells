use clap::{Parser, Subcommand};
use minicells_chain::{FilesystemBulletinStore, MiniCellsChain, TrainSide};
use minicells_protocol::MetaV1;
use std::{
    env, fs,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(name = "minicells", about = "Direct MiniJAM MINI Cells client")]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Deploy {
        artifact: String,
    },
    Status,
    StatusProbe,
    Infer {
        text: String,
    },
    TrainOne,
    Train {
        generations: u64,
    },
    ReplayTrain {
        generation: u64,
        parent_model_hash: String,
        #[arg(long,value_parser=["plus","minus"])]
        side: String,
    },
    Watch,
}
fn seed() -> Result<[u8; 32], String> {
    let value = env::var("MINICELLS_KEEPER_SIGNER_URI")
        .or_else(|_| env::var("MINICELLS_SIGNER_URI"))
        .map_err(|_| "MINICELLS_KEEPER_SIGNER_URI must be a 32-byte hex seed".to_string())?;
    let bytes = hex::decode(value.trim_start_matches("0x")).map_err(|e| e.to_string())?;
    bytes
        .try_into()
        .map_err(|_| "signer seed must be exactly 32 bytes".into())
}
fn id() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
async fn chain() -> Result<MiniCellsChain<FilesystemBulletinStore>, String> {
    let rpc = env::var("MINICELLS_RPC_URL").unwrap_or_else(|_| "ws://127.0.0.1:9944".into());
    let service = env::var("MINICELLS_SERVICE_ID")
        .map_err(|_| "MINICELLS_SERVICE_ID is required".to_string())?
        .parse()
        .map_err(|_| "invalid service id".to_string())?;
    let root =
        env::var("MINICELLS_BULLETIN_DIR").unwrap_or_else(|_| ".local/minicells-bulletin".into());
    let bulletin = Arc::new(FilesystemBulletinStore::new(root).map_err(|e| e.to_string())?);
    MiniCellsChain::connect(rpc, seed()?, service, bulletin)
        .await
        .map_err(|e| e.to_string())
}
fn print_state(meta: Option<&MetaV1>, plus: bool, minus: bool) {
    if let Some(m) = meta {
        println!(
            "MINI Cells\nGeneration: {}\nModel hash: 0x{}\nPending PLUS: {}\nPending MINUS: {}",
            m.generation,
            hex::encode(m.model_hash),
            plus,
            minus
        )
    } else {
        println!("MINI Cells is not initialized")
    }
}
#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1)
    }
}
async fn run() -> Result<(), String> {
    let args = Args::parse();
    let chain = chain().await?;
    match args.command {
        Command::Deploy { artifact } => {
            let bytes = fs::read(artifact).map_err(|e| e.to_string())?;
            let s = chain
                .deploy_service(&bytes, 20_000_000, 1)
                .await
                .map_err(|e| e.to_string())?;
            println!(
                "Service deployment submitted: 0x{}",
                hex::encode(s.extrinsic_hash)
            );
        }
        Command::Status => {
            let s = chain.finalized_state().await.map_err(|e| e.to_string())?;
            print_state(
                s.meta.as_ref(),
                s.pending_plus.is_some(),
                s.pending_minus.is_some(),
            );
        }
        Command::StatusProbe => {
            let s = chain
                .submit_status_probe(id())
                .await
                .map_err(|e| e.to_string())?;
            chain.wait_work(&s).await.map_err(|e| e.to_string())?;
            println!("Status probe finalized");
        }
        Command::Infer { text } => {
            let text = text.to_lowercase();
            let s = chain
                .submit_infer(id(), text.as_bytes())
                .await
                .map_err(|e| e.to_string())?;
            chain.wait_work(&s).await.map_err(|e| e.to_string())?;
            println!(
                "Inference Work finalized: 0x{}",
                hex::encode(s.package_hash)
            );
        }
        Command::TrainOne => run_train(&chain).await?,
        Command::Train { generations } => {
            for _ in 0..generations {
                run_train(&chain).await?
            }
        }
        Command::ReplayTrain {
            generation,
            parent_model_hash,
            side,
        } => {
            let parent = hex::decode(parent_model_hash.trim_start_matches("0x"))
                .map_err(|e| e.to_string())?;
            let parent: [u8; 32] = parent
                .try_into()
                .map_err(|_| "parent hash must be 32 bytes")?;
            let s = chain
                .submit_train_side(
                    if side == "plus" {
                        TrainSide::Plus
                    } else {
                        TrainSide::Minus
                    },
                    generation,
                    parent,
                    id(),
                )
                .await
                .map_err(|e| e.to_string())?;
            chain.wait_work(&s).await.map_err(|e| e.to_string())?;
            println!("Replay Work finalized");
        }
        Command::Watch => loop {
            let s = chain.finalized_state().await.map_err(|e| e.to_string())?;
            print_state(
                s.meta.as_ref(),
                s.pending_plus.is_some(),
                s.pending_minus.is_some(),
            );
            tokio::time::sleep(std::time::Duration::from_secs(6)).await;
        },
    }
    Ok(())
}
async fn run_train(chain: &MiniCellsChain<FilesystemBulletinStore>) -> Result<(), String> {
    let s = chain.finalized_state().await.map_err(|e| e.to_string())?;
    let m = s.meta.ok_or("service is not initialized")?;
    for (side, pending) in [
        (TrainSide::Plus, s.pending_plus),
        (TrainSide::Minus, s.pending_minus),
    ] {
        if pending.is_none() {
            let w = chain
                .submit_train_side(side, m.generation, m.model_hash, id())
                .await
                .map_err(|e| e.to_string())?;
            chain.wait_work(&w).await.map_err(|e| e.to_string())?;
        }
    }
    chain
        .wait_generation_after(m.generation)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}
