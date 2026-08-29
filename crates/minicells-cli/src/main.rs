use clap::{Parser, Subcommand};
use minicells_chain::{
    DeployedService, FilesystemBulletinStore, FinalizedMiniCellsState, MiniCellsChain, TrainSide,
};
use minicells_protocol::MetaV1;
use sp_core::{sr25519, Pair};
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
    #[command(about = "Debug-only stale training replay")]
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
    if let Ok(bytes) = hex::decode(value.trim_start_matches("0x")) {
        return bytes
            .try_into()
            .map_err(|_| "signer seed must be exactly 32 bytes".into());
    }
    let (_, seed) = sr25519::Pair::from_string_with_seed(&value, None)
        .map_err(|e| format!("invalid signer URI: {e}"))?;
    seed.ok_or_else(|| "signer URI did not expose a reproducible raw seed".into())
}
fn id() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
async fn chain_for_service(
    service: u32,
) -> Result<MiniCellsChain<FilesystemBulletinStore>, String> {
    let rpc = env::var("MINICELLS_RPC_URL").unwrap_or_else(|_| "ws://127.0.0.1:9944".into());
    let root =
        env::var("MINICELLS_BULLETIN_DIR").unwrap_or_else(|_| ".local/minicells-bulletin".into());
    let bulletin = Arc::new(FilesystemBulletinStore::new(root).map_err(|e| e.to_string())?);
    MiniCellsChain::connect(rpc, seed()?, service, bulletin)
        .await
        .map_err(|e| e.to_string())
}
async fn chain() -> Result<MiniCellsChain<FilesystemBulletinStore>, String> {
    let service = env::var("MINICELLS_SERVICE_ID")
        .map_err(|_| "MINICELLS_SERVICE_ID is required".to_string())?
        .parse()
        .map_err(|_| "invalid service id".to_string())?;
    chain_for_service(service).await
}
async fn deploy_chain() -> Result<MiniCellsChain<FilesystemBulletinStore>, String> {
    let service = match env::var("MINICELLS_SERVICE_ID") {
        Ok(value) => value
            .parse()
            .map_err(|_| "invalid service id".to_string())?,
        Err(_) => 0,
    };
    chain_for_service(service).await
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
    let command = Args::parse().command;
    if let Command::Deploy { artifact } = &command {
        let chain = deploy_chain().await?;
        let bytes = fs::read(artifact).map_err(|e| e.to_string())?;
        let service = chain
            .deploy_service_and_wait(&bytes, 20_000_000, 1)
            .await
            .map_err(|e| e.to_string())?;
        print_deployed_service(&service);
        return Ok(());
    }
    let chain = chain().await?;
    match command {
        Command::Deploy { .. } => unreachable!("deploy handled before connecting to a service"),
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
        Command::TrainOne => train_one_generation(&chain).await?,
        Command::Train { generations } => {
            for _ in 0..generations {
                train_one_generation(&chain).await?
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
fn print_deployed_service(service: &DeployedService) {
    println!("{}", deployed_service_output(service));
}

fn deployed_service_output(service: &DeployedService) -> String {
    format!(
        "Service created\nService ID: {}\nCode hash: 0x{}\nExtrinsic: 0x{}\nCreate correlation: 0x{}\nIncluded block: {}\nExtrinsic index: {}\nSet MINICELLS_SERVICE_ID={} for subsequent commands",
        service.service_id,
        hex::encode(service.code_hash),
        hex::encode(service.create_extrinsic_hash),
        hex::encode(service.create_correlation),
        service
            .create_included_block
            .map(|hash| format!("0x{}", hex::encode(hash)))
            .unwrap_or_else(|| "unknown".into()),
        service
            .create_extrinsic_index
            .map(|index| index.to_string())
            .unwrap_or_else(|| "unknown".into()),
        service.service_id
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TrainingAction {
    SubmitPlus {
        generation: u64,
        model_hash: [u8; 32],
    },
    SubmitMinus {
        generation: u64,
        model_hash: [u8; 32],
    },
    Wait,
    Complete,
}

fn decide_training_action(
    start_generation: u64,
    state: &FinalizedMiniCellsState,
) -> Result<TrainingAction, String> {
    let meta = state.meta.as_ref().ok_or("service is not initialized")?;
    if meta.generation > start_generation {
        return Ok(TrainingAction::Complete);
    }
    if state.pending_plus.is_none() {
        return Ok(TrainingAction::SubmitPlus {
            generation: meta.generation,
            model_hash: meta.model_hash,
        });
    }
    if state.pending_minus.is_none() {
        return Ok(TrainingAction::SubmitMinus {
            generation: meta.generation,
            model_hash: meta.model_hash,
        });
    }
    Ok(TrainingAction::Wait)
}

async fn train_one_generation(
    chain: &MiniCellsChain<FilesystemBulletinStore>,
) -> Result<(), String> {
    let initial = chain.finalized_state().await.map_err(|e| e.to_string())?;
    let start_generation = initial
        .meta
        .as_ref()
        .ok_or("service is not initialized")?
        .generation;
    loop {
        let state = chain.finalized_state().await.map_err(|e| e.to_string())?;
        match decide_training_action(start_generation, &state)? {
            TrainingAction::Complete => return Ok(()),
            TrainingAction::SubmitPlus {
                generation,
                model_hash,
            } => {
                let work = chain
                    .submit_train_side(TrainSide::Plus, generation, model_hash, id())
                    .await
                    .map_err(|e| e.to_string())?;
                chain.wait_work(&work).await.map_err(|e| e.to_string())?;
            }
            TrainingAction::SubmitMinus {
                generation,
                model_hash,
            } => {
                let work = chain
                    .submit_train_side(TrainSide::Minus, generation, model_hash, id())
                    .await
                    .map_err(|e| e.to_string())?;
                chain.wait_work(&work).await.map_err(|e| e.to_string())?;
            }
            TrainingAction::Wait => {
                chain
                    .wait_generation_after(start_generation)
                    .await
                    .map_err(|e| e.to_string())?;
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minicells_protocol::PendingV1;

    fn state(generation: u64, hash: [u8; 32], plus: bool, minus: bool) -> FinalizedMiniCellsState {
        FinalizedMiniCellsState {
            block_hash: [0; 32],
            block_number: 0,
            state_root: [0; 32],
            lookup_anchor_slot: 0,
            meta: Some(MetaV1 {
                generation,
                model_hash: hash,
                ..MetaV1::new(hash)
            }),
            model: None,
            pending_plus: plus.then_some(PendingV1 {
                generation,
                parent_hash: hash,
                side: 1,
                loss: 0,
                correct: 0,
                tokens: 0,
                digest: [0; 32],
            }),
            pending_minus: minus.then_some(PendingV1 {
                generation,
                parent_hash: hash,
                side: -1,
                loss: 0,
                correct: 0,
                tokens: 0,
                digest: [0; 32],
            }),
            history: Vec::new(),
            inferences: Vec::new(),
        }
    }

    #[test]
    fn training_decision_handles_all_pending_shapes() {
        assert_eq!(
            decide_training_action(0, &state(0, [1; 32], false, false)).unwrap(),
            TrainingAction::SubmitPlus {
                generation: 0,
                model_hash: [1; 32]
            }
        );
        assert_eq!(
            decide_training_action(0, &state(0, [2; 32], true, false)).unwrap(),
            TrainingAction::SubmitMinus {
                generation: 0,
                model_hash: [2; 32]
            }
        );
        assert_eq!(
            decide_training_action(0, &state(0, [3; 32], false, true)).unwrap(),
            TrainingAction::SubmitPlus {
                generation: 0,
                model_hash: [3; 32]
            }
        );
        assert_eq!(
            decide_training_action(0, &state(0, [4; 32], true, true)).unwrap(),
            TrainingAction::Wait
        );
        assert_eq!(
            decide_training_action(0, &state(1, [5; 32], true, true)).unwrap(),
            TrainingAction::Complete
        );
    }

    #[test]
    fn training_decision_uses_current_meta_hash_after_refresh() {
        let action = decide_training_action(4, &state(4, [9; 32], true, false)).unwrap();
        assert_eq!(
            action,
            TrainingAction::SubmitMinus {
                generation: 4,
                model_hash: [9; 32]
            }
        );
    }

    #[test]
    fn deploy_output_uses_receipt_service_id() {
        let output = deployed_service_output(&DeployedService {
            service_id: 42,
            code_hash: [2; 32],
            create_extrinsic_hash: [3; 32],
            create_correlation: [4; 32],
            create_included_block: Some([5; 32]),
            create_extrinsic_index: Some(0),
        });
        assert!(output.contains("Service ID: 42"));
        assert!(output.contains("Set MINICELLS_SERVICE_ID=42"));
    }
}
