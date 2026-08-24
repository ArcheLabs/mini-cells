use base64::{engine::general_purpose::STANDARD, Engine};
use blake2b_simd::Params;
use clap::{Parser, Subcommand};
use minicells_protocol::{keys, InferenceV1, MetaV1, Op, PendingV1, WorkBody, WorkPayload};
use reqwest::blocking::Client;
use schnorrkel::{signing_context, ExpansionMode, Keypair, MiniSecretKey};
use serde_json::{json, Value};
use std::{
    env, fs, thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(
    name = "minicells",
    about = "MiniJAM scheduler for canonical MINI Cells Work"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Deploy {
        #[arg(long)]
        artifact: String,
    },
    Status,
    StatusProbe,
    Infer {
        text: String,
    },
    TrainOne,
    Train {
        #[arg(long)]
        generations: u64,
    },
    ReplayTrain {
        #[arg(long)]
        generation: u64,
        #[arg(long)]
        parent_model_hash: String,
        #[arg(long,value_parser=["plus","minus"])]
        side: String,
    },
    Watch,
}
struct Api {
    base: String,
    service_id: Option<u32>,
    signer: Keypair,
    http: Client,
}
fn hash(bytes: &[u8]) -> [u8; 32] {
    let h = Params::new().hash_length(32).hash(bytes);
    let mut o = [0; 32];
    o.copy_from_slice(h.as_bytes());
    o
}
fn hex(bytes: &[u8]) -> String {
    format!("0x{}", hex::encode(bytes))
}
fn decode_hex<const N: usize>(text: &str) -> Result<[u8; N], String> {
    let bytes = hex::decode(text.trim_start_matches("0x")).map_err(|e| e.to_string())?;
    bytes.try_into().map_err(|_| format!("expected {N} bytes"))
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
fn request_id() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}
fn canonical_hash(value: &Value) -> [u8; 32] {
    hash(serde_json::to_string(value).unwrap().as_bytes())
}
impl Api {
    fn from_env() -> Result<Self, String> {
        let base = env::var("MINICELLS_PLAYGROUND_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:18080".into());
        let service_id = env::var("MINICELLS_SERVICE_ID")
            .ok()
            .and_then(|v| v.parse().ok());
        let uri = env::var("MINICELLS_SIGNER_URI")
            .map_err(|_| "MINICELLS_SIGNER_URI must be a 0x-prefixed 32-byte seed".to_string())?;
        let seed = decode_hex::<32>(&uri)?;
        let signer = MiniSecretKey::from_bytes(&seed)
            .map_err(|e| e.to_string())?
            .expand_to_keypair(ExpansionMode::Ed25519);
        Ok(Self {
            base: base.trim_end_matches('/').into(),
            service_id,
            signer,
            http: Client::new(),
        })
    }
    fn get(&self, path: &str) -> Result<Value, String> {
        let response = self
            .http
            .get(format!("{}{}", self.base, path))
            .send()
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let body = response.text().map_err(|e| e.to_string())?;
        if !status.is_success() {
            Err(format!("GET {path}: {status}: {body}"))
        } else {
            serde_json::from_str(&body).map_err(|e| e.to_string())
        }
    }
    fn signed_post(&self, path: &str, action: &str, params: Value) -> Result<Value, String> {
        let expiry = now() + 120;
        let prepared=self.http.post(format!("{}/api/v1/actions/prepare",self.base)).json(&json!({"account":hex(&self.signer.public.to_bytes()),"action":action,"paramsHash":hex(&canonical_hash(&params)),"expiry":expiry})).send().map_err(|e|e.to_string())?.error_for_status().map_err(|e|e.to_string())?.json::<Value>().map_err(|e|e.to_string())?;
        let signing = decode_hex::<32>(
            prepared["signingPayload"]
                .as_str()
                .ok_or("missing signingPayload")?,
        )?;
        let mut wrapped = Vec::with_capacity(46);
        wrapped.extend_from_slice(b"<Bytes>");
        wrapped.extend_from_slice(&signing);
        wrapped.extend_from_slice(b"</Bytes>");
        let signature = self
            .signer
            .sign(signing_context(b"substrate").bytes(&wrapped));
        let mut body = params;
        body.as_object_mut()
            .ok_or("params must be an object")?
            .insert(
                "authorization".into(),
                json!({"actionId":prepared["actionId"],"signature":hex(&signature.to_bytes())}),
            );
        let response = self
            .http
            .post(format!("{}{}", self.base, path))
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let text = response.text().map_err(|e| e.to_string())?;
        if !status.is_success() {
            Err(format!("POST {path}: {status}: {text}"))
        } else {
            serde_json::from_str(&text).map_err(|e| e.to_string())
        }
    }
    fn service(&self) -> Result<u32, String> {
        self.service_id
            .ok_or("MINICELLS_SERVICE_ID is required".into())
    }
    fn service_view(&self) -> Result<Value, String> {
        self.get(&format!("/api/v1/services/{}", self.service()?))
    }
    fn storage(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let encoded = hex(key);
        let value = self.get(&format!(
            "/api/v1/services/{}/storage?key={encoded}",
            self.service()?
        ))?;
        match value["value"].as_str() {
            Some(v) => Ok(Some(decode_hex_vec(v)?)),
            None => Ok(None),
        }
    }
    fn meta(&self) -> Result<MetaV1, String> {
        let value = self
            .storage(keys::META)?
            .ok_or("Service is not initialized")?;
        MetaV1::decode(&value).map_err(|_| "invalid MetaV1".into())
    }
    fn wait(&self, operation: &Value) -> Result<Value, String> {
        let id = operation["operationId"]
            .as_str()
            .ok_or("operationId missing")?;
        for _ in 0..1800 {
            let value = self.get(&format!("/api/v1/operations/{id}"))?;
            match value["status"].as_str() {
                Some("succeeded") => return Ok(value),
                Some("failed") => return Err(format!("operation failed: {}", value["error"])),
                _ => thread::sleep(Duration::from_secs(1)),
            }
        }
        Err("operation polling timed out after 30 minutes".into())
    }
    fn submit_work(&self, payload: &[u8]) -> Result<Value, String> {
        let service = self.service_view()?;
        let meta = self.storage(keys::META)?;
        let model = self.storage(keys::MODEL)?;
        let extrinsics = match (meta, model) {
            (Some(meta), Some(model)) => vec![STANDARD.encode(meta), STANDARD.encode(model)],
            (None, None) => Vec::new(),
            _ => return Err("finalized MINI Cells state is incomplete".into()),
        };
        let params = json!({"serviceId":self.service()?,"serviceCodeHash":service["codeHash"],"payloadBase64":STANDARD.encode(payload),"extrinsicsBase64":extrinsics});
        self.signed_post("/api/v1/work", "work", params)
    }
}
fn decode_hex_vec(text: &str) -> Result<Vec<u8>, String> {
    hex::decode(text.trim_start_matches("0x")).map_err(|e| e.to_string())
}
fn encoded(work: WorkPayload) -> Vec<u8> {
    let mut b = [0; 96];
    let n = work.encode_into(&mut b).unwrap();
    b[..n].to_vec()
}
fn print_status(api: &Api) -> Result<(), String> {
    let meta = api.meta()?;
    println!(
        "MINI Cells\nCapability: ECHO\nGeneration: {}\nParameters: 4,476\nModel hash: {}",
        meta.generation,
        hex(&meta.model_hash)
    );
    for (name, key) in [("PLUS", keys::PENDING_PLUS), ("MINUS", keys::PENDING_MINUS)] {
        match api.storage(key)? {
            Some(v) => {
                let p = PendingV1::decode(&v).map_err(|_| "invalid pending record")?;
                println!(
                    "Pending {name}: loss={} correct={}/{}",
                    p.loss, p.correct, p.tokens
                )
            }
            None => println!("Pending {name}: none"),
        }
    }
    Ok(())
}
fn infer(api: &Api, text: String) -> Result<(), String> {
    let normalized = text.to_lowercase();
    let bytes = normalized.as_bytes();
    if bytes.len() > 32 {
        return Err("Echo input exceeds 32 bytes".into());
    }
    let mut body = [0; 32];
    body[..bytes.len()].copy_from_slice(bytes);
    let id = request_id();
    let op = api.submit_work(&encoded(WorkPayload {
        op: Op::Infer,
        flags: 0,
        request_id: id,
        body: WorkBody::Infer {
            expected_generation: u64::MAX,
            text_len: bytes.len() as u8,
            text: body,
        },
    }))?;
    api.wait(&op)?;
    let key = keys::inference_key(keys::inference_slot(id));
    for _ in 0..300 {
        if let Some(raw) = api.storage(&key)? {
            let value = InferenceV1::decode(&raw).map_err(|_| "invalid InferenceV1")?;
            if value.request_id == id {
                let input = String::from_utf8_lossy(&value.input[..value.input_len as usize]);
                let output = String::from_utf8_lossy(&value.output[..value.output_len as usize]);
                let similarity = if value.output_len == 0 {
                    0.0
                } else {
                    f64::from(value.matching_tokens) / f64::from(value.output_len)
                };
                println!("Input: {input}\nPrediction: {output}\nSimilarity: {similarity:.3}\nGeneration: {}\nModel hash: {}",value.generation,hex(&value.model_hash));
                return Ok(());
            }
        }
        thread::sleep(Duration::from_secs(1))
    }
    Err("inference result did not appear before timeout".into())
}
fn probe(api: &Api) -> Result<(), String> {
    let op = api.submit_work(&encoded(WorkPayload {
        op: Op::StatusProbe,
        flags: 0,
        request_id: request_id(),
        body: WorkBody::StatusProbe,
    }))?;
    let done = api.wait(&op)?;
    println!("Status probe finalized: {}", done["operationId"]);
    Ok(())
}
fn replay_train(
    api: &Api,
    generation: u64,
    parent_model_hash: String,
    side: String,
) -> Result<(), String> {
    let op = if side == "plus" {
        Op::TrainPlus
    } else {
        Op::TrainMinus
    };
    let submitted = api.submit_work(&encoded(WorkPayload {
        op,
        flags: 0,
        request_id: request_id(),
        body: WorkBody::Train {
            generation,
            parent_model_hash: decode_hex::<32>(&parent_model_hash)?,
        },
    }))?;
    let done = api.wait(&submitted)?;
    println!(
        "Stale replay Work finalized without changing canonical state: {}",
        done["operationId"]
    );
    Ok(())
}
fn train_one(api: &Api) -> Result<(), String> {
    let meta = api.meta()?;
    for (op, key) in [
        (Op::TrainPlus, keys::PENDING_PLUS),
        (Op::TrainMinus, keys::PENDING_MINUS),
    ] {
        if api.storage(key)?.is_none() {
            let work = WorkPayload {
                op,
                flags: 0,
                request_id: request_id(),
                body: WorkBody::Train {
                    generation: meta.generation,
                    parent_model_hash: meta.model_hash,
                },
            };
            let submitted = api.submit_work(&encoded(work))?;
            api.wait(&submitted)?;
        }
    }
    for _ in 0..120 {
        let current = api.meta()?;
        if current.generation > meta.generation {
            println!(
                "Generation {} finalized; model {}",
                current.generation,
                hex(&current.model_hash)
            );
            return Ok(());
        }
        thread::sleep(Duration::from_secs(1))
    }
    Err("generation did not advance before timeout".into())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1)
    }
}
fn run() -> Result<(), String> {
    let args = Args::parse();
    let api = Api::from_env()?;
    match args.command {
        Command::Deploy { artifact } => {
            let blob = fs::read(artifact).map_err(|e| e.to_string())?;
            let params = json!({"blobBase64":STANDARD.encode(&blob),"codeHash":hex(&hash(&blob)),"minItemGas":20_000_000u64,"minMemoGas":1u64});
            let op = api.signed_post("/api/v1/services", "create_service", params)?;
            let done = api.wait(&op)?;
            println!("Service deployed: {}", done["result"]["serviceId"]);
        }
        Command::Status => print_status(&api)?,
        Command::StatusProbe => probe(&api)?,
        Command::Infer { text } => infer(&api, text)?,
        Command::TrainOne => train_one(&api)?,
        Command::Train { generations } => {
            for _ in 0..generations {
                train_one(&api)?
            }
        }
        Command::ReplayTrain {
            generation,
            parent_model_hash,
            side,
        } => replay_train(&api, generation, parent_model_hash, side)?,
        Command::Watch => loop {
            print_status(&api)?;
            thread::sleep(Duration::from_secs(6));
        },
    }
    Ok(())
}
