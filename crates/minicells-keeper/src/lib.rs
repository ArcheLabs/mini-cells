//! Non-canonical Keeper boundary for MINI Cells.

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use minicells_chain::{FilesystemBulletinStore, FinalizedMiniCellsState, MiniCellsChain};
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer};

pub type Chain = MiniCellsChain<FilesystemBulletinStore>;
#[derive(Clone, Debug, Serialize)]
pub struct KeeperSnapshot {
    pub startup_id: String,
    pub training_enabled: bool,
    pub phase: String,
    pub chain: ChainSnapshot,
    pub last_error: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ChainSnapshot {
    pub block_number: u32,
    pub block_hash: String,
    pub state_root: String,
    pub generation: Option<u64>,
    pub model_hash: Option<String>,
    pub pending_plus: bool,
    pub pending_minus: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct HistoryEntry {
    pub generation: u64,
    pub model_hash: String,
    pub plus_loss: i64,
    pub minus_loss: i64,
    pub tokens: u32,
}
#[derive(Clone, Debug, Deserialize)]
pub struct InferRequest {
    pub text: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct InferAccepted {
    pub request_id: u64,
    pub status: &'static str,
    pub package_hash: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct InferStatus {
    pub request_id: u64,
    pub status: String,
    pub error: Option<String>,
}
#[derive(Clone)]
struct AppState {
    keeper: Arc<Keeper>,
}

pub struct Keeper {
    pub chain: Arc<Chain>,
    snapshot: Mutex<KeeperSnapshot>,
    events: broadcast::Sender<String>,
    inferences: Arc<Mutex<std::collections::BTreeMap<u64, InferStatus>>>,
    next_request: std::sync::atomic::AtomicU64,
}
impl Keeper {
    pub fn new(chain: Arc<Chain>, training_enabled: bool) -> Arc<Self> {
        let (events, _) = broadcast::channel(256);
        Arc::new(Self {
            chain,
            snapshot: Mutex::new(KeeperSnapshot {
                startup_id: format!("{}-{}", std::process::id(), now()),
                training_enabled,
                phase: "starting".into(),
                chain: ChainSnapshot {
                    block_number: 0,
                    block_hash: String::new(),
                    state_root: String::new(),
                    generation: None,
                    model_hash: None,
                    pending_plus: false,
                    pending_minus: false,
                },
                last_error: None,
            }),
            events,
            inferences: Arc::new(Mutex::new(Default::default())),
            next_request: std::sync::atomic::AtomicU64::new(now()),
        })
    }
    pub fn router(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/healthz", get(healthz))
            .route("/v1/status", get(status))
            .route("/v1/history", get(history))
            .route("/v1/events", get(events))
            .route("/v1/training", post(training))
            .route("/v1/infer", post(infer))
            .route("/v1/infer/{id}", get(infer_status))
            .route("/ipfs/{cid}", get(bundle))
            .layer(CorsLayer::permissive())
            .layer(RequestBodyLimitLayer::new(1024 * 1024))
            .with_state(AppState {
                keeper: Arc::clone(self),
            })
    }
    pub async fn refresh(&self) -> Result<(), String> {
        let state = self
            .chain
            .finalized_state()
            .await
            .map_err(|e| e.to_string())?;
        let mut old = self.snapshot.lock().unwrap().clone();
        old.chain = snapshot_chain(&state);
        old.phase = "watching".into();
        old.last_error = None;
        *self.snapshot.lock().unwrap() = old.clone();
        self.emit("snapshot", &old);
        Ok(())
    }
    pub fn start(self: &Arc<Self>) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut delay = std::time::Duration::from_secs(1);
            loop {
                match this.refresh().await {
                    Ok(()) => {
                        delay = std::time::Duration::from_secs(1);
                        if this.snapshot.lock().unwrap().training_enabled {
                            if let Err(error) = this.schedule_once().await {
                                this.snapshot.lock().unwrap().last_error = Some(error);
                            }
                        }
                    }
                    Err(error) => {
                        this.snapshot.lock().unwrap().last_error = Some(error);
                        delay = (delay * 2).min(std::time::Duration::from_secs(30));
                    }
                }
                tokio::time::sleep(delay).await;
            }
        });
    }
    async fn schedule_once(&self) -> Result<(), String> {
        let state = self
            .chain
            .finalized_state()
            .await
            .map_err(|e| e.to_string())?;
        let meta = match state.meta {
            Some(value) => value,
            None => return Ok(()),
        };
        let (side, pending) = if state.pending_plus.is_none() {
            (minicells_chain::TrainSide::Plus, state.pending_plus)
        } else if state.pending_minus.is_none() {
            (minicells_chain::TrainSide::Minus, state.pending_minus)
        } else {
            return Ok(());
        };
        if pending.is_none() {
            let request_id = self
                .next_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let work = self
                .chain
                .submit_train_side(side, meta.generation, meta.model_hash, request_id)
                .await
                .map_err(|e| e.to_string())?;
            self.snapshot.lock().unwrap().phase = format!("training-{}", side.as_str());
            self.chain
                .wait_work(&work)
                .await
                .map_err(|e| e.to_string())?;
            self.emit("training",&serde_json::json!({"side":side.as_str(),"generation":meta.generation,"request_id":request_id}));
        }
        Ok(())
    }
    fn emit<T: Serialize>(&self, kind: &str, value: &T) {
        let _ = self
            .events
            .send(serde_json::to_string(&serde_json::json!({"type":kind,"data":value})).unwrap());
    }
    async fn submit_infer(self: &Arc<Self>, text: &str) -> Result<InferAccepted, String> {
        if !text.is_ascii() || text.len() > 32 || text.chars().any(|c| c.is_ascii_uppercase()) {
            return Err("text must be lowercase ASCII and at most 32 bytes".into());
        }
        let id = self
            .next_request
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let submitted = self
            .chain
            .submit_infer(id, text.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        let package_hash = submitted.package_hash;
        self.inferences.lock().unwrap().insert(
            id,
            InferStatus {
                request_id: id,
                status: "submitted".into(),
                error: None,
            },
        );
        let chain = Arc::clone(&self.chain);
        let events = self.events.clone();
        let records = self.inferences.clone();
        tokio::spawn(async move {
            let result = chain.wait_work(&submitted).await;
            let (status, error): (String, Option<String>) = match result {
                Ok(_) => ("finalized".into(), None),
                Err(e) => ("failed".into(), Some(e.to_string())),
            };
            if let Ok(mut map) = records.lock() {
                if let Some(v) = map.get_mut(&id) {
                    v.status = status.clone();
                    v.error = error.clone()
                }
            }
            let _=events.send(serde_json::to_string(&serde_json::json!({"type":"inference","request_id":id,"status":status,"error":error})).unwrap());
        });
        Ok(InferAccepted {
            request_id: id,
            status: "submitted",
            package_hash: format!("0x{}", hex::encode(package_hash)),
        })
    }
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn snapshot_chain(s: &FinalizedMiniCellsState) -> ChainSnapshot {
    ChainSnapshot {
        block_number: s.block_number,
        block_hash: format!("0x{}", hex::encode(s.block_hash)),
        state_root: format!("0x{}", hex::encode(s.state_root)),
        generation: s.meta.as_ref().map(|m| m.generation),
        model_hash: s
            .meta
            .as_ref()
            .map(|m| format!("0x{}", hex::encode(m.model_hash))),
        pending_plus: s.pending_plus.is_some(),
        pending_minus: s.pending_minus.is_some(),
    }
}
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({"ok":true,"service":"minicells-keeper"}))
}
async fn status(State(app): State<AppState>) -> impl IntoResponse {
    Json(app.keeper.snapshot.lock().unwrap().clone())
}
async fn history(State(app): State<AppState>) -> impl IntoResponse {
    match app.keeper.chain.finalized_state().await {
        Ok(s) => Json(
            serde_json::json!({"items":s.history.into_iter().map(|h|HistoryEntry{generation:h.generation,model_hash:format!("0x{}",hex::encode(h.model_hash)),plus_loss:h.plus_loss,minus_loss:h.minus_loss,tokens:h.tokens}).collect::<Vec<_>>()}),
        ),
        Err(e) => Json(serde_json::json!({"error":e.to_string()})),
    }
}
async fn training(
    State(app): State<AppState>,
    Json(value): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(v) = value.get("enabled").and_then(|v| v.as_bool()) {
        app.keeper.snapshot.lock().unwrap().training_enabled = v
    }
    Json(app.keeper.snapshot.lock().unwrap().clone())
}
async fn infer(State(app): State<AppState>, Json(value): Json<InferRequest>) -> Response {
    match app.keeper.submit_infer(&value.text).await {
        Ok(v) => (StatusCode::ACCEPTED, Json(v)).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":e})),
        )
            .into_response(),
    }
}
async fn infer_status(Path(id): Path<u64>, State(app): State<AppState>) -> Response {
    match app.keeper.inferences.lock().unwrap().get(&id).cloned() {
        Some(v) => Json(v).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"unknown request"})),
        )
            .into_response(),
    }
}
async fn events(
    State(app): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let initial = serde_json::to_string(
        &serde_json::json!({"type":"snapshot","data":*app.keeper.snapshot.lock().unwrap()}),
    )
    .unwrap();
    let initial_event = tokio_stream::once(Ok(Event::default().data(initial.clone())));
    let updates = BroadcastStream::new(app.keeper.events.subscribe()).map(move |item| {
        Ok(match item {
            Ok(v) => Event::default().data(v),
            Err(_) => Event::default().event("resync").data(initial.clone()),
        })
    });
    let stream = initial_event.chain(updates);
    Sse::new(stream).keep_alive(KeepAlive::default())
}
async fn bundle(Path(cid): Path<String>, State(app): State<AppState>) -> Response {
    let cid = match cid.parse::<cid::Cid>() {
        Ok(v) => v,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let digest = cid.hash().digest();
    if digest.len() != 32 {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let mut hash = [0; 32];
    hash.copy_from_slice(digest);
    match app.keeper.chain.bulletin_store().read_by_hash(&hash) {
        Ok(v) => ([(header::CONTENT_TYPE, "application/octet-stream")], v).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}
pub async fn serve(
    keeper: Arc<Keeper>,
    address: std::net::SocketAddr,
) -> Result<(), std::io::Error> {
    keeper.start();
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, keeper.router()).await
}
