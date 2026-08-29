//! Non-canonical Keeper boundary for MINI Cells.
//!
//! The Keeper authenticates browser users with a short-lived sr25519 challenge
//! session, serves finalized model bytes, schedules official training, and
//! exposes SSE as a read/update transport. It never becomes canonical state.

use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use minicells_chain::{FilesystemBulletinStore, FinalizedMiniCellsState, MiniCellsChain};
use minicells_core::{model_hash, PackedModel, MODEL_BYTES, PARAMETER_COUNT};
use serde::{Deserialize, Serialize};
use sp_core::{
    crypto::{AccountId32, Ss58Codec},
    sr25519, Pair,
};
use std::{
    collections::BTreeMap,
    convert::Infallible,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer};

pub type Chain = MiniCellsChain<FilesystemBulletinStore>;
const SESSION_COOKIE: &str = "minicells_session";
const CHALLENGE_TTL: Duration = Duration::from_secs(5 * 60);
const SESSION_TTL: Duration = Duration::from_secs(12 * 60 * 60);
const MAX_CHALLENGES: usize = 1024;
const MAX_SESSIONS: usize = 4096;

#[derive(Clone, Debug, Default)]
pub struct AuthConfig {
    pub origin: Option<String>,
    pub operator: Option<[u8; 32]>,
    pub secure_cookie: bool,
}
impl AuthConfig {
    pub fn from_env() -> Self {
        let operator = std::env::var("MINICELLS_OPERATOR_ACCOUNT")
            .ok()
            .and_then(|v| decode_account(&v).ok());
        let origin = std::env::var("MINICELLS_WEB_ORIGIN")
            .ok()
            .filter(|v| !v.trim().is_empty());
        let secure_cookie = match std::env::var("MINICELLS_COOKIE_SECURE") {
            Ok(value) => value == "1" || value.eq_ignore_ascii_case("true"),
            Err(_) => origin
                .as_deref()
                .is_some_and(|value| value.starts_with("https://")),
        };
        Self {
            origin,
            operator,
            secure_cookie,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct KeeperSnapshot {
    pub startup_id: String,
    pub training_enabled: bool,
    pub phase: String,
    pub chain: ChainSnapshot,
    pub last_error: Option<String>,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
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
pub struct InferenceResult {
    pub input: String,
    pub output: String,
    pub generation: u64,
    pub model_hash: String,
    pub similarity: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct InferStatus {
    pub request_id: u64,
    pub status: String,
    pub result: Option<InferenceResult>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct ChallengeQuery {
    pub account: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct ChallengeResponse {
    #[serde(rename = "challengeId")]
    pub challenge_id: String,
    pub account: String,
    pub message: String,
    #[serde(rename = "messageHex")]
    pub message_hex: String,
}
#[derive(Clone, Debug, Deserialize)]
pub struct VerifyRequest {
    #[serde(rename = "challengeId")]
    pub challenge_id: String,
    pub account: String,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct AuthMe {
    pub authenticated: bool,
    pub account: Option<String>,
    #[serde(rename = "isOperator")]
    pub is_operator: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct ModelResponse {
    pub format: &'static str,
    pub generation: u64,
    #[serde(rename = "modelFormat")]
    pub model_format: u16,
    pub capability: &'static str,
    #[serde(rename = "parameterCount")]
    pub parameter_count: usize,
    #[serde(rename = "modelHash")]
    pub model_hash: String,
    pub encoding: &'static str,
    #[serde(rename = "modelBytes")]
    pub model_bytes: String,
}
#[derive(Clone, Debug, Serialize)]
struct WireEvent {
    kind: String,
    data: serde_json::Value,
}
#[derive(Clone)]
struct AppState {
    keeper: Arc<Keeper>,
}

#[derive(Clone)]
struct Challenge {
    account: [u8; 32],
    message: Vec<u8>,
    expires_at: SystemTime,
}
#[derive(Clone)]
struct Session {
    account: [u8; 32],
    expires_at: SystemTime,
    is_operator: bool,
}
#[derive(Default)]
struct AuthStore {
    challenges: BTreeMap<String, Challenge>,
    sessions: BTreeMap<String, Session>,
}

pub struct Keeper {
    pub chain: Arc<Chain>,
    auth: Mutex<AuthStore>,
    auth_config: AuthConfig,
    snapshot: Mutex<KeeperSnapshot>,
    events: broadcast::Sender<WireEvent>,
    inferences: Arc<Mutex<BTreeMap<u64, InferStatus>>>,
    next_request: std::sync::atomic::AtomicU64,
}

impl Keeper {
    pub fn new(chain: Arc<Chain>, training_enabled: bool) -> Arc<Self> {
        Self::new_with_config(chain, training_enabled, AuthConfig::from_env())
    }
    pub fn new_with_config(
        chain: Arc<Chain>,
        training_enabled: bool,
        auth_config: AuthConfig,
    ) -> Arc<Self> {
        let (events, _) = broadcast::channel(256);
        Arc::new(Self {
            chain,
            auth: Mutex::new(AuthStore::default()),
            auth_config,
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
            inferences: Arc::new(Mutex::new(BTreeMap::new())),
            next_request: std::sync::atomic::AtomicU64::new(now()),
        })
    }
    pub fn router(self: &Arc<Self>) -> Router {
        let cors = if let Some(origin) = &self.auth_config.origin {
            CorsLayer::new()
                .allow_origin(
                    origin
                        .parse::<HeaderValue>()
                        .unwrap_or_else(|_| HeaderValue::from_static("null")),
                )
                .allow_credentials(true)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::CONTENT_TYPE, header::COOKIE])
        } else {
            CorsLayer::new()
        };
        Router::new()
            .route("/healthz", get(healthz))
            .route("/v1/auth/challenge", get(auth_challenge))
            .route("/v1/auth/verify", post(auth_verify))
            .route("/v1/auth/me", get(auth_me))
            .route("/v1/auth/logout", post(auth_logout))
            .route("/v1/status", get(status))
            .route("/v1/history", get(history))
            .route("/v1/model", get(model))
            .route("/v1/events", get(events))
            .route("/v1/verify/infer", post(verify_infer))
            .route("/v1/verify/infer/{id}", get(verify_infer_status))
            .route("/v1/admin/training/{action}", post(admin_training))
            .route("/ipfs/{cid}", get(bundle))
            .layer(cors)
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
        let previous = old.chain.clone();
        old.chain = snapshot_chain(&state);
        old.phase = "watching".into();
        old.last_error = None;
        *self.snapshot.lock().unwrap() = old.clone();
        self.emit("snapshot", serde_json::to_value(&old).unwrap());
        self.emit("chain", serde_json::to_value(&old.chain).unwrap());
        if previous.generation != old.chain.generation {
            self.emit(
                "generation",
                serde_json::json!({"generation":old.chain.generation}),
            );
        }
        if previous.model_hash != old.chain.model_hash {
            self.emit("model", serde_json::json!({"generation":old.chain.generation,"modelHash":old.chain.model_hash}));
        }
        Ok(())
    }
    pub fn start(self: &Arc<Self>) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut delay = Duration::from_secs(1);
            loop {
                match this.refresh().await {
                    Ok(()) => {
                        delay = Duration::from_secs(1);
                        if this.snapshot.lock().unwrap().training_enabled {
                            if let Err(e) = this.schedule_once().await {
                                this.set_error(e)
                            }
                        }
                    }
                    Err(e) => {
                        this.set_error(e);
                        delay = (delay * 2).min(Duration::from_secs(30));
                    }
                }
                tokio::time::sleep(delay).await;
            }
        });
    }
    fn set_error(&self, error: String) {
        self.snapshot.lock().unwrap().last_error = Some(error.clone());
        self.emit("error", serde_json::json!({"message":error}));
    }
    async fn schedule_once(&self) -> Result<(), String> {
        let state = self
            .chain
            .finalized_state()
            .await
            .map_err(|e| e.to_string())?;
        let meta = match state.meta {
            Some(v) => v,
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
            let id = self
                .next_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let work = self
                .chain
                .submit_train_side(side, meta.generation, meta.model_hash, id)
                .await
                .map_err(|e| e.to_string())?;
            self.snapshot.lock().unwrap().phase = format!("training-{}", side.as_str());
            self.chain
                .wait_work(&work)
                .await
                .map_err(|e| e.to_string())?;
            self.emit("training",serde_json::json!({"side":side.as_str(),"generation":meta.generation,"requestId":id}));
        }
        Ok(())
    }
    fn emit(&self, kind: &str, data: serde_json::Value) {
        let _ = self.events.send(WireEvent {
            kind: kind.into(),
            data,
        });
    }
    fn current_snapshot_event(&self) -> Event {
        Event::default().event("snapshot").data(serde_json::to_string(&serde_json::json!({"type":"snapshot","data":self.snapshot.lock().unwrap().clone()})).unwrap())
    }
    async fn challenge(&self, account: [u8; 32]) -> Result<ChallengeResponse, String> {
        let nonce = random_bytes::<32>()?;
        let id = hex::encode(random_bytes::<32>()?);
        let issued = SystemTime::now();
        let expires = issued + CHALLENGE_TTL;
        let genesis = self
            .chain
            .client()
            .genesis_hash()
            .await
            .map_err(|e| e.to_string())?;
        let mut message = format!(
            "MINI Cells Authentication\nVersion: 1\nAccount: 0x{}\nNonce: 0x{}\nGenesis: 0x{}\nService: {}\nIssuedAt: {}\nExpiresAt: {}",
            hex::encode(account),
            hex::encode(nonce),
            hex::encode(genesis),
            self.chain.service_id(),
            unix(issued),
            unix(expires)
        );
        if let Some(origin) = &self.auth_config.origin {
            message.push_str("\nOrigin: ");
            message.push_str(origin);
        }
        let message_bytes = message.as_bytes().to_vec();
        let mut store = self.auth.lock().unwrap();
        cleanup(&mut store);
        if store.challenges.len() >= MAX_CHALLENGES {
            if let Some(oldest) = store.challenges.keys().next().cloned() {
                store.challenges.remove(&oldest);
            }
        }
        store.challenges.insert(
            id.clone(),
            Challenge {
                account,
                message: message_bytes.clone(),
                expires_at: expires,
            },
        );
        Ok(ChallengeResponse {
            challenge_id: id,
            account: format!("0x{}", hex::encode(account)),
            message,
            message_hex: format!("0x{}", hex::encode(message_bytes)),
        })
    }
    fn verify(&self, request: VerifyRequest) -> Result<(String, AuthMe), String> {
        let mut store = self.auth.lock().unwrap();
        verify_request(&mut store, &self.auth_config, request)
    }
    fn session(&self, headers: &HeaderMap) -> Result<Session, StatusCode> {
        let token = headers
            .get(header::COOKIE)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_cookie);
        let Some(token) = token else {
            return Err(StatusCode::UNAUTHORIZED);
        };
        let mut store = self.auth.lock().unwrap();
        cleanup(&mut store);
        store
            .sessions
            .get(token)
            .cloned()
            .ok_or(StatusCode::UNAUTHORIZED)
    }
    async fn submit_verify_infer(&self, text: &str) -> Result<InferAccepted, String> {
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
                result: None,
                error: None,
            },
        );
        let chain = Arc::clone(&self.chain);
        let records = self.inferences.clone();
        tokio::spawn(async move {
            let result = chain.wait_inference(&submitted).await;
            let (status, output, error): (String, Option<InferenceResult>, Option<String>) =
                match result {
                    Ok(record) => ("completed".into(), Some(inference_result(&record)), None),
                    Err(e) => ("failed".into(), None, Some(e.to_string())),
                };
            if let Ok(mut map) = records.lock() {
                if let Some(v) = map.get_mut(&id) {
                    v.status = status;
                    v.result = output;
                    v.error = error
                }
            }
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
fn unix(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
fn random_bytes<const N: usize>() -> Result<[u8; N], String> {
    let mut out = [0u8; N];
    getrandom::fill(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}
fn decode_account(value: &str) -> Result<[u8; 32], String> {
    if let Some(raw) = value.strip_prefix("0x") {
        let bytes = hex::decode(raw).map_err(|e| e.to_string())?;
        return bytes
            .try_into()
            .map_err(|_| "account must be 32 bytes".into());
    }
    let account = AccountId32::from_ss58check(value).map_err(|e| e.to_string())?;
    Ok(account.into())
}
fn cleanup(store: &mut AuthStore) {
    let now = SystemTime::now();
    store.challenges.retain(|_, v| v.expires_at > now);
    store.sessions.retain(|_, v| v.expires_at > now)
}
fn verify_model_bytes(expected: [u8; 32], bytes: &[u8]) -> Result<PackedModel, &'static str> {
    if bytes.len() != MODEL_BYTES {
        return Err("canonical model length mismatch");
    }
    let decoded = PackedModel::decode_from(bytes).map_err(|_| "canonical model decode failed")?;
    if model_hash(&decoded) != expected {
        return Err("canonical model hash mismatch");
    }
    Ok(decoded)
}
fn inference_result(record: &minicells_protocol::InferenceV1) -> InferenceResult {
    let input_len = usize::from(record.input_len).min(record.input.len());
    let output_len = usize::from(record.output_len).min(record.output.len());
    let input = String::from_utf8_lossy(&record.input[..input_len]).into_owned();
    let output = String::from_utf8_lossy(&record.output[..output_len]).into_owned();
    InferenceResult {
        input,
        output,
        generation: record.generation,
        model_hash: format!("0x{}", hex::encode(record.model_hash)),
        similarity: if record.output_len == 0 {
            0.0
        } else {
            f64::from(record.matching_tokens) / f64::from(record.output_len)
        },
    }
}
fn verify_request(
    store: &mut AuthStore,
    config: &AuthConfig,
    request: VerifyRequest,
) -> Result<(String, AuthMe), String> {
    let account = decode_account(&request.account)?;
    let signature =
        hex::decode(request.signature.trim_start_matches("0x")).map_err(|e| e.to_string())?;
    if signature.len() != 64 {
        return Err("signature must be 64 bytes".into());
    }
    cleanup(store);
    let challenge = store
        .challenges
        .get(&request.challenge_id)
        .cloned()
        .ok_or("challenge missing, expired, or used")?;
    if challenge.account != account {
        return Err("challenge account mismatch".into());
    }
    if !sr25519::Pair::verify(
        &sr25519::Signature::from_raw(signature.try_into().unwrap()),
        &challenge.message,
        &sr25519::Public::from_raw(account),
    ) {
        return Err("invalid sr25519 signature".into());
    }
    store.challenges.remove(&request.challenge_id);
    let token = hex::encode(random_bytes::<32>()?);
    let operator = config.operator == Some(account);
    if store.sessions.len() >= MAX_SESSIONS {
        if let Some(oldest) = store.sessions.keys().next().cloned() {
            store.sessions.remove(&oldest);
        }
    }
    store.sessions.insert(
        token.clone(),
        Session {
            account,
            expires_at: SystemTime::now() + SESSION_TTL,
            is_operator: operator,
        },
    );
    Ok((
        token,
        AuthMe {
            authenticated: true,
            account: Some(format!("0x{}", hex::encode(account))),
            is_operator: operator,
        },
    ))
}
fn parse_cookie(value: &str) -> Option<&str> {
    value.split(';').find_map(|item| {
        let mut pair = item.trim().splitn(2, '=');
        if pair.next()? == SESSION_COOKIE {
            Some(pair.next()?)
        } else {
            None
        }
    })
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
fn cookie(token: &str, config: &AuthConfig) -> String {
    format!(
        "{}={}; HttpOnly; SameSite=Strict; Path=/; Max-Age={}{}",
        SESSION_COOKIE,
        token,
        SESSION_TTL.as_secs(),
        if config.secure_cookie { "; Secure" } else { "" }
    )
}
fn expired_cookie(config: &AuthConfig) -> String {
    format!(
        "{}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0{}",
        SESSION_COOKIE,
        if config.secure_cookie { "; Secure" } else { "" }
    )
}
fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error":"authentication required"})),
    )
        .into_response()
}
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({"ok":true,"service":"minicells-keeper"}))
}
async fn auth_challenge(
    Query(query): Query<ChallengeQuery>,
    State(app): State<AppState>,
) -> Response {
    let account = match decode_account(&query.account) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error":e})),
            )
                .into_response()
        }
    };
    match app.keeper.challenge(account).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error":e})),
        )
            .into_response(),
    }
}
async fn auth_verify(State(app): State<AppState>, Json(request): Json<VerifyRequest>) -> Response {
    match app.keeper.verify(request) {
        Ok((token, me)) => (
            [(header::SET_COOKIE, cookie(&token, &app.keeper.auth_config))],
            Json(me),
        )
            .into_response(),
        Err(e) => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error":e})),
        )
            .into_response(),
    }
}
async fn auth_me(headers: HeaderMap, State(app): State<AppState>) -> Response {
    match app.keeper.session(&headers) {
        Ok(session) => Json(AuthMe {
            authenticated: true,
            account: Some(format!("0x{}", hex::encode(session.account))),
            is_operator: session.is_operator,
        })
        .into_response(),
        Err(_) => Json(AuthMe {
            authenticated: false,
            account: None,
            is_operator: false,
        })
        .into_response(),
    }
}
async fn auth_logout(headers: HeaderMap, State(app): State<AppState>) -> Response {
    if let Some(token) = headers
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_cookie)
    {
        app.keeper.auth.lock().unwrap().sessions.remove(token);
    }
    (
        [(header::SET_COOKIE, expired_cookie(&app.keeper.auth_config))],
        Json(serde_json::json!({"ok":true})),
    )
        .into_response()
}
async fn status(State(app): State<AppState>) -> impl IntoResponse {
    Json(app.keeper.snapshot.lock().unwrap().clone())
}
async fn history(State(app): State<AppState>) -> impl IntoResponse {
    match app.keeper.chain.finalized_state().await {
        Ok(s) => Json(
            serde_json::json!({"items":s.history.into_iter().map(|h|HistoryEntry{generation:h.generation,model_hash:format!("0x{}",hex::encode(h.model_hash)),plus_loss:h.plus_loss,minus_loss:h.minus_loss,tokens:h.tokens}).collect::<Vec<_>>() }),
        ),
        Err(e) => Json(serde_json::json!({"error":e.to_string()})),
    }
}
async fn model(headers: HeaderMap, State(app): State<AppState>) -> Response {
    if app.keeper.session(&headers).is_err() {
        return unauthorized();
    }
    match app.keeper.chain.finalized_state().await {
        Ok(s) => {
            let Some(meta) = s.meta else {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error":"model is not initialized"})),
                )
                    .into_response();
            };
            let Some(bytes) = s.model else {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error":"model state is missing"})),
                )
                    .into_response();
            };
            let Ok(decoded) = verify_model_bytes(meta.model_hash, &bytes) else {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error":"canonical model integrity check failed"})),
                )
                    .into_response();
            };
            let computed = model_hash(&decoded);
            let body = ModelResponse {
                format: "minicells.model.v1",
                generation: meta.generation,
                model_format: meta.model_version,
                capability: "ECHO",
                parameter_count: PARAMETER_COUNT,
                model_hash: format!("0x{}", hex::encode(computed)),
                encoding: "base64",
                model_bytes: BASE64.encode(bytes),
            };
            let mut response = Json(body).into_response();
            response.headers_mut().insert(
                header::ETAG,
                HeaderValue::from_str(&format!("\"0x{}\"", hex::encode(computed))).unwrap(),
            );
            response.headers_mut().insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("private, no-cache"),
            );
            response
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error":e.to_string()})),
        )
            .into_response(),
    }
}
async fn events(headers: HeaderMap, State(app): State<AppState>) -> Response {
    if app.keeper.session(&headers).is_err() {
        return unauthorized();
    }
    let initial = tokio_stream::once(Ok::<Event, Infallible>(app.keeper.current_snapshot_event()));
    let keeper = Arc::clone(&app.keeper);
    let updates = BroadcastStream::new(app.keeper.events.subscribe()).map(move |item| {
        Ok::<Event, Infallible>(match item {
            Ok(v) => Event::default().event(v.kind).data(v.data.to_string()),
            Err(_) => keeper.current_snapshot_event(),
        })
    });
    Sse::new(initial.chain(updates))
        .keep_alive(KeepAlive::default())
        .into_response()
}
async fn verify_infer(
    headers: HeaderMap,
    State(app): State<AppState>,
    Json(request): Json<InferRequest>,
) -> Response {
    if app.keeper.session(&headers).is_err() {
        return unauthorized();
    }
    match app.keeper.submit_verify_infer(&request.text).await {
        Ok(v) => (StatusCode::ACCEPTED, Json(v)).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":e})),
        )
            .into_response(),
    }
}
async fn verify_infer_status(
    Path(id): Path<u64>,
    headers: HeaderMap,
    State(app): State<AppState>,
) -> Response {
    if app.keeper.session(&headers).is_err() {
        return unauthorized();
    }
    match app.keeper.inferences.lock().unwrap().get(&id).cloned() {
        Some(v) => Json(v).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"unknown request"})),
        )
            .into_response(),
    }
}
async fn admin_training(
    Path(action): Path<String>,
    headers: HeaderMap,
    State(app): State<AppState>,
) -> Response {
    let Ok(session) = app.keeper.session(&headers) else {
        return unauthorized();
    };
    if !session.is_operator {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"operator session required"})),
        )
            .into_response();
    };
    match action.as_str() {
        "start" => app.keeper.snapshot.lock().unwrap().training_enabled = true,
        "pause" => app.keeper.snapshot.lock().unwrap().training_enabled = false,
        "step" => {
            let result = app.keeper.schedule_once().await;
            if let Err(e) = result {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"error":e})),
                )
                    .into_response();
            }
        }
        _ => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error":"unknown training action"})),
            )
                .into_response()
        }
    };
    Json(app.keeper.snapshot.lock().unwrap().clone()).into_response()
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cookie_and_account_round_trip() {
        let seed = [7u8; 32];
        let pair = sr25519::Pair::from_seed(&seed);
        let account = pair.public().0;
        assert_eq!(
            decode_account(&format!("0x{}", hex::encode(account))).unwrap(),
            account
        );
        let config = AuthConfig::default();
        assert!(cookie("abc", &config).contains("HttpOnly"));
        assert_eq!(parse_cookie("foo=x; minicells_session=abc"), Some("abc"));
    }
    fn challenge_for(pair: &sr25519::Pair, id: &str, message: &[u8]) -> ([u8; 32], VerifyRequest) {
        let account = pair.public().0;
        let signature = pair.sign(message);
        (
            account,
            VerifyRequest {
                challenge_id: id.into(),
                account: format!("0x{}", hex::encode(account)),
                signature: format!("0x{}", hex::encode(signature.0)),
            },
        )
    }
    #[test]
    fn auth_valid_wrong_account_wrong_signature_expiry_and_replay() {
        let good = sr25519::Pair::from_seed(&[7u8; 32]);
        let other = sr25519::Pair::from_seed(&[8u8; 32]);
        let account = good.public().0;
        let config = AuthConfig {
            operator: Some(account),
            ..AuthConfig::default()
        };
        let mut store = AuthStore::default();
        store.challenges.insert(
            "valid".into(),
            Challenge {
                account,
                message: b"hello".to_vec(),
                expires_at: SystemTime::now() + CHALLENGE_TTL,
            },
        );
        let (_, request) = challenge_for(&good, "valid", b"hello");
        let (token, me) = verify_request(&mut store, &config, request).unwrap();
        assert!(me.is_operator && store.sessions.contains_key(&token));
        assert!(verify_request(
            &mut store,
            &config,
            VerifyRequest {
                challenge_id: "valid".into(),
                account: format!("0x{}", hex::encode(account)),
                signature: format!("0x{}", hex::encode(good.sign(b"hello").0))
            }
        )
        .is_err());
        store.challenges.insert(
            "wrong-account".into(),
            Challenge {
                account,
                message: b"hello".to_vec(),
                expires_at: SystemTime::now() + CHALLENGE_TTL,
            },
        );
        let (_, mut wrong_account) = challenge_for(&good, "wrong-account", b"hello");
        wrong_account.account = format!("0x{}", hex::encode(other.public().0));
        assert!(verify_request(&mut store, &config, wrong_account).is_err());
        store.challenges.insert(
            "wrong-signature".into(),
            Challenge {
                account,
                message: b"hello".to_vec(),
                expires_at: SystemTime::now() + CHALLENGE_TTL,
            },
        );
        let (_, mut wrong_signature) = challenge_for(&other, "wrong-signature", b"hello");
        wrong_signature.account = format!("0x{}", hex::encode(account));
        assert!(verify_request(&mut store, &config, wrong_signature).is_err());
        store.challenges.insert(
            "regular".into(),
            Challenge {
                account: other.public().0,
                message: b"hello".to_vec(),
                expires_at: SystemTime::now() + CHALLENGE_TTL,
            },
        );
        let (_, regular_request) = challenge_for(&other, "regular", b"hello");
        let (_, regular_me) = verify_request(&mut store, &config, regular_request).unwrap();
        assert!(!regular_me.is_operator);
        store.challenges.insert(
            "expired".into(),
            Challenge {
                account,
                message: b"hello".to_vec(),
                expires_at: SystemTime::now() - Duration::from_secs(1),
            },
        );
        let (_, expired) = challenge_for(&good, "expired", b"hello");
        assert!(verify_request(&mut store, &config, expired).is_err());
        store.sessions.insert(
            "expired-session".into(),
            Session {
                account,
                expires_at: SystemTime::now() - Duration::from_secs(1),
                is_operator: true,
            },
        );
        cleanup(&mut store);
        assert!(!store.sessions.contains_key("expired-session"));
    }
    #[test]
    fn model_integrity_requires_exact_length_and_hash() {
        let model = PackedModel::default();
        let mut bytes = vec![0u8; MODEL_BYTES];
        model.encode_into(&mut bytes).unwrap();
        let hash = model_hash(&model);
        assert!(verify_model_bytes(hash, &bytes).is_ok());
        assert!(verify_model_bytes([9; 32], &bytes).is_err());
        assert!(verify_model_bytes(hash, &bytes[..bytes.len() - 1]).is_err());
    }
}
