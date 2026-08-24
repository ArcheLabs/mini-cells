use async_trait::async_trait;
use bounded_collections::BoundedVec;
use cid::Cid;
use jam_codec::Decode as JamDecode;
use jp_core_primitives::types::ServiceInfo;
use minicells_protocol::InferenceV1;
use minicells_protocol::{keys, HistoryV1, MetaV1, PendingV1, WorkBody, WorkPayload};
use minijam_bulletin_api::{
    AccountId, Authorization, BulletinError, BulletinStore, ContentStatus, RenewalRef,
};
use minijam_chain_client::{FinalizedContext, MiniJamChainClient};
use minijam_protocol_external::{
    blake2_256, CidConfig, ContentRef, Hash, HashingAlgorithm, StorageLocation, StorageReceipt,
};
use minijam_work_package_builder::{build_work_package, BuildWorkInput};
use multihash::Multihash;
use sp_core::{sr25519, Pair};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use thiserror::Error;

const RAW_CODEC: u64 = 0x55;
const BLAKE2B_256_MULTIHASH: u64 = 0xb220;

#[derive(Debug, Error)]
pub enum ChainError {
    #[error("chain client: {0}")]
    Chain(#[from] minijam_chain_client::ChainClientError),
    #[error("bulletin: {0}")]
    Bulletin(#[from] BulletinError),
    #[error("invalid service info: {0}")]
    ServiceInfo(String),
    #[error("invalid service state: {0}")]
    State(String),
    #[error("work timed out")]
    WorkTimeout,
    #[error("work failed")]
    WorkFailed,
    #[error("invalid input: {0}")]
    Input(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalizedMiniCellsState {
    pub block_hash: Hash,
    pub block_number: u32,
    pub state_root: Hash,
    pub lookup_anchor_slot: u32,
    pub meta: Option<MetaV1>,
    pub model: Option<Vec<u8>>,
    pub pending_plus: Option<PendingV1>,
    pub pending_minus: Option<PendingV1>,
    pub history: Vec<HistoryV1>,
    pub inferences: Vec<InferenceV1>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainSide {
    Plus,
    Minus,
}

impl TrainSide {
    fn op(self) -> minicells_protocol::Op {
        match self {
            Self::Plus => minicells_protocol::Op::TrainPlus,
            Self::Minus => minicells_protocol::Op::TrainMinus,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Plus => "PLUS",
            Self::Minus => "MINUS",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubmittedWork {
    pub request_id: u64,
    pub side: Option<TrainSide>,
    pub package_hash: Hash,
    pub content_ref: ContentRef,
    pub extrinsic_hash: Hash,
    pub submitted_nonce: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalizedWork {
    pub submitted: SubmittedWork,
    pub work_id: u64,
    pub execution_receipt: Hash,
}

#[async_trait]
pub trait ModelWitnessProvider: Send + Sync {
    async fn load_model_witness(
        &self,
        finalized: &FinalizedMiniCellsState,
    ) -> Result<Vec<u8>, ChainError>;
}

pub struct InlineServiceStateProvider;

#[async_trait]
impl ModelWitnessProvider for InlineServiceStateProvider {
    async fn load_model_witness(
        &self,
        finalized: &FinalizedMiniCellsState,
    ) -> Result<Vec<u8>, ChainError> {
        finalized
            .model
            .clone()
            .ok_or_else(|| ChainError::State("mc:v1:model is missing".into()))
    }
}

pub struct MiniCellsChain<B> {
    client: Arc<MiniJamChainClient>,
    bulletin: Arc<B>,
    service_id: u32,
    bulletin_account: AccountId,
    poll_interval: Duration,
}

impl<B: BulletinStore + 'static> MiniCellsChain<B> {
    pub async fn connect(
        rpc_url: impl Into<String>,
        signer_seed: [u8; 32],
        service_id: u32,
        bulletin: Arc<B>,
    ) -> Result<Self, ChainError> {
        let signer = sr25519::Pair::from_seed(&signer_seed);
        let account = signer.public().0;
        let client = MiniJamChainClient::connect(rpc_url, signer, Duration::from_secs(15)).await?;
        Ok(Self {
            client: Arc::new(client),
            bulletin,
            service_id,
            bulletin_account: account,
            poll_interval: Duration::from_millis(500),
        })
    }

    pub fn service_id(&self) -> u32 {
        self.service_id
    }

    pub fn client(&self) -> &Arc<MiniJamChainClient> {
        &self.client
    }

    pub fn bulletin_store(&self) -> &B {
        &self.bulletin
    }

    pub async fn deploy_service(
        &self,
        artifact: &[u8],
        min_item_gas: u64,
        min_memo_gas: u64,
    ) -> Result<minijam_chain_client::Submission, ChainError> {
        let code_hash = blake2_256(artifact);
        self.client.submit_preimage(artifact.to_vec()).await?;
        Ok(self
            .client
            .submit_create_service(
                self.bulletin_account,
                code_hash,
                artifact.len() as u32,
                min_item_gas,
                min_memo_gas,
            )
            .await?)
    }

    pub async fn finalized_context(&self) -> Result<FinalizedContext, ChainError> {
        Ok(self.client.finalized_context().await?)
    }

    pub async fn finalized_state(&self) -> Result<FinalizedMiniCellsState, ChainError> {
        let context = self.client.finalized_context().await?;
        let read = |key: Vec<u8>| async move {
            self.client
                .service_storage_at(context.block_hash, self.service_id, &key)
                .await
                .map_err(ChainError::from)
        };
        let meta = read(keys::META.to_vec())
            .await?
            .map(|bytes| {
                MetaV1::decode(&bytes).map_err(|_| ChainError::State("invalid MetaV1".into()))
            })
            .transpose()?;
        let model = read(keys::MODEL.to_vec()).await?;
        let pending_plus = read(keys::PENDING_PLUS.to_vec())
            .await?
            .map(|bytes| {
                PendingV1::decode(&bytes)
                    .map_err(|_| ChainError::State("invalid PLUS pending record".into()))
            })
            .transpose()?;
        let pending_minus = read(keys::PENDING_MINUS.to_vec())
            .await?
            .map(|bytes| {
                PendingV1::decode(&bytes)
                    .map_err(|_| ChainError::State("invalid MINUS pending record".into()))
            })
            .transpose()?;
        let history = if let Some(meta) = &meta {
            let mut records = Vec::new();
            for offset in 0..64u8 {
                let slot = (meta.history_head + 64 - 1 - offset) % 64;
                if let Some(bytes) = read(keys::history_key(slot).to_vec()).await? {
                    records.push(
                        HistoryV1::decode(&bytes)
                            .map_err(|_| ChainError::State("invalid history record".into()))?,
                    );
                }
            }
            records
        } else {
            Vec::new()
        };
        let mut inferences = Vec::new();
        for slot in 0..16u8 {
            if let Some(bytes) = read(keys::inference_key(slot).to_vec()).await? {
                inferences.push(
                    InferenceV1::decode(&bytes)
                        .map_err(|_| ChainError::State("invalid inference record".into()))?,
                );
            }
        }
        Ok(FinalizedMiniCellsState {
            block_hash: context.block_hash,
            block_number: context.block_number,
            state_root: context.state_root,
            lookup_anchor_slot: context.slot,
            meta,
            model,
            pending_plus,
            pending_minus,
            history,
            inferences,
        })
    }

    async fn service_code_hash(&self, block: Hash) -> Result<Hash, ChainError> {
        let bytes = self
            .client
            .service_info_at(block, self.service_id)
            .await?
            .ok_or_else(|| ChainError::State("service info is missing".into()))?;
        let info = ServiceInfo::decode(&mut bytes.as_slice())
            .map_err(|error| ChainError::ServiceInfo(error.to_string()))?;
        Ok(info.code_hash.0)
    }

    async fn submit_payload(
        &self,
        request_id: u64,
        side: Option<TrainSide>,
        payload: Vec<u8>,
    ) -> Result<SubmittedWork, ChainError> {
        let finalized = self.finalized_state().await?;
        let context = FinalizedContext {
            block_hash: finalized.block_hash,
            block_number: finalized.block_number,
            state_root: finalized.state_root,
            slot: finalized.lookup_anchor_slot,
        };
        let code_hash = self.service_code_hash(context.block_hash).await?;
        let mut extrinsics = Vec::new();
        if let (Some(meta), Some(model)) = (&finalized.meta, &finalized.model) {
            let mut encoded = vec![0; minicells_protocol::META_ENCODED_LEN];
            let size = meta
                .encode_into(&mut encoded)
                .map_err(|_| ChainError::State("cannot encode MetaV1".into()))?;
            encoded.truncate(size);
            extrinsics.push(encoded);
            extrinsics.push(model.clone());
        } else if finalized.meta.is_some() || finalized.model.is_some() {
            return Err(ChainError::State("META/MODEL state is incomplete".into()));
        }
        let built = build_work_package(BuildWorkInput {
            service_id: self.service_id,
            service_code_hash: code_hash,
            payload,
            extrinsics,
            anchor_hash: context.block_hash,
            state_root: context.state_root,
            lookup_anchor_slot: context.slot,
        })
        .map_err(|error| ChainError::Input(error.to_string()))?;
        let receipt = self
            .bulletin
            .store(&self.bulletin_account, &built.bundle_bytes)
            .await?;
        if receipt.content != built.content_ref {
            return Err(ChainError::State("Bulletin ContentRef mismatch".into()));
        }
        let stored = self.bulletin.fetch(&built.content_ref).await?;
        if stored != built.bundle_bytes {
            return Err(ChainError::State(
                "Bulletin bundle verification failed".into(),
            ));
        }
        let submission = self
            .client
            .submit_work(
                built.canonical_work_package,
                built.content_ref,
                built.package_hash,
            )
            .await?;
        Ok(SubmittedWork {
            request_id,
            side,
            package_hash: built.package_hash,
            content_ref: receipt.content,
            extrinsic_hash: submission.extrinsic_hash,
            submitted_nonce: submission.submitted_nonce,
        })
    }

    pub async fn submit_infer(
        &self,
        request_id: u64,
        text: &[u8],
    ) -> Result<SubmittedWork, ChainError> {
        if text.len() > 32 {
            return Err(ChainError::Input("Echo input exceeds 32 bytes".into()));
        }
        let mut body = [0; 32];
        body[..text.len()].copy_from_slice(text);
        let payload = encode_work(WorkPayload {
            op: minicells_protocol::Op::Infer,
            flags: 0,
            request_id,
            body: WorkBody::Infer {
                expected_generation: u64::MAX,
                text_len: text.len() as u8,
                text: body,
            },
        })?;
        self.submit_payload(request_id, None, payload).await
    }

    pub async fn submit_status_probe(&self, request_id: u64) -> Result<SubmittedWork, ChainError> {
        let payload = encode_work(WorkPayload {
            op: minicells_protocol::Op::StatusProbe,
            flags: 0,
            request_id,
            body: WorkBody::StatusProbe,
        })?;
        self.submit_payload(request_id, None, payload).await
    }

    pub async fn submit_train_side(
        &self,
        side: TrainSide,
        generation: u64,
        model_hash: Hash,
        request_id: u64,
    ) -> Result<SubmittedWork, ChainError> {
        let payload = encode_work(WorkPayload {
            op: side.op(),
            flags: 0,
            request_id,
            body: WorkBody::Train {
                generation,
                parent_model_hash: model_hash,
            },
        })?;
        self.submit_payload(request_id, Some(side), payload).await
    }

    pub async fn wait_work(&self, submitted: &SubmittedWork) -> Result<FinalizedWork, ChainError> {
        for _ in 0..3600 {
            if let Some(work_id) = self
                .client
                .work_id_by_package_hash(submitted.package_hash)
                .await?
            {
                if let Some(receipt) = self.client.execution_receipt(work_id).await? {
                    return Ok(FinalizedWork {
                        submitted: submitted.clone(),
                        work_id,
                        execution_receipt: receipt,
                    });
                }
            }
            tokio::time::sleep(self.poll_interval).await;
        }
        Err(ChainError::WorkTimeout)
    }

    pub async fn wait_inference(
        &self,
        submitted: &SubmittedWork,
    ) -> Result<InferenceV1, ChainError> {
        self.wait_work(submitted).await?;
        for _ in 0..3600 {
            let state = self.finalized_state().await?;
            if let Some(record) = state
                .inferences
                .into_iter()
                .find(|record| record.request_id == submitted.request_id)
            {
                return Ok(record);
            }
            tokio::time::sleep(self.poll_interval).await;
        }
        Err(ChainError::WorkTimeout)
    }

    pub async fn wait_generation_after(
        &self,
        generation: u64,
    ) -> Result<FinalizedMiniCellsState, ChainError> {
        for _ in 0..3600 {
            let state = self.finalized_state().await?;
            if state
                .meta
                .as_ref()
                .is_some_and(|meta| meta.generation > generation)
            {
                return Ok(state);
            }
            tokio::time::sleep(self.poll_interval).await;
        }
        Err(ChainError::WorkTimeout)
    }
}

fn encode_work(work: WorkPayload) -> Result<Vec<u8>, ChainError> {
    let mut bytes = [0; 96];
    let size = work
        .encode_into(&mut bytes)
        .map_err(|_| ChainError::Input("work payload is too large".into()))?;
    Ok(bytes[..size].to_vec())
}

#[derive(Default)]
struct FileState {
    next_index: u32,
    authorizations: BTreeMap<AccountId, Authorization>,
}

pub struct FilesystemBulletinStore {
    root: PathBuf,
    state: Mutex<FileState>,
}

impl FilesystemBulletinStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, BulletinError> {
        let root = root.into();
        fs::create_dir_all(root.join("blobs"))
            .map_err(|error| BulletinError::Io(error.to_string()))?;
        Ok(Self {
            root,
            state: Mutex::new(FileState::default()),
        })
    }

    pub fn blob_path(&self, hash: &Hash) -> PathBuf {
        self.root.join("blobs").join(hex::encode(hash))
    }

    pub fn read_by_hash(&self, hash: &Hash) -> Result<Vec<u8>, BulletinError> {
        fs::read(self.blob_path(hash)).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                BulletinError::Missing
            } else {
                BulletinError::Io(error.to_string())
            }
        })
    }
}

#[async_trait]
impl BulletinStore for FilesystemBulletinStore {
    async fn authorization(
        &self,
        account: &AccountId,
    ) -> Result<Option<Authorization>, BulletinError> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .authorizations
            .get(account)
            .copied())
    }

    async fn store_with_cid_config(
        &self,
        account: &AccountId,
        config: CidConfig,
        data: &[u8],
    ) -> Result<StorageReceipt, BulletinError> {
        if config.codec != RAW_CODEC || config.hashing != HashingAlgorithm::Blake2b256 {
            return Err(BulletinError::UnsupportedCid);
        }
        let authorization = self
            .state
            .lock()
            .unwrap()
            .authorizations
            .get(account)
            .copied();
        if let Some(mut authorization) = authorization {
            if authorization.transactions_left == 0 || authorization.bytes_left < data.len() as u64
            {
                return Err(BulletinError::QuotaExhausted);
            }
            authorization.transactions_left -= 1;
            authorization.bytes_left -= data.len() as u64;
            self.state
                .lock()
                .unwrap()
                .authorizations
                .insert(*account, authorization);
        }
        let content_hash = blake2_256(data);
        let multihash = Multihash::<64>::wrap(BLAKE2B_256_MULTIHASH, &content_hash)
            .map_err(|error| BulletinError::InvalidCid(error.to_string()))?;
        let cid = Cid::new_v1(RAW_CODEC, multihash).to_bytes();
        let cid_v1 = BoundedVec::try_from(cid)
            .map_err(|_| BulletinError::InvalidCid("CID exceeds protocol bound".into()))?;
        let path = self.blob_path(&content_hash);
        if path.exists() {
            if fs::read(&path).map_err(|error| BulletinError::Io(error.to_string()))? != data {
                return Err(BulletinError::Corrupt);
            }
        } else {
            fs::write(&path, data).map_err(|error| BulletinError::Io(error.to_string()))?;
        }
        let mut state = self.state.lock().unwrap();
        let location = StorageLocation {
            block_number: 0,
            transaction_index: state.next_index,
        };
        state.next_index = state.next_index.saturating_add(1);
        Ok(StorageReceipt {
            content: ContentRef {
                cid_v1,
                content_hash,
                size: data.len() as u64,
            },
            location,
            retention_until: u32::MAX,
        })
    }

    async fn fetch(&self, content: &ContentRef) -> Result<Vec<u8>, BulletinError> {
        let bytes = self.read_by_hash(&content.content_hash)?;
        if bytes.len() as u64 != content.size || blake2_256(&bytes) != content.content_hash {
            return Err(BulletinError::Corrupt);
        }
        Ok(bytes)
    }

    async fn renew(
        &self,
        account: &AccountId,
        reference: RenewalRef,
    ) -> Result<StorageReceipt, BulletinError> {
        let hash = match reference {
            RenewalRef::ByContentHash(hash) => hash,
            RenewalRef::ByLocation(_) => return Err(BulletinError::Missing),
        };
        let bytes = self.read_by_hash(&hash)?;
        self.store(account, &bytes).await
    }

    async fn status(&self, content: &ContentRef) -> Result<ContentStatus, BulletinError> {
        match self.fetch(content).await {
            Ok(_) => Ok(ContentStatus::Available {
                retention_until: u32::MAX,
            }),
            Err(BulletinError::Missing) => Ok(ContentStatus::Missing),
            Err(BulletinError::Expired) => Ok(ContentStatus::Expired),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn filesystem_bulletin_round_trip_verifies_content_ref() {
        let directory = tempdir().unwrap();
        let store = FilesystemBulletinStore::new(directory.path()).unwrap();
        let account = [7; 32];
        let receipt = store.store(&account, b"bundle").await.unwrap();
        assert_eq!(store.fetch(&receipt.content).await.unwrap(), b"bundle");
        assert_eq!(
            store.status(&receipt.content).await.unwrap(),
            ContentStatus::Available {
                retention_until: u32::MAX
            }
        );
    }
}
