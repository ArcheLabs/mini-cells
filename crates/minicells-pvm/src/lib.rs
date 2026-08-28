//! Direct PVM harness boundary.
//!
//! `LocalPvmHost` is the complete chain-free host surface needed by the
//! service: payload/results, storage, external data and yielded values.  The
//! pinned Jambda `VmEngine` is used directly; no chain `StateView` is involved.

use blake2b_simd::Params;
use jp_vm_engine::{run_standalone, StandaloneProgram};
use jp_vm_interp::{memory::InnerInterpMemory, register::InterpRegister, InterpBackend};
use jp_vm_primitives::{
    error::VmError,
    host::HostCallTrait,
    state::{VmMemory, VmRegister, VmState},
    ExitKind,
};
use minicells_runtime::{Host, HostError};
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Debug, thiserror::Error)]
pub enum PvmError {
    #[error("service artifact not found: {0}")]
    ArtifactMissing(PathBuf),
    #[error("service artifact is empty")]
    EmptyArtifact,
    #[error("PVM decode error: {0}")]
    Decode(String),
    #[error("PVM execution error: {0}")]
    Execution(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

#[derive(Clone, Debug)]
pub struct ServicePvmArtifact {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub blake2_hash: [u8; 32],
}

impl ServicePvmArtifact {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PvmError> {
        let path = path.as_ref().to_owned();
        let bytes = fs::read(&path).map_err(|_| PvmError::ArtifactMissing(path.clone()))?;
        if bytes.is_empty() {
            return Err(PvmError::EmptyArtifact);
        }
        let mut state = Params::new().hash_length(32).to_state();
        state.update(b"mini-cells:pvm-artifact:v1");
        state.update(&bytes);
        let mut hash = [0; 32];
        hash.copy_from_slice(state.finalize().as_bytes());
        Ok(Self {
            path,
            bytes,
            blake2_hash: hash,
        })
    }
}

#[derive(Default)]
pub struct LocalPvmHost {
    pub payload: Vec<u8>,
    pub results: Vec<Vec<u8>>,
    pub storage: BTreeMap<Vec<u8>, Vec<u8>>,
    pub external_data: Vec<Vec<u8>>,
    pub yields: Vec<Vec<u8>>,
}

impl Host for LocalPvmHost {
    fn payload(&self, output: &mut [u8]) -> Result<usize, HostError> {
        copy_out(&self.payload, output)
    }
    fn result_count(&self) -> usize {
        self.results.len()
    }
    fn result(&self, index: usize, output: &mut [u8]) -> Result<usize, HostError> {
        self.results
            .get(index)
            .ok_or(HostError::Missing)
            .and_then(|v| copy_out(v, output))
    }
    fn storage_read(&self, key: &[u8], output: &mut [u8]) -> Result<Option<usize>, HostError> {
        match self.storage.get(key) {
            Some(v) => copy_out(v, output).map(Some),
            None => Ok(None),
        }
    }
    fn storage_write(&mut self, key: &[u8], value: &[u8]) -> Result<(), HostError> {
        self.storage.insert(key.to_vec(), value.to_vec());
        Ok(())
    }
    fn storage_delete(&mut self, key: &[u8]) -> Result<(), HostError> {
        self.storage.remove(key);
        Ok(())
    }
    fn yield_value(&mut self, value: &[u8]) {
        self.yields.push(value.to_vec());
    }
}

const HOST_NONE: u64 = u64::MAX;
const HOST_GAS: u32 = 0;
const HOST_FETCH: u32 = 1;
const HOST_READ: u32 = 3;
const HOST_WRITE: u32 = 4;
const HOST_YIELD: u32 = 25;
const HOST_LOG: u32 = 100;

impl HostCallTrait<InterpRegister, InnerInterpMemory> for LocalPvmHost {
    fn ecalli(
        &mut self,
        id: u32,
        state: &mut VmState<InterpRegister, InnerInterpMemory>,
        gas: &mut i64,
    ) -> Result<ExitKind, VmError> {
        let arg = |index: u8| state.registers.get(index);
        let set_result = |state: &mut VmState<InterpRegister, InnerInterpMemory>, value: u64| {
            state.registers.set_a0(value);
        };
        match id {
            HOST_GAS => set_result(state, (*gas).max(0) as u64),
            HOST_FETCH => {
                let ptr = arg(7) as u32;
                let offset = arg(8) as usize;
                let capacity = arg(9) as usize;
                let mode = arg(10);
                let index = arg(11) as usize;
                let item = match mode {
                    13 => Some(&self.payload),
                    4 => self.external_data.get(index),
                    15 => self.results.get(index),
                    _ => None,
                };
                let Some(item) = item else {
                    set_result(state, HOST_NONE);
                    return Ok(ExitKind::Continue);
                };
                let remaining = item.get(offset..).unwrap_or_default();
                let copy_len = remaining.len().min(capacity);
                state
                    .memory
                    .write_bytes(ptr, &remaining[..copy_len])
                    .map_err(|_| VmError::Panic)?;
                // FETCH returns the full item length even when an offset/length
                // window was requested.  The SDK uses this to validate the
                // second half of minijam_result.
                set_result(state, item.len() as u64);
            }
            HOST_READ => {
                let key_ptr = arg(8) as u32;
                let key_len = arg(9) as usize;
                let out_ptr = arg(10) as u32;
                let capacity = arg(12) as usize;
                let mut key = vec![0; key_len];
                state
                    .memory
                    .read_bytes_into(key_ptr, &mut key)
                    .map_err(|_| VmError::Panic)?;
                let Some(value) = self.storage.get(&key) else {
                    set_result(state, HOST_NONE);
                    return Ok(ExitKind::Continue);
                };
                let copy_len = value.len().min(capacity);
                state
                    .memory
                    .write_bytes(out_ptr, &value[..copy_len])
                    .map_err(|_| VmError::Panic)?;
                set_result(state, value.len() as u64);
            }
            HOST_WRITE => {
                let key_ptr = arg(7) as u32;
                let key_len = arg(8) as usize;
                let value_ptr = arg(9) as u32;
                let value_len = arg(10) as usize;
                let mut key = vec![0; key_len];
                let mut value = vec![0; value_len];
                state
                    .memory
                    .read_bytes_into(key_ptr, &mut key)
                    .map_err(|_| VmError::Panic)?;
                state
                    .memory
                    .read_bytes_into(value_ptr, &mut value)
                    .map_err(|_| VmError::Panic)?;
                if value.is_empty() {
                    self.storage.remove(&key);
                } else {
                    self.storage.insert(key, value);
                }
                set_result(state, 0);
            }
            HOST_YIELD => {
                let ptr = arg(7) as u32;
                let mut value = vec![0; 32];
                state
                    .memory
                    .read_bytes_into(ptr, &mut value)
                    .map_err(|_| VmError::Panic)?;
                self.yields.push(value);
                set_result(state, 0);
            }
            HOST_LOG => {
                let ptr = arg(7) as u32;
                let len = arg(8) as usize;
                let mut message = vec![0; len];
                state
                    .memory
                    .read_bytes_into(ptr, &mut message)
                    .map_err(|_| VmError::Panic)?;
                let _ = message;
                set_result(state, 0);
            }
            _ => set_result(state, 0),
        }
        Ok(ExitKind::Continue)
    }
}

fn copy_out(value: &[u8], output: &mut [u8]) -> Result<usize, HostError> {
    if output.len() < value.len() {
        return Err(HostError::BufferTooSmall);
    }
    output[..value.len()].copy_from_slice(value);
    Ok(value.len())
}

pub struct DirectPvmHarness {
    pub artifact: ServicePvmArtifact,
    pub host: LocalPvmHost,
    pub gas_limit: u64,
    pub pinned_client_revision: String,
    pub pinned_jambda_revision: String,
}

#[derive(Clone, Debug)]
pub struct PvmExecution {
    pub output: Vec<u8>,
    pub gas_used: u64,
    pub gas_remaining: u64,
}

impl DirectPvmHarness {
    pub fn load(
        path: impl AsRef<Path>,
        gas_limit: u64,
        client_revision: impl Into<String>,
        jambda_revision: impl Into<String>,
    ) -> Result<Self, PvmError> {
        Ok(Self {
            artifact: ServicePvmArtifact::load(path)?,
            host: LocalPvmHost::default(),
            gas_limit,
            pinned_client_revision: client_revision.into(),
            pinned_jambda_revision: jambda_revision.into(),
        })
    }

    /// Execute the service refine export through Jambda's chain-free runner.
    pub fn execute_refine(&mut self, payload: &[u8]) -> Result<Vec<u8>, PvmError> {
        self.host.payload = payload.to_vec();
        Ok(self.execute_refine_measured(payload)?.output)
    }

    pub fn execute_refine_measured(&mut self, payload: &[u8]) -> Result<PvmExecution, PvmError> {
        self.host.payload = payload.to_vec();
        self.execute_entry(payload, 0)
    }

    /// Execute the second converter dispatch entry, which is the service's
    /// accumulate export.  Accumulate communicates through host storage/yield
    /// side effects and normally returns no output bytes.
    pub fn execute_accumulate(&mut self) -> Result<Vec<u8>, PvmError> {
        self.host.payload.clear();
        Ok(self.execute_entry(&[], 5)?.output)
    }

    /// Wrap a RefineResult in the MiniJAM accumulation operand envelope.
    /// This is shared by the PVM trainer and parity tests so the ABI is not
    /// reimplemented by individual callers.
    pub fn pack_accumulation_result(refine_output: &[u8]) -> Result<Vec<u8>, PvmError> {
        if refine_output.is_empty() || refine_output.len() > u8::MAX as usize {
            return Err(PvmError::Decode("invalid refine result length".into()));
        }
        // MiniJAM's accumulation operand is the four-hash/gas envelope that
        // the SDK unwraps before handing the payload to RefineResult::decode.
        let mut item = vec![0; 1 + 4 * 32];
        item.push(0); // fnencode(gas)
        item.push(0); // refine result marker
        jp_vm_primitives::encode_fnencode(refine_output.len() as u64, &mut item);
        item.extend_from_slice(refine_output);
        Ok(item)
    }

    fn execute_entry(
        &mut self,
        payload: &[u8],
        entry_point: u32,
    ) -> Result<PvmExecution, PvmError> {
        let program = StandaloneProgram::from_bytes(&self.artifact.bytes)
            .map_err(|error| PvmError::Decode(error.to_string()))?;
        let result = run_standalone(
            &program,
            InterpBackend::new(),
            &mut self.host,
            std::sync::Arc::from(payload.to_vec()),
            entry_point,
            self.gas_limit,
        )
        .map_err(|error| PvmError::Execution(error.to_string()))?;
        let gas_used = result.gas_used;
        let gas_remaining = result.gas_remaining as u64;
        match result.result {
            jp_vm_primitives::VmResult::Ok(Some(output)) => Ok(PvmExecution {
                output: output.into_vec(),
                gas_used,
                gas_remaining,
            }),
            jp_vm_primitives::VmResult::Ok(None) => Ok(PvmExecution {
                output: Vec::new(),
                gas_used,
                gas_remaining,
            }),
            jp_vm_primitives::VmResult::Oog => Err(PvmError::Execution("out of gas".into())),
            jp_vm_primitives::VmResult::Panic => Err(PvmError::Execution("PVM panic".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minicells_protocol::{
        keys, HistoryV1, MetaV1, Op, RefineResult, ResultBody, WorkBody, WorkPayload,
    };
    use minicells_runtime::{
        accumulate, ensure_initialized, genesis::GENESIS_MODEL_HASH, refine, AccumulateWorkspace,
        RefineWorkspace,
    };

    fn artifact_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../service/artifacts/service.blob")
    }

    fn harness() -> DirectPvmHarness {
        DirectPvmHarness::load(
            artifact_path(),
            100_000_000,
            "0b352d42726c548e932f81138c8dff7bc9b5a786",
            "788bc054223f81282e4d88a83f05f2fe9e94c121",
        )
        .expect("tracked service artifact")
    }

    fn training_result(op: Op, request_id: u64, side: i8, base_loss: i64, loss: i64) -> Vec<u8> {
        let result = RefineResult {
            op,
            status: 0,
            request_id,
            generation: 0,
            model_hash: GENESIS_MODEL_HASH,
            body: ResultBody::Training {
                side,
                base_loss,
                base_correct_tokens: 8,
                base_eval_digest: [request_id as u8 + 1; 32],
                loss,
                correct_tokens: 7,
                total_tokens: 16,
                eval_digest: [request_id as u8; 32],
            },
        };
        let mut encoded = [0; 160];
        let size = result
            .encode_into(&mut encoded)
            .expect("training result codec");
        encoded[..size].to_vec()
    }

    fn packed_result(payload: &[u8]) -> Vec<u8> {
        DirectPvmHarness::pack_accumulation_result(payload).unwrap()
    }

    fn assert_storage_equal(native: &LocalPvmHost, pvm: &LocalPvmHost) {
        for key in native.storage.keys().chain(pvm.storage.keys()) {
            if native.storage.get(key) != pvm.storage.get(key) {
                eprintln!(
                    "storage mismatch key={:?} native_len={:?} pvm_len={:?}",
                    String::from_utf8_lossy(key),
                    native.storage.get(key).map(Vec::len),
                    pvm.storage.get(key).map(Vec::len)
                );
            }
        }
        assert_eq!(native.storage, pvm.storage);
    }
    #[test]
    fn local_host_preserves_jam_surface() {
        let mut host = LocalPvmHost::default();
        host.storage_write(b"k", b"v").unwrap();
        let mut out = [0; 1];
        assert_eq!(host.storage_read(b"k", &mut out), Ok(Some(1)));
        assert_eq!(out, [b'v']);
        host.external_data.push(b"sample".to_vec());
        host.yield_value(b"ok");
        assert_eq!(host.yields, vec![b"ok".to_vec()]);
    }

    #[test]
    fn service_pvm_matches_native_runtime_for_status_probe() {
        let mut harness = harness();
        let work = WorkPayload {
            op: Op::StatusProbe,
            flags: 0,
            request_id: 1,
            body: WorkBody::StatusProbe,
        };
        let mut payload = [0u8; 96];
        let payload_len = work.encode_into(&mut payload).expect("encode work");
        let pvm_output = harness
            .execute_refine(&payload[..payload_len])
            .expect("PVM status probe");

        let mut native_host = LocalPvmHost::default();
        native_host.payload = payload[..payload_len].to_vec();
        let mut workspace = RefineWorkspace::new();
        let mut native_output = [0u8; 160];
        let native_len = refine(&mut native_host, &mut workspace, &mut native_output)
            .expect("native status probe");
        assert_eq!(pvm_output, native_output[..native_len]);
    }

    #[test]
    fn service_pvm_matches_native_runtime_for_plus_and_minus() {
        for op in [Op::TrainPlus, Op::TrainMinus] {
            let mut harness = harness();
            let work = WorkPayload {
                op,
                flags: 0,
                request_id: 2,
                body: WorkBody::Train {
                    generation: 0,
                    parent_model_hash: GENESIS_MODEL_HASH,
                },
            };
            let mut payload = [0u8; 96];
            let payload_len = work.encode_into(&mut payload).expect("encode work");
            let pvm_output = harness
                .execute_refine(&payload[..payload_len])
                .expect("PVM training probe");

            let mut native_host = LocalPvmHost::default();
            native_host.payload = payload[..payload_len].to_vec();
            let mut workspace = RefineWorkspace::new();
            let mut native_output = [0u8; 160];
            let native_len = refine(&mut native_host, &mut workspace, &mut native_output)
                .expect("native training probe");
            assert_eq!(pvm_output, native_output[..native_len], "op={op:?}");
        }
    }

    #[test]
    fn service_pvm_accumulate_entry_is_chain_free() {
        let mut harness = harness();
        harness.execute_accumulate().expect("PVM accumulate");
        assert!(!harness.host.storage.is_empty());
    }

    #[test]
    fn service_pvm_accumulate_matches_native_storage_and_yield() {
        let plus = training_result(Op::TrainPlus, 11, 1, 90, 100);
        let minus = training_result(Op::TrainMinus, 12, -1, 90, 110);

        let mut native = LocalPvmHost::default();
        ensure_initialized(&mut native).expect("native genesis");
        native.results = vec![plus.clone(), minus.clone()];
        let mut native_workspace = AccumulateWorkspace::new();
        accumulate(&mut native, &mut native_workspace).expect("native accumulate");
        // This is the service ABI acknowledgement emitted by minijam_accumulate.
        native.yields.push(vec![0; 32]);

        let mut pvm = harness();
        ensure_initialized(&mut pvm.host).expect("PVM genesis");
        pvm.host.results = vec![packed_result(&plus), packed_result(&minus)];
        pvm.execute_accumulate().expect("PVM accumulate");

        assert_storage_equal(&native, &pvm.host);
        assert_eq!(native.yields, pvm.host.yields);
        let meta = native.storage.get(keys::META).expect("META");
        let model = native.storage.get(keys::MODEL).expect("MODEL");
        assert!(!model.is_empty());
        assert!(native.storage.contains_key(keys::history_key(0).as_slice()));
        assert!(!native.storage.contains_key(keys::PENDING_PLUS));
        assert!(!native.storage.contains_key(keys::PENDING_MINUS));
        assert_eq!(meta, pvm.host.storage.get(keys::META).expect("PVM META"));
    }

    #[test]
    fn service_pvm_keep_and_accept_paths_match_native() {
        for (base, plus_loss, minus_loss, updated) in [(100, 110, 120, 0u8), (100, 80, 120, 1u8)] {
            let plus = training_result(Op::TrainPlus, 31, 1, base, plus_loss);
            let minus = training_result(Op::TrainMinus, 32, -1, base, minus_loss);
            let mut native = LocalPvmHost::default();
            ensure_initialized(&mut native).expect("native genesis");
            let parent_model = native.storage.get(keys::MODEL).unwrap().clone();
            native.results = vec![plus.clone(), minus.clone()];
            accumulate(&mut native, &mut AccumulateWorkspace::new()).expect("native accumulate");
            let mut pvm = harness();
            ensure_initialized(&mut pvm.host).expect("PVM genesis");
            pvm.host.results = vec![
                DirectPvmHarness::pack_accumulation_result(&minus).unwrap(),
                DirectPvmHarness::pack_accumulation_result(&plus).unwrap(),
            ];
            pvm.execute_accumulate().expect("PVM accumulate");
            assert_storage_equal(&native, &pvm.host);
            let meta = MetaV1::decode(native.storage.get(keys::META).unwrap()).unwrap();
            let history =
                HistoryV1::decode(native.storage.get(keys::history_key(0).as_slice()).unwrap())
                    .unwrap();
            assert_eq!(history.updated, updated);
            assert_eq!(
                meta.current_eval_loss,
                if updated == 0 { base } else { plus_loss }
            );
            if updated == 0 {
                assert_eq!(native.storage.get(keys::MODEL), Some(&parent_model));
            } else {
                assert_ne!(native.storage.get(keys::MODEL), Some(&parent_model));
            }
        }
    }

    #[test]
    fn service_pvm_matches_native_full_zero_to_one_transition() {
        let mut native = LocalPvmHost::default();
        ensure_initialized(&mut native).expect("native genesis");
        let mut native_refine_workspace = RefineWorkspace::new();
        let mut pvm = harness();
        ensure_initialized(&mut pvm.host).expect("PVM genesis");

        for (request_id, op) in [(21, Op::TrainPlus), (22, Op::TrainMinus)] {
            let work = WorkPayload {
                op,
                flags: 0,
                request_id,
                body: WorkBody::Train {
                    generation: 0,
                    parent_model_hash: GENESIS_MODEL_HASH,
                },
            };
            let mut payload = [0u8; 96];
            let payload_len = work.encode_into(&mut payload).expect("encode work");
            native.payload = payload[..payload_len].to_vec();
            let mut native_output = [0u8; 160];
            let native_len = refine(
                &mut native,
                &mut native_refine_workspace,
                &mut native_output,
            )
            .expect("native refine");
            let pvm_output = pvm
                .execute_refine(&payload[..payload_len])
                .expect("PVM refine");
            assert_eq!(&pvm_output, &native_output[..native_len], "op={op:?}");
            native.results.push(native_output[..native_len].to_vec());
            pvm.host.results.push(packed_result(&pvm_output));
        }

        let mut native_accumulate_workspace = AccumulateWorkspace::new();
        accumulate(&mut native, &mut native_accumulate_workspace).expect("native accumulate");
        native.yields.push(vec![0; 32]);
        pvm.execute_accumulate().expect("PVM accumulate");

        assert_storage_equal(&native, &pvm.host);
        assert_eq!(native.yields, pvm.host.yields);
        let native_meta = native.storage.get(keys::META).expect("native META");
        let pvm_meta = pvm.host.storage.get(keys::META).expect("PVM META");
        assert_eq!(native_meta, pvm_meta);
        assert_eq!(
            native.storage.get(keys::MODEL),
            pvm.host.storage.get(keys::MODEL)
        );
        assert!(native.storage.contains_key(keys::history_key(0).as_slice()));
        assert!(!native.storage.contains_key(keys::PENDING_PLUS));
        assert!(!native.storage.contains_key(keys::PENDING_MINUS));
    }
}
