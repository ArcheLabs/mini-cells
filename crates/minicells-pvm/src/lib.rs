//! Direct PVM harness boundary.
//!
//! `LocalPvmHost` is the complete chain-free host surface needed by the
//! service: payload/results, storage, external data and yielded values.  The
//! pinned MiniJAM/Jambda executor currently exposes execution through its
//! chain-oriented `RefineCtx`; this crate intentionally refuses to substitute
//! the native runtime until a real `RefineCtx` adapter is available.

use blake2b_simd::Params;
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
    #[error("pinned direct executor adapter is blocked: {0}")]
    ExecutorAdapterBlocked(String),
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

    /// Deliberately does not execute native code under the name “PVM”.
    /// Completing this method requires wiring `VmEngine<InterpBackend>` to a
    /// chain-free `RefineCtx`/`StateView` implementation in the pinned Jambda
    /// tree.  Returning a typed blocker keeps reports honest and prevents a
    /// false parity result.
    pub fn execute_refine(&mut self, _payload: &[u8]) -> Result<Vec<u8>, PvmError> {
        Err(PvmError::ExecutorAdapterBlocked(format!("pinned MiniJAM client {} / Jambda {} only exposes RefineCtx through chain StateView; service artifact hash 0x{}", self.pinned_client_revision, self.pinned_jambda_revision, hex::encode(self.artifact.blake2_hash))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
