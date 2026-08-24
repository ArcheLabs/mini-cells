use minicells_runtime::{Host, HostError};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct MemoryHost {
    pub payload: Vec<u8>,
    pub results: Vec<Vec<u8>>,
    pub storage: BTreeMap<Vec<u8>, Vec<u8>>,
    pub yields: Vec<Vec<u8>>,
}
impl MemoryHost {
    pub fn with_payload(payload: Vec<u8>) -> Self {
        Self {
            payload,
            ..Self::default()
        }
    }
}
impl Host for MemoryHost {
    fn payload(&self, o: &mut [u8]) -> Result<usize, HostError> {
        if o.len() < self.payload.len() {
            Err(HostError::BufferTooSmall)
        } else {
            o[..self.payload.len()].copy_from_slice(&self.payload);
            Ok(self.payload.len())
        }
    }
    fn result_count(&self) -> usize {
        self.results.len()
    }
    fn result(&self, i: usize, o: &mut [u8]) -> Result<usize, HostError> {
        let r = self.results.get(i).ok_or(HostError::Missing)?;
        if o.len() < r.len() {
            Err(HostError::BufferTooSmall)
        } else {
            o[..r.len()].copy_from_slice(r);
            Ok(r.len())
        }
    }
    fn storage_read(&self, k: &[u8], o: &mut [u8]) -> Result<Option<usize>, HostError> {
        match self.storage.get(k) {
            Some(v) if v.len() <= o.len() => {
                o[..v.len()].copy_from_slice(v);
                Ok(Some(v.len()))
            }
            Some(_) => Err(HostError::BufferTooSmall),
            None => Ok(None),
        }
    }
    fn storage_write(&mut self, k: &[u8], v: &[u8]) -> Result<(), HostError> {
        self.storage.insert(k.to_vec(), v.to_vec());
        Ok(())
    }
    fn storage_delete(&mut self, k: &[u8]) -> Result<(), HostError> {
        self.storage.remove(k);
        Ok(())
    }
    fn yield_value(&mut self, v: &[u8]) {
        self.yields.push(v.to_vec())
    }
}
