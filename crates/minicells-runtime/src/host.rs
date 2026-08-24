#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostError {
    Missing,
    BufferTooSmall,
    Failure,
}
pub trait Host {
    fn payload(&self, output: &mut [u8]) -> Result<usize, HostError>;
    fn result_count(&self) -> usize;
    fn result(&self, index: usize, output: &mut [u8]) -> Result<usize, HostError>;
    fn storage_read(&self, key: &[u8], output: &mut [u8]) -> Result<Option<usize>, HostError>;
    fn storage_write(&mut self, key: &[u8], value: &[u8]) -> Result<(), HostError>;
    fn storage_delete(&mut self, key: &[u8]) -> Result<(), HostError>;
    fn yield_value(&mut self, value: &[u8]);
}
