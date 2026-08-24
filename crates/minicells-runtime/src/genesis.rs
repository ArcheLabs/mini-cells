use crate::{Host, HostError};
use minicells_core::{genesis::genesis_model, model::MODEL_BYTES};
use minicells_protocol::{keys, MetaV1, META_ENCODED_LEN};

static GENESIS_MODEL_BYTES: &[u8; MODEL_BYTES] =
    include_bytes!("../../../service/generated/genesis_model.bin");

pub fn ensure_initialized<H: Host>(host: &mut H) -> Result<MetaV1, HostError> {
    let mut meta_bytes = [0u8; META_ENCODED_LEN];
    if let Some(size) = host.storage_read(keys::META, &mut meta_bytes)? {
        return MetaV1::decode(&meta_bytes[..size]).map_err(|_| HostError::Failure);
    }
    let meta = MetaV1::new(GENESIS_MODEL_HASH);
    let mut encoded = [0u8; META_ENCODED_LEN];
    let size = meta
        .encode_into(&mut encoded)
        .map_err(|_| HostError::Failure)?;
    host.storage_write(keys::MODEL, GENESIS_MODEL_BYTES)?;
    host.storage_write(keys::META, &encoded[..size])?;
    host.storage_delete(keys::PENDING_PLUS)?;
    host.storage_delete(keys::PENDING_MINUS)?;
    Ok(meta)
}

pub fn read_meta_or_genesis<H: Host>(host: &H) -> Result<MetaV1, HostError> {
    let mut bytes = [0u8; META_ENCODED_LEN];
    match host.storage_read(keys::META, &mut bytes)? {
        Some(n) => MetaV1::decode(&bytes[..n]).map_err(|_| HostError::Failure),
        None => Ok(MetaV1::new(GENESIS_MODEL_HASH)),
    }
}

pub const GENESIS_MODEL_HASH: [u8; 32] = [
    0x0a, 0xf6, 0x95, 0x37, 0x31, 0x04, 0x1b, 0x36, 0x12, 0xfd, 0xcb, 0x3c, 0xc4, 0x81, 0xa0, 0x9d,
    0x50, 0x1d, 0x9b, 0xfe, 0x4d, 0x84, 0x01, 0x27, 0x9d, 0x02, 0x83, 0x12, 0x32, 0xff, 0x2f, 0xd2,
];
pub fn read_model_or_genesis<H: Host>(host: &H) -> Result<minicells_core::PackedModel, HostError> {
    let mut model = minicells_core::PackedModel::default();
    let mut bytes = [0; MODEL_BYTES];
    read_model_into_or_genesis(host, &mut model, &mut bytes)?;
    Ok(model)
}

#[inline(never)]
pub fn read_model_into_or_genesis<H: Host>(
    host: &H,
    model: &mut minicells_core::PackedModel,
    bytes: &mut [u8; MODEL_BYTES],
) -> Result<(), HostError> {
    bytes.fill(0);
    match host.storage_read(keys::MODEL, bytes)? {
        Some(n) => model
            .decode_into(&bytes[..n])
            .map_err(|_| HostError::Failure),
        None => {
            *model = genesis_model();
            Ok(())
        }
    }
}
