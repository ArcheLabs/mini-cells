#![cfg_attr(target_arch = "wasm32", no_std)]
#![allow(static_mut_refs)]

#[cfg(target_arch = "wasm32")]
use core::panic::PanicInfo;
use minicells_core::model::MAX_SEQ_LEN;
use minicells_core::{forward, model_hash, vocab::encode_text, PackedModel, Scratch, MODEL_BYTES};

static mut MODEL: [u8; MODEL_BYTES] = [0; MODEL_BYTES];
static mut EXPECTED_HASH: [u8; 32] = [0; 32];
static mut INPUT: [u8; MAX_SEQ_LEN] = [0; MAX_SEQ_LEN];
static mut OUTPUT: [u8; MAX_SEQ_LEN] = [0; MAX_SEQ_LEN];
static mut OUTPUT_LEN: u32 = 0;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    loop {}
}

pub const OK: i32 = 0;
pub const BAD_MODEL_LEN: i32 = 1;
pub const BAD_MODEL_HASH: i32 = 2;
pub const BAD_INPUT_LEN: i32 = 3;
pub const BAD_VOCAB: i32 = 4;
pub const MODEL_DECODE_ERROR: i32 = 5;
pub const INTERNAL_ERROR: i32 = 6;

#[no_mangle]
pub extern "C" fn minicells_model_ptr() -> *mut u8 {
    unsafe { MODEL.as_mut_ptr() }
}
#[no_mangle]
pub extern "C" fn minicells_hash_ptr() -> *mut u8 {
    unsafe { EXPECTED_HASH.as_mut_ptr() }
}
#[no_mangle]
pub extern "C" fn minicells_input_ptr() -> *mut u8 {
    unsafe { INPUT.as_mut_ptr() }
}
#[no_mangle]
pub extern "C" fn minicells_output_ptr() -> *const u8 {
    unsafe { OUTPUT.as_ptr() }
}
#[no_mangle]
pub extern "C" fn minicells_output_len() -> u32 {
    unsafe { OUTPUT_LEN }
}

#[no_mangle]
pub extern "C" fn minicells_infer(model_len: u32, input_len: u32) -> i32 {
    if model_len as usize != MODEL_BYTES {
        return BAD_MODEL_LEN;
    }
    if input_len as usize > MAX_SEQ_LEN {
        return BAD_INPUT_LEN;
    }
    unsafe {
        let model = match PackedModel::decode_from(&MODEL) {
            Ok(value) => value,
            Err(_) => return MODEL_DECODE_ERROR,
        };
        if model_hash(&model) != EXPECTED_HASH {
            return BAD_MODEL_HASH;
        }
        let input = &INPUT[..input_len as usize];
        let mut ids = [0u8; 64];
        if encode_text(input, &mut ids).is_err() {
            return BAD_VOCAB;
        }
        let mut scratch = Scratch::new();
        let mut core_output = [0u8; 64];
        if forward(
            &model,
            &ids,
            input_len as usize,
            &mut scratch,
            &mut core_output,
            None,
        )
        .is_err()
        {
            return INTERNAL_ERROR;
        }
        OUTPUT[..input_len as usize].copy_from_slice(&core_output[..input_len as usize]);
        OUTPUT_LEN = input_len;
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use minicells_core::model::PackedModel;
    #[test]
    fn abi_checks_hash_length_vocab_and_determinism() {
        let model = PackedModel::default();
        let mut bytes = [0u8; MODEL_BYTES];
        model.encode_into(&mut bytes).unwrap();
        let hash = model_hash(&model);
        unsafe {
            MODEL.copy_from_slice(&bytes);
            EXPECTED_HASH.copy_from_slice(&hash);
            INPUT[..5].copy_from_slice(b"hello");
        }
        assert_eq!(minicells_infer((MODEL_BYTES - 1) as u32, 5), BAD_MODEL_LEN);
        assert_eq!(minicells_infer(MODEL_BYTES as u32, 5), OK);
        let first = unsafe { OUTPUT[..5].to_vec() };
        assert_eq!(minicells_output_len(), 5);
        assert_eq!(minicells_infer(MODEL_BYTES as u32, 5), OK);
        assert_eq!(first, unsafe { OUTPUT[..5].to_vec() });
        unsafe {
            INPUT[0] = b'@';
        }
        assert_eq!(minicells_infer(MODEL_BYTES as u32, 1), BAD_VOCAB);
        unsafe {
            EXPECTED_HASH[0] ^= 1;
        }
        assert_eq!(minicells_infer(MODEL_BYTES as u32, 5), BAD_MODEL_HASH);
    }
}
