#![no_std]

use minicells_training_ref::{
    accumulate_batch_gradients, diagnostic_sample_backward, diagnostic_sample_forward,
    finalize_adamw_step, train_step_with_accumulator, GradientAccumulator, TrainingBatch,
    TrainingState, TrainingWorkspace, PARAMETER_COUNT,
};

const OUTPUT_CAPACITY: usize = 24 + MODEL_BYTES * 3;
const HEADER: usize = 4 + 8;
const MODEL_BYTES: usize = PARAMETER_COUNT * 4;
const STATE_BYTES: usize = HEADER + MODEL_BYTES * 3 + 256 * 64 + 256;
const DIAGNOSTIC_HEADER: usize = 5;
const PAYLOAD_CAPACITY: usize = STATE_BYTES + DIAGNOSTIC_HEADER + 2 + 8 + MODEL_BYTES;

extern "C" {
    fn minijam_payload(output: *mut u8, capacity: usize, output_size: *mut usize) -> u32;
}

static mut INPUT: [u8; PAYLOAD_CAPACITY] = [0; PAYLOAD_CAPACITY];
static mut OUTPUT: [u8; OUTPUT_CAPACITY] = [0; OUTPUT_CAPACITY];
// Keep optimizer state in static data rather than copying it onto the small
// PVM call stack.
static mut STATE: TrainingState = TrainingState::from_weights([0.0; PARAMETER_COUNT]);
static mut BATCH: TrainingBatch = TrainingBatch::empty();
static mut WORKSPACE: TrainingWorkspace = TrainingWorkspace::new();
static mut ACCUMULATOR: GradientAccumulator = GradientAccumulator::new();

#[repr(C)]
pub struct RefineOutput {
    pub data: *const u8,
    pub size: usize,
}

fn read_f32(input: &[u8], cursor: &mut usize) -> f32 {
    let bytes = [
        input[*cursor],
        input[*cursor + 1],
        input[*cursor + 2],
        input[*cursor + 3],
    ];
    *cursor += 4;
    f32::from_le_bytes(bytes)
}

fn fail(output: &mut [u8; OUTPUT_CAPACITY]) -> RefineOutput {
    output[..4].copy_from_slice(&u32::MAX.to_le_bytes());
    RefineOutput {
        data: output.as_ptr(),
        size: 4,
    }
}

fn diagnostic(output: &mut [u8; OUTPUT_CAPACITY], stage: u8) -> RefineOutput {
    output[..4].copy_from_slice(b"MDG1");
    output[4] = stage;
    RefineOutput {
        data: output.as_ptr(),
        size: 5,
    }
}

#[no_mangle]
pub extern "C" fn minijam_refine() -> RefineOutput {
    let mut size = 0usize;
    let raw = unsafe {
        let ptr = core::ptr::addr_of_mut!(INPUT).cast::<u8>();
        if minijam_payload(ptr, PAYLOAD_CAPACITY, &mut size) != 0 {
            return fail(&mut *core::ptr::addr_of_mut!(OUTPUT));
        }
        core::slice::from_raw_parts(ptr, size)
    };
    let (diagnostic_stage, input) =
        if raw.first().copied() == Some(b'M') && raw.get(1..4) == Some(b"CD1") {
            let stage = raw.get(4).copied().unwrap_or(u8::MAX);
            if stage > 8 || raw.len() < DIAGNOSTIC_HEADER {
                return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
            }
            if stage == 0 {
                return unsafe { diagnostic(&mut *core::ptr::addr_of_mut!(OUTPUT), stage) };
            }
            (Some(stage), &raw[DIAGNOSTIC_HEADER..])
        } else {
            (None, raw)
        };
    if input.len() < HEADER + MODEL_BYTES * 3 {
        return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
    }
    let mode = &input[..4];
    if mode != b"MCT1" && mode != b"MCP1" && mode != b"MCA1" && mode != b"MCF1" {
        return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
    }
    let mut cursor = 4usize;
    let step = u64::from_le_bytes(input[cursor..cursor + 8].try_into().unwrap());
    cursor += 8;
    let state = unsafe { &mut *core::ptr::addr_of_mut!(STATE) };
    for value in &mut state.weights {
        *value = read_f32(input, &mut cursor);
    }
    for value in &mut state.adam_m {
        *value = read_f32(input, &mut cursor);
    }
    for value in &mut state.adam_v {
        *value = read_f32(input, &mut cursor);
    }
    state.step = step;
    if diagnostic_stage == Some(1) {
        return unsafe { diagnostic(&mut *core::ptr::addr_of_mut!(OUTPUT), 1) };
    }
    if mode == b"MCF1" {
        if input.len() < cursor + 8 + MODEL_BYTES { return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) }; }
        let token_count = u32::from_le_bytes(input[cursor..cursor + 4].try_into().unwrap()); cursor += 4;
        let loss_sum = read_f32(input, &mut cursor);
        let accumulator = unsafe { &mut *core::ptr::addr_of_mut!(ACCUMULATOR) };
        accumulator.reset(); accumulator.token_count = token_count; accumulator.loss_sum = loss_sum;
        for value in &mut accumulator.gradient { *value = read_f32(input, &mut cursor); }
        let report = finalize_adamw_step(state, accumulator);
        let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) };
        output[..4].copy_from_slice(b"MCPR"); output[4..8].copy_from_slice(&report.loss.to_le_bytes());
        output[8..12].copy_from_slice(&report.grad_norm.to_le_bytes()); output[12..16].copy_from_slice(&report.token_count.to_le_bytes());
        output[16..24].copy_from_slice(&state.step.to_le_bytes());
        let mut out_cursor = 24;
        for value in &state.weights { output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes()); out_cursor += 4; }
        for value in &state.adam_m { output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes()); out_cursor += 4; }
        for value in &state.adam_v { output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes()); out_cursor += 4; }
        return RefineOutput { data: output.as_ptr(), size: out_cursor };
    }
    let batch = unsafe { &mut *core::ptr::addr_of_mut!(BATCH) };
    let count = if mode == b"MCP1" || mode == b"MCA1" {
        if input.len() < cursor + 2 { return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) }; }
        let count = u16::from_le_bytes(input[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;
        if count == 0 || count as usize > 256 { return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) }; }
        count as usize
    } else { 256 };
    if mode == b"MCA1" {
        if input.len() < cursor + 8 + MODEL_BYTES { return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) }; }
        let accumulator = unsafe { &mut *core::ptr::addr_of_mut!(ACCUMULATOR) };
        accumulator.token_count = u32::from_le_bytes(input[cursor..cursor + 4].try_into().unwrap()); cursor += 4;
        accumulator.loss_sum = read_f32(input, &mut cursor);
        for value in &mut accumulator.gradient { *value = read_f32(input, &mut cursor); }
    }
    let required = cursor + count * 64 + count;
    if input.len() < required { return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) }; }
    batch.size = count as u16;
    for row in 0..count {
        batch.ids[row].copy_from_slice(&input[cursor..cursor + 64]);
        cursor += 64;
    }
    batch.lengths[..count].copy_from_slice(&input[cursor..cursor + count]);
    if diagnostic_stage == Some(2) {
        return unsafe { diagnostic(&mut *core::ptr::addr_of_mut!(OUTPUT), 2) };
    }
    let workspace = unsafe { &mut *core::ptr::addr_of_mut!(WORKSPACE) };
    if diagnostic_stage == Some(3) {
        return unsafe { diagnostic(&mut *core::ptr::addr_of_mut!(OUTPUT), 3) };
    }
    // MCA1 is an accumulation-only ABI.  It deliberately bypasses the
    // optimizer-step dispatcher below: weights, Adam state and step remain
    // frozen while the supplied shard is added exactly once.
    if mode == b"MCA1" {
        if let Some(stage @ 4..=8) = diagnostic_stage {
            return unsafe { diagnostic(&mut *core::ptr::addr_of_mut!(OUTPUT), stage) };
        }
        let accumulator = unsafe { &mut *core::ptr::addr_of_mut!(ACCUMULATOR) };
        accumulate_batch_gradients(state, batch, workspace, accumulator);
        let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) };
        output[..4].copy_from_slice(b"MCAR"); output[4..8].copy_from_slice(&accumulator.loss_sum.to_le_bytes());
        output[8..12].copy_from_slice(&accumulator.token_count.to_le_bytes());
        let mut out_cursor = 12;
        for value in &accumulator.gradient { output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes()); out_cursor += 4; }
        return RefineOutput { data: output.as_ptr(), size: out_cursor };
    }
    let report = match diagnostic_stage {
        Some(4) => diagnostic_sample_forward(state, batch, workspace),
        Some(5) => diagnostic_sample_backward(state, batch, workspace),
        Some(6) | Some(7) | Some(8) | None => unsafe {
            train_step_with_accumulator(state, batch, workspace, &mut *core::ptr::addr_of_mut!(ACCUMULATOR))
        },
        _ => unreachable!(),
    };
    if let Some(stage) = diagnostic_stage {
        return unsafe { diagnostic(&mut *core::ptr::addr_of_mut!(OUTPUT), stage) };
    }
    let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) };
    if mode == b"MCP1" {
        output[..4].copy_from_slice(b"MCPR");
        output[4..8].copy_from_slice(&report.loss.to_le_bytes());
        output[8..12].copy_from_slice(&report.grad_norm.to_le_bytes());
        output[12..16].copy_from_slice(&report.token_count.to_le_bytes());
        output[16..24].copy_from_slice(&state.step.to_le_bytes());
        let mut out_cursor = 24;
        for value in &state.weights { output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes()); out_cursor += 4; }
        for value in &state.adam_m { output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes()); out_cursor += 4; }
        for value in &state.adam_v { output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes()); out_cursor += 4; }
        return RefineOutput { data: output.as_ptr(), size: out_cursor };
    }
    output[..4].copy_from_slice(&report.loss.to_le_bytes());
    output[4..8].copy_from_slice(&report.grad_norm.to_le_bytes());
    output[8..12].copy_from_slice(&report.token_count.to_le_bytes());
    output[12..20].copy_from_slice(&state.step.to_le_bytes());
    RefineOutput {
        data: output.as_ptr(),
        size: 20,
    }
}

// The MiniJAM SDK emits metadata for both exports.  The fidelity benchmark
// only executes Refine, but the linker still requires the Accumulate symbol
// referenced by that metadata.  Keeping this no-op export makes the guest a
// valid SDK service without introducing any training state transition.
#[no_mangle]
pub extern "C" fn minijam_accumulate() {}

#[cfg(not(test))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { core::arch::asm!("unimp", options(noreturn)) }
}

#[no_mangle]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    for index in 0..n {
        core::ptr::write_volatile(dst.add(index), core::ptr::read_volatile(src.add(index)));
    }
    dst
}
#[no_mangle]
pub unsafe extern "C" fn memset(dst: *mut u8, value: i32, n: usize) -> *mut u8 {
    for index in 0..n {
        core::ptr::write_volatile(dst.add(index), value as u8);
    }
    dst
}
#[no_mangle]
pub unsafe extern "C" fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    for index in 0..n {
        let av = *a.add(index);
        let bv = *b.add(index);
        if av != bv {
            return av as i32 - bv as i32;
        }
    }
    0
}
