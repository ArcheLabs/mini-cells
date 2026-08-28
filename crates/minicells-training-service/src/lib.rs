#![no_std]

use minicells_training_ref::{
    accumulate_batch_gradients, compute_gradient_leaf, diagnostic_sample_backward,
    diagnostic_sample_forward, finalize_adamw_step, reduce_32_leaves_in_place_ref,
    train_step_with_accumulator, GradientAccumulator, PartialGradientV1, TrainingBatch,
    TrainingState, TrainingWorkspace, PARALLEL_LEAF_COUNT, PARALLEL_SHARD_SIZE, PARAMETER_COUNT,
};

#[cfg(not(feature = "tree"))]
const OUTPUT_CAPACITY: usize = 24 + MODEL_BYTES * 3;
#[cfg(feature = "tree")]
const OUTPUT_CAPACITY: usize = 24 + MODEL_BYTES * 3;
const HEADER: usize = 4 + 8;
const MODEL_BYTES: usize = PARAMETER_COUNT * 4;
const STATE_BYTES: usize = HEADER + MODEL_BYTES * 3 + 256 * 64 + 256;
const DIAGNOSTIC_HEADER: usize = 5;
#[cfg(not(feature = "tree"))]
const PAYLOAD_CAPACITY: usize = STATE_BYTES + DIAGNOSTIC_HEADER + 2 + 8 + MODEL_BYTES;
#[cfg(all(feature = "tree", feature = "tree_leaf_only"))]
const PAYLOAD_CAPACITY: usize = 32_768;
#[cfg(all(feature = "tree", not(feature = "tree_leaf_only")))]
const PAYLOAD_CAPACITY: usize = 900_000;

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
#[cfg(all(feature = "tree", not(feature = "tree_leaf_only")))]
static mut TREE_LEAVES: [PartialGradientV1; PARALLEL_LEAF_COUNT] =
    [const { PartialGradientV1::new() }; PARALLEL_LEAF_COUNT];
#[cfg(feature = "tree")]
static mut TREE_LEAF_OUTPUT: PartialGradientV1 = PartialGradientV1::new();

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

#[cfg(feature = "tree")]
fn take<'a>(input: &'a [u8], cursor: &mut usize, len: usize) -> Option<&'a [u8]> {
    let end = cursor.checked_add(len)?;
    let value = input.get(*cursor..end)?;
    *cursor = end;
    Some(value)
}

#[cfg(feature = "tree")]
fn tree_u16(input: &[u8], cursor: &mut usize) -> Option<u16> {
    Some(u16::from_le_bytes(take(input, cursor, 2)?.try_into().ok()?))
}

#[cfg(feature = "tree")]
fn tree_u64(input: &[u8], cursor: &mut usize) -> Option<u64> {
    Some(u64::from_le_bytes(take(input, cursor, 8)?.try_into().ok()?))
}

#[cfg(feature = "tree")]
fn tree_f32(input: &[u8], cursor: &mut usize) -> Option<f32> {
    Some(f32::from_le_bytes(take(input, cursor, 4)?.try_into().ok()?))
}

#[cfg(feature = "tree")]
fn tree_array32(input: &[u8], cursor: &mut usize) -> Option<[u8; 32]> {
    Some(take(input, cursor, 32)?.try_into().ok()?)
}

#[cfg(feature = "tree")]
fn tree_fail(output: &mut [u8; OUTPUT_CAPACITY]) -> RefineOutput {
    output[..4].copy_from_slice(&u32::MAX.to_le_bytes());
    RefineOutput {
        data: output.as_ptr(),
        size: 4,
    }
}

#[cfg(feature = "tree")]
fn tree_leaf(raw: &[u8], output: &mut [u8; OUTPUT_CAPACITY]) -> RefineOutput {
    // MCG1: magic, version, job, step, model/batch commitments, range,
    // frozen weights, eight fixed-width ids and lengths.
    let mut cursor = 4usize;
    let version = match tree_u16(raw, &mut cursor) {
        Some(v) if v == 1 => v,
        _ => return tree_fail(output),
    };
    let job = match tree_array32(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let step = match tree_u64(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let model = match tree_array32(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let batch_commitment = match tree_array32(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let leaf_index = match tree_u16(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let sample_start = match tree_u16(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let sample_end = match tree_u16(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    if leaf_index as usize >= PARALLEL_LEAF_COUNT
        || sample_start as usize != leaf_index as usize * PARALLEL_SHARD_SIZE
        || sample_end != sample_start + PARALLEL_SHARD_SIZE as u16
    {
        return tree_fail(output);
    }
    let state = unsafe { &mut *core::ptr::addr_of_mut!(STATE) };
    for value in &mut state.weights {
        *value = match tree_f32(raw, &mut cursor) {
            Some(v) => v,
            None => return tree_fail(output),
        };
    }
    let batch = unsafe { &mut *core::ptr::addr_of_mut!(BATCH) };
    batch.size = PARALLEL_SHARD_SIZE as u16;
    for row in 0..PARALLEL_SHARD_SIZE {
        let ids = match take(raw, &mut cursor, 64) {
            Some(v) => v,
            None => return tree_fail(output),
        };
        batch.ids[row].copy_from_slice(ids);
    }
    for row in 0..PARALLEL_SHARD_SIZE {
        batch.lengths[row] = match take(raw, &mut cursor, 1) {
            Some(v) => v[0],
            None => return tree_fail(output),
        };
    }
    let workspace = unsafe { &mut *core::ptr::addr_of_mut!(WORKSPACE) };
    let leaf = unsafe { &mut *core::ptr::addr_of_mut!(TREE_LEAF_OUTPUT) };
    compute_gradient_leaf(state, batch, workspace, leaf);
    let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) };
    let mut out = 0usize;
    output[out..out + 4].copy_from_slice(b"MCGR");
    out += 4;
    output[out..out + 2].copy_from_slice(&version.to_le_bytes());
    out += 2;
    output[out..out + 32].copy_from_slice(&job);
    out += 32;
    output[out..out + 8].copy_from_slice(&step.to_le_bytes());
    out += 8;
    output[out..out + 32].copy_from_slice(&model);
    out += 32;
    output[out..out + 32].copy_from_slice(&batch_commitment);
    out += 32;
    output[out..out + 2].copy_from_slice(&leaf_index.to_le_bytes());
    out += 2;
    output[out..out + 2].copy_from_slice(&sample_start.to_le_bytes());
    out += 2;
    output[out..out + 2].copy_from_slice(&sample_end.to_le_bytes());
    out += 2;
    output[out..out + 4].copy_from_slice(&leaf.loss_sum.to_le_bytes());
    out += 4;
    output[out..out + 4].copy_from_slice(&leaf.token_count.to_le_bytes());
    out += 4;
    output[out..out + 2].copy_from_slice(&leaf.processed_samples.to_le_bytes());
    out += 2;
    for value in &leaf.gradient {
        output[out..out + 4].copy_from_slice(&value.to_le_bytes());
        out += 4;
    }
    RefineOutput {
        data: output.as_ptr(),
        size: out,
    }
}

#[cfg(all(feature = "tree", not(feature = "tree_leaf_only")))]
fn tree_root(raw: &[u8], output: &mut [u8; OUTPUT_CAPACITY]) -> RefineOutput {
    // MCRF1 carries all job metadata, the frozen optimizer state, and the 32
    // immutable MCGR records.  Records may be supplied in any order; they are
    // validated and placed into index-addressed storage before tree reduction.
    let mut cursor = 5usize;
    let version = match tree_u16(raw, &mut cursor) {
        Some(v) if v == 1 => v,
        _ => return tree_fail(output),
    };
    let job = match tree_array32(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let optimizer_step = match tree_u64(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let model = match tree_array32(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let optimizer = match tree_array32(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let batch_commitment = match tree_array32(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    let state_step = match tree_u64(raw, &mut cursor) {
        Some(v) => v,
        None => return tree_fail(output),
    };
    if state_step != optimizer_step {
        return tree_fail(output);
    }
    let state = unsafe { &mut *core::ptr::addr_of_mut!(STATE) };
    for value in &mut state.weights {
        *value = match tree_f32(raw, &mut cursor) {
            Some(v) => v,
            None => return tree_fail(output),
        };
    }
    for value in &mut state.adam_m {
        *value = match tree_f32(raw, &mut cursor) {
            Some(v) => v,
            None => return tree_fail(output),
        };
    }
    for value in &mut state.adam_v {
        *value = match tree_f32(raw, &mut cursor) {
            Some(v) => v,
            None => return tree_fail(output),
        };
    }
    state.step = state_step;
    let leaves = unsafe { &mut *core::ptr::addr_of_mut!(TREE_LEAVES) };
    let mut seen = 0u32;
    const RECORD_BYTES: usize = 4 + 2 + 32 + 8 + 32 + 32 + 6 + 4 + 4 + 2 + MODEL_BYTES;
    for _ in 0..PARALLEL_LEAF_COUNT {
        let record = match take(raw, &mut cursor, RECORD_BYTES) {
            Some(v) => v,
            None => return tree_fail(output),
        };
        let mut rc = 0usize;
        if record.get(0..4) != Some(b"MCGR") {
            return tree_fail(output);
        }
        rc += 4;
        if u16::from_le_bytes(record[rc..rc + 2].try_into().unwrap()) != version {
            return tree_fail(output);
        }
        rc += 2;
        if record[rc..rc + 32] != job {
            return tree_fail(output);
        }
        rc += 32;
        if u64::from_le_bytes(record[rc..rc + 8].try_into().unwrap()) != optimizer_step {
            return tree_fail(output);
        }
        rc += 8;
        if record[rc..rc + 32] != model {
            return tree_fail(output);
        }
        rc += 32;
        if record[rc..rc + 32] != batch_commitment {
            return tree_fail(output);
        }
        rc += 32;
        let leaf_index = u16::from_le_bytes(record[rc..rc + 2].try_into().unwrap()) as usize;
        rc += 2;
        let sample_start = u16::from_le_bytes(record[rc..rc + 2].try_into().unwrap()) as usize;
        rc += 2;
        let sample_end = u16::from_le_bytes(record[rc..rc + 2].try_into().unwrap()) as usize;
        rc += 2;
        if leaf_index >= PARALLEL_LEAF_COUNT
            || sample_start != leaf_index * PARALLEL_SHARD_SIZE
            || sample_end != sample_start + PARALLEL_SHARD_SIZE
        {
            return tree_fail(output);
        }
        if seen & (1u32 << leaf_index) != 0 {
            return tree_fail(output);
        }
        seen |= 1u32 << leaf_index;
        let leaf = &mut leaves[leaf_index];
        leaf.loss_sum = f32::from_le_bytes(record[rc..rc + 4].try_into().unwrap());
        rc += 4;
        leaf.token_count = u32::from_le_bytes(record[rc..rc + 4].try_into().unwrap());
        rc += 4;
        leaf.processed_samples = u16::from_le_bytes(record[rc..rc + 2].try_into().unwrap());
        rc += 2;
        if leaf.processed_samples != PARALLEL_SHARD_SIZE as u16 {
            return tree_fail(output);
        }
        for value in &mut leaf.gradient {
            *value = f32::from_le_bytes(record[rc..rc + 4].try_into().unwrap());
            rc += 4;
        }
    }
    if seen != u32::MAX {
        return tree_fail(output);
    }
    let scratch = unsafe { &mut *core::ptr::addr_of_mut!(ACCUMULATOR) };
    let root = reduce_32_leaves_in_place_ref(leaves);
    scratch.gradient = root.gradient;
    scratch.loss_sum = root.loss_sum;
    scratch.token_count = root.token_count;
    let report = finalize_adamw_step(state, scratch);
    let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) };
    output[..4].copy_from_slice(b"MCPR");
    output[4..8].copy_from_slice(&report.loss.to_le_bytes());
    output[8..12].copy_from_slice(&report.grad_norm.to_le_bytes());
    output[12..16].copy_from_slice(&report.token_count.to_le_bytes());
    output[16..24].copy_from_slice(&state.step.to_le_bytes());
    let mut out = 24usize;
    for value in &state.weights {
        output[out..out + 4].copy_from_slice(&value.to_le_bytes());
        out += 4;
    }
    for value in &state.adam_m {
        output[out..out + 4].copy_from_slice(&value.to_le_bytes());
        out += 4;
    }
    for value in &state.adam_v {
        output[out..out + 4].copy_from_slice(&value.to_le_bytes());
        out += 4;
    }
    let _ = optimizer; // commitment is validated as part of the root envelope.
    RefineOutput {
        data: output.as_ptr(),
        size: out,
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
    #[cfg(feature = "tree")]
    {
        if raw.get(..4) == Some(b"MCG1") {
            return tree_leaf(raw, unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) });
        }
        #[cfg(not(feature = "tree_leaf_only"))]
        if raw.get(..5) == Some(b"MCRF1") {
            return tree_root(raw, unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) });
        }
        return unsafe { tree_fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
    }
    #[cfg(feature = "production")]
    if raw.first().copied() == Some(b'M') && raw.get(1..4) == Some(b"CD1") {
        return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
    }
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
        if input.len() < cursor + 8 + MODEL_BYTES {
            return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
        }
        let token_count = u32::from_le_bytes(input[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let loss_sum = read_f32(input, &mut cursor);
        let accumulator = unsafe { &mut *core::ptr::addr_of_mut!(ACCUMULATOR) };
        accumulator.reset();
        accumulator.token_count = token_count;
        accumulator.loss_sum = loss_sum;
        for value in &mut accumulator.gradient {
            *value = read_f32(input, &mut cursor);
        }
        let report = finalize_adamw_step(state, accumulator);
        let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) };
        output[..4].copy_from_slice(b"MCPR");
        output[4..8].copy_from_slice(&report.loss.to_le_bytes());
        output[8..12].copy_from_slice(&report.grad_norm.to_le_bytes());
        output[12..16].copy_from_slice(&report.token_count.to_le_bytes());
        output[16..24].copy_from_slice(&state.step.to_le_bytes());
        let mut out_cursor = 24;
        for value in &state.weights {
            output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes());
            out_cursor += 4;
        }
        for value in &state.adam_m {
            output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes());
            out_cursor += 4;
        }
        for value in &state.adam_v {
            output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes());
            out_cursor += 4;
        }
        return RefineOutput {
            data: output.as_ptr(),
            size: out_cursor,
        };
    }
    let batch = unsafe { &mut *core::ptr::addr_of_mut!(BATCH) };
    let count = if mode == b"MCP1" || mode == b"MCA1" {
        if input.len() < cursor + 2 {
            return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
        }
        let count = u16::from_le_bytes(input[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;
        if count == 0 || count as usize > 256 {
            return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
        }
        count as usize
    } else {
        256
    };
    if mode == b"MCA1" {
        if input.len() < cursor + 8 + MODEL_BYTES {
            return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
        }
        let accumulator = unsafe { &mut *core::ptr::addr_of_mut!(ACCUMULATOR) };
        accumulator.token_count = u32::from_le_bytes(input[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        accumulator.loss_sum = read_f32(input, &mut cursor);
        for value in &mut accumulator.gradient {
            *value = read_f32(input, &mut cursor);
        }
    }
    let required = cursor + count * 64 + count;
    if input.len() < required {
        return unsafe { fail(&mut *core::ptr::addr_of_mut!(OUTPUT)) };
    }
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
        output[..4].copy_from_slice(b"MCAR");
        output[4..8].copy_from_slice(&accumulator.loss_sum.to_le_bytes());
        output[8..12].copy_from_slice(&accumulator.token_count.to_le_bytes());
        let mut out_cursor = 12;
        for value in &accumulator.gradient {
            output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes());
            out_cursor += 4;
        }
        return RefineOutput {
            data: output.as_ptr(),
            size: out_cursor,
        };
    }
    let report = match diagnostic_stage {
        Some(4) => diagnostic_sample_forward(state, batch, workspace),
        Some(5) => diagnostic_sample_backward(state, batch, workspace),
        Some(6) | Some(7) | Some(8) | None => unsafe {
            train_step_with_accumulator(
                state,
                batch,
                workspace,
                &mut *core::ptr::addr_of_mut!(ACCUMULATOR),
            )
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
        for value in &state.weights {
            output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes());
            out_cursor += 4;
        }
        for value in &state.adam_m {
            output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes());
            out_cursor += 4;
        }
        for value in &state.adam_v {
            output[out_cursor..out_cursor + 4].copy_from_slice(&value.to_le_bytes());
            out_cursor += 4;
        }
        return RefineOutput {
            data: output.as_ptr(),
            size: out_cursor,
        };
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
