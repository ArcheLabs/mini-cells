#![no_std]

use core::cell::UnsafeCell;
use minicells_runtime::{
    accumulate, refine, AccumulateWorkspace, Host, HostError, RefineWorkspace,
};

#[repr(C)]
pub struct RefineOutput {
    pub data: *const u8,
    pub size: usize,
}

struct WorkspaceCell(UnsafeCell<RefineWorkspace>);
// SAFETY: MiniJAM invokes one guest export at a time.
unsafe impl Sync for WorkspaceCell {}
static WORKSPACE: WorkspaceCell = WorkspaceCell(UnsafeCell::new(RefineWorkspace::new()));
struct AccumulateWorkspaceCell(UnsafeCell<AccumulateWorkspace>);
// SAFETY: MiniJAM invokes one guest export at a time.
unsafe impl Sync for AccumulateWorkspaceCell {}
static ACC_WORKSPACE: AccumulateWorkspaceCell =
    AccumulateWorkspaceCell(UnsafeCell::new(AccumulateWorkspace::new()));

extern "C" {
    fn minijam_payload(output: *mut u8, capacity: usize, output_size: *mut usize) -> u32;
    fn minijam_extrinsic(
        index: usize,
        output: *mut u8,
        capacity: usize,
        output_size: *mut usize,
    ) -> u32;
    fn minijam_result_count() -> usize;
    fn minijam_result(
        index: usize,
        output: *mut u8,
        capacity: usize,
        output_size: *mut usize,
    ) -> u32;
    fn minijam_storage_read(
        key: *const u8,
        key_size: usize,
        output: *mut u8,
        capacity: usize,
        output_size: *mut usize,
    ) -> u32;
    fn minijam_storage_write(
        key: *const u8,
        key_size: usize,
        value: *const u8,
        value_size: usize,
    ) -> u32;
    fn minijam_storage_delete(key: *const u8, key_size: usize) -> u32;
    fn minijam_yield(value: *const u8, value_size: usize);
}

static mut OUTPUT: [u8; 160] = [0; 160];
struct SdkHost {
    refine: bool,
}
fn status(code: u32) -> Result<(), HostError> {
    match code {
        0 => Ok(()),
        1 => Err(HostError::Missing),
        2 => Err(HostError::BufferTooSmall),
        _ => Err(HostError::Failure),
    }
}
impl Host for SdkHost {
    fn payload(&self, output: &mut [u8]) -> Result<usize, HostError> {
        let mut n = 0; /* SAFETY: output is valid for its declared length and n is a valid out pointer. */
        status(unsafe { minijam_payload(output.as_mut_ptr(), output.len(), &mut n) })?;
        Ok(n)
    }
    fn result_count(&self) -> usize {
        /* SAFETY: host call takes no pointers and returns a bounded count. */
        unsafe { minijam_result_count() }
    }
    fn result(&self, index: usize, output: &mut [u8]) -> Result<usize, HostError> {
        let mut n = 0; /* SAFETY: output is valid for its declared length and n is a valid out pointer. */
        status(unsafe { minijam_result(index, output.as_mut_ptr(), output.len(), &mut n) })?;
        Ok(n)
    }
    fn storage_read(&self, key: &[u8], output: &mut [u8]) -> Result<Option<usize>, HostError> {
        if self.refine {
            let index = if key == minicells_protocol::keys::META {
                0
            } else if key == minicells_protocol::keys::MODEL {
                1
            } else {
                return Ok(None);
            };
            let mut n = 0;
            /* SAFETY: output is a valid writable slice and n is a valid out pointer. */
            let code =
                unsafe { minijam_extrinsic(index, output.as_mut_ptr(), output.len(), &mut n) };
            return match code {
                0 => Ok(Some(n)),
                1 => Ok(None),
                2 => Err(HostError::BufferTooSmall),
                _ => Err(HostError::Failure),
            };
        }
        let mut n = 0; /* SAFETY: key/output slices provide valid pointers and lengths for this call. */
        let code = unsafe {
            minijam_storage_read(
                key.as_ptr(),
                key.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut n,
            )
        };
        match code {
            0 => Ok(Some(n)),
            1 => Ok(None),
            2 => Err(HostError::BufferTooSmall),
            _ => Err(HostError::Failure),
        }
    }
    fn storage_write(&mut self, key: &[u8], value: &[u8]) -> Result<(), HostError> {
        /* SAFETY: both slices provide valid pointers and lengths for the duration of the call. */
        status(unsafe {
            minijam_storage_write(key.as_ptr(), key.len(), value.as_ptr(), value.len())
        })
    }
    fn storage_delete(&mut self, key: &[u8]) -> Result<(), HostError> {
        /* SAFETY: key provides a valid pointer and length for the duration of the call. */
        status(unsafe { minijam_storage_delete(key.as_ptr(), key.len()) })
    }
    fn yield_value(&mut self, value: &[u8]) {
        /* SAFETY: value provides a valid pointer and length for the duration of the call. */
        unsafe { minijam_yield(value.as_ptr(), value.len()) }
    }
}

#[no_mangle]
pub extern "C" fn minijam_refine() -> RefineOutput {
    let mut host = SdkHost { refine: true }; /* SAFETY: MiniJAM invokes one guest export at a time, so this is the only live access to OUTPUT. Using a raw pointer avoids creating a reference directly to a mutable static. */
    let output = unsafe {
        core::slice::from_raw_parts_mut(core::ptr::addr_of_mut!(OUTPUT).cast::<u8>(), 160)
    };
    // SAFETY: MiniJAM invokes one guest export at a time.
    let workspace = unsafe { &mut *WORKSPACE.0.get() };
    match refine(&mut host, workspace, output) {
        Ok(size) => RefineOutput {
            data: output.as_ptr(),
            size,
        },
        Err(error) => {
            let code = match error {
                HostError::Missing => 2u32,
                HostError::BufferTooSmall => 3,
                HostError::Failure => 4,
            };
            output[..4].copy_from_slice(&code.to_le_bytes());
            RefineOutput {
                data: output.as_ptr(),
                size: 4,
            }
        }
    }
}
#[no_mangle]
pub extern "C" fn minijam_accumulate() {
    let mut host = SdkHost { refine: false };
    // SAFETY: MiniJAM invokes one guest export at a time.
    let workspace = unsafe { &mut *ACC_WORKSPACE.0.get() };
    if accumulate(&mut host, workspace).is_ok() {
        host.yield_value(&[]);
    }
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // SAFETY: deliberately terminate the guest so MiniJAM records a PVM panic
    // instead of consuming the entire gas allowance in a spin loop.
    unsafe { core::arch::asm!("unimp", options(noreturn)) }
}

#[no_mangle]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    for index in 0..n {
        /* SAFETY: caller guarantees non-overlapping regions valid for n bytes;
         * volatile accesses prevent LLVM from lowering this builtin to a
         * recursive call to itself. */
        let value = unsafe { core::ptr::read_volatile(src.add(index)) };
        unsafe { core::ptr::write_volatile(dst.add(index), value) };
    }
    dst
}
#[no_mangle]
pub unsafe extern "C" fn memset(dst: *mut u8, value: i32, n: usize) -> *mut u8 {
    for index in 0..n {
        /* SAFETY: caller guarantees dst is valid for n bytes; volatile writes
         * keep this implementation from becoming a recursive memset call. */
        unsafe { core::ptr::write_volatile(dst.add(index), value as u8) };
    }
    dst
}
#[no_mangle]
pub unsafe extern "C" fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    for i in 0..n {
        /* SAFETY: caller guarantees both buffers are valid for n bytes. */
        let av = unsafe { *a.add(i) }; /* SAFETY: same caller invariant as above. */
        let bv = unsafe { *b.add(i) };
        if av != bv {
            return av as i32 - bv as i32;
        }
    }
    0
}
