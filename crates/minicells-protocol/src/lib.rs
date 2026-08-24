#![no_std]

pub mod codec;
pub mod keys;
pub mod result;
pub mod state;
pub mod work;

pub use result::*;
pub use state::*;
pub use work::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Truncated,
    Invalid,
    Overflow,
}
