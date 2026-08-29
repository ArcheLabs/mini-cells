#![no_std]
#![allow(
    clippy::field_reassign_with_default,
    clippy::needless_range_loop,
    clippy::result_unit_err,
    clippy::should_implement_trait
)]

pub mod batch;
pub mod fixed;
pub mod genesis;
pub mod model;
pub mod optimizer;
pub mod vocab;

pub use batch::{canonical_batch, evaluate_batch, EchoBatch, Evaluation};
pub use model::{forward, model_hash, PackedModel, Scratch, MODEL_BYTES, PARAMETER_COUNT};
