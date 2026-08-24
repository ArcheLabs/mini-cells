include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../service/generated/genesis_model.rs"
));

use crate::model::PackedModel;

pub fn genesis_model() -> PackedModel {
    PackedModel::from_parameters(GENESIS_MODEL)
}
