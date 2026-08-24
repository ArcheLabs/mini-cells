pub mod memory_host;

use memory_host::MemoryHost;
use minicells_core::{
    batch::{batch_digest, canonical_batch},
    genesis::genesis_model,
    model::model_hash,
};
use minicells_protocol::{keys, MetaV1, Op, WorkBody, WorkPayload};
use minicells_runtime::{accumulate, refine, AccumulateWorkspace, RefineWorkspace};

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq)]
pub struct GenerationVector {
    pub genesis_model_hash: String,
    pub train_batch_digest: String,
    pub plus_result_hex: String,
    pub minus_result_hex: String,
    pub next_model_hash: String,
    pub history_record_hex: String,
}

fn encode_work(work: &WorkPayload) -> Vec<u8> {
    let mut bytes = [0u8; 96];
    let n = work.encode_into(&mut bytes).unwrap();
    bytes[..n].to_vec()
}
fn execute_refine(host: &mut MemoryHost, work: WorkPayload) -> Vec<u8> {
    host.payload = encode_work(&work);
    let mut output = [0u8; 160];
    let n = refine(host, &mut RefineWorkspace::new(), &mut output).unwrap();
    output[..n].to_vec()
}
fn meta(host: &MemoryHost) -> MetaV1 {
    let bytes = host.storage.get(keys::META).unwrap();
    MetaV1::decode(bytes).unwrap()
}

pub fn run_generation_zero() -> (MemoryHost, GenerationVector, Vec<u8>) {
    let mut host = MemoryHost::default();
    let genesis_hash = model_hash(&genesis_model());
    let mut text = [0u8; 32];
    text[..5].copy_from_slice(b"hello");
    let inference = execute_refine(
        &mut host,
        WorkPayload {
            op: Op::Infer,
            flags: 0,
            request_id: 7,
            body: WorkBody::Infer {
                expected_generation: u64::MAX,
                text_len: 5,
                text,
            },
        },
    );
    host.results = vec![inference];
    accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
    let plus = execute_refine(
        &mut host,
        WorkPayload {
            op: Op::TrainPlus,
            flags: 0,
            request_id: 8,
            body: WorkBody::Train {
                generation: 0,
                parent_model_hash: genesis_hash,
            },
        },
    );
    host.results = vec![plus.clone()];
    accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
    assert!(host.storage.contains_key(keys::PENDING_PLUS));
    let minus = execute_refine(
        &mut host,
        WorkPayload {
            op: Op::TrainMinus,
            flags: 0,
            request_id: 9,
            body: WorkBody::Train {
                generation: 0,
                parent_model_hash: genesis_hash,
            },
        },
    );
    host.results = vec![minus.clone()];
    accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
    let after = meta(&host);
    let history_key = keys::history_key(0);
    let history = host.storage.get(history_key.as_slice()).unwrap().clone();
    let vector = GenerationVector {
        genesis_model_hash: format!("0x{}", hex::encode(genesis_hash)),
        train_batch_digest: format!(
            "0x{}",
            hex::encode(batch_digest(&canonical_batch(&genesis_hash, 0, 4).unwrap()))
        ),
        plus_result_hex: hex::encode(&plus),
        minus_result_hex: hex::encode(&minus),
        next_model_hash: format!("0x{}", hex::encode(after.model_hash)),
        history_record_hex: hex::encode(history),
    };
    (host, vector, plus)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_generation_and_replay() {
        let (mut host, _, old_plus) = run_generation_zero();
        let before = meta(&host);
        assert_eq!(before.generation, 1);
        assert!(!host.storage.contains_key(keys::PENDING_PLUS));
        assert!(!host.storage.contains_key(keys::PENDING_MINUS));
        assert!(host.storage.contains_key(keys::history_key(0).as_slice()));
        host.results = vec![old_plus];
        accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
        let after = meta(&host);
        assert_eq!(after.generation, 1);
        assert_eq!(after.stale_results, before.stale_results + 1);
    }
    #[test]
    fn duplicate_side_is_counted() {
        let mut host = MemoryHost::default();
        accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
        let m = meta(&host);
        let plus = execute_refine(
            &mut host,
            WorkPayload {
                op: Op::TrainPlus,
                flags: 0,
                request_id: 1,
                body: WorkBody::Train {
                    generation: 0,
                    parent_model_hash: m.model_hash,
                },
            },
        );
        host.results = vec![plus.clone(), plus];
        accumulate(&mut host, &mut AccumulateWorkspace::new()).unwrap();
        assert_eq!(meta(&host).duplicate_results, 1);
        assert_eq!(meta(&host).generation, 0);
    }
    #[test]
    fn inference_ring_detects_collision_by_request_id() {
        let (host, _, _) = run_generation_zero();
        let key = keys::inference_key(keys::inference_slot(7));
        let record =
            minicells_protocol::InferenceV1::decode(host.storage.get(key.as_slice()).unwrap())
                .unwrap();
        assert_eq!(record.request_id, 7);
        assert_ne!(record.request_id, 23);
    }
    #[test]
    fn fixed_prediction_matches_parity_fixture() {
        let (host, _, _) = run_generation_zero();
        let key = keys::inference_key(keys::inference_slot(7));
        let record =
            minicells_protocol::InferenceV1::decode(host.storage.get(key.as_slice()).unwrap())
                .unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../fixtures/v1/fixed-parity.json")).unwrap();
        assert_eq!(
            &record.output[..record.output_len as usize],
            fixture["prediction"].as_str().unwrap().as_bytes()
        );
    }
    #[test]
    fn golden_generation_vector_matches() {
        let (_, actual, _) = run_generation_zero();
        let expected: GenerationVector =
            serde_json::from_str(include_str!("../../../fixtures/v1/generation-0.json")).unwrap();
        assert_eq!(actual, expected);
    }
}
