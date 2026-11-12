# B0-P2 E7-Innovation

{
  "status": "COMPLETE",
  "ledger": {
    "input_tokens": 327680,
    "processed_tensor_tokens": 327680,
    "padding_tokens": 0,
    "supervised_target_tokens": 5120,
    "query_target_tokens": 5120,
    "committed_updates": 640
  },
  "training": {
    "correct": 4,
    "query_targets": 256,
    "nll": 5.5946714878082275,
    "type_nll": 0.001969982583253531,
    "value_conditional_nll": 5.5927015243529965
  },
  "validation": {
    "correct": 2,
    "query_targets": 512,
    "nll": 5.634478345513344,
    "type_nll": 0.0019770726936730373,
    "value_conditional_nll": 5.632501279745988
  },
  "independent_verification": {
    "download": {
      "status": "PASS",
      "sha256": "b203865874e161ad2afb447e009172f0fa063f8c8b39c597e8cc119af7fe5543",
      "bytes": 306898582,
      "files_verified": 55,
      "scope": "All archived files, including checkpoint/probabilities/metrics/report; no trusted status substitution"
    },
    "payload": {
      "status": "PASS",
      "ledger": {
        "input_tokens": 327680,
        "processed_tensor_tokens": 327680,
        "padding_tokens": 0,
        "supervised_target_tokens": 5120,
        "query_target_tokens": 5120,
        "committed_updates": 640
      },
      "history_steps_verified": 640,
      "test_accessed": false
    },
    "restore": {
      "status": "PASS",
      "restored_state_sha256": "44dd870392962608e22ec7e3e3ea91137c574861a3c604b216daff4e666c920d",
      "new_optimizer_updates": 0,
      "checks": {
        "training": {
          "status": "PASS",
          "correct": 4,
          "nll_abs_error": 0.0,
          "probability_max_abs_error": 0.0,
          "tolerance": 2e-05
        },
        "validation": {
          "status": "PASS",
          "correct": 2,
          "nll_abs_error": 0.0,
          "probability_max_abs_error": 0.0,
          "tolerance": 2e-05
        }
      },
      "test_accessed": false
    }
  }
}

Single-seed development evidence only. Split and exposure changed jointly; no isolated attribution or H5 confirmation. Test unaccessed; no12M, main three-seed experiment or M6. Store position hits do not establish causal utility.
