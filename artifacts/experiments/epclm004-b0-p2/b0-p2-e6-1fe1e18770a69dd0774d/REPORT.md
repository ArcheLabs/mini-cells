# B0-P2 E6

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
    "nll": 5.600373789668083,
    "type_nll": 0.001891501084379816,
    "value_conditional_nll": 5.59848223716536
  },
  "validation": {
    "correct": 2,
    "query_targets": 512,
    "nll": 5.637417793273926,
    "type_nll": 0.0018915546268202903,
    "value_conditional_nll": 5.635526203867592
  },
  "independent_verification": {
    "download": {
      "status": "PASS",
      "sha256": "68e7b3b1cf7cc97adc655845a8d402599f0c58d71dc3c205600a81e065a4e15b",
      "bytes": 305381920,
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
      "restored_state_sha256": "30e722454c9eceed668f1cc19b2c4848597aff432d0add359b1a00257d3716a5",
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
