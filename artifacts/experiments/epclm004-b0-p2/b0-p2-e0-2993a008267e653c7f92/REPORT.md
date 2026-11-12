# B0-P2 E0

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
    "nll": 5.598904520273209,
    "type_nll": 0.0014054412224287875,
    "value_conditional_nll": 5.597499021678381
  },
  "validation": {
    "correct": 2,
    "query_targets": 512,
    "nll": 5.632760435342789,
    "type_nll": 0.0014037961109080333,
    "value_conditional_nll": 5.631356573423225
  },
  "independent_verification": {
    "download": {
      "status": "PASS",
      "sha256": "8a1a75ab0af096f22f7856a75ee72818285f769339225c6b637fa4da32593d9c",
      "bytes": 306490731,
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
      "restored_state_sha256": "4aaf5bd3e9f6a354e773a8051eb425e4de8e6eff41c850243751dd411f5c4df7",
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
