# Kaggle CPU checkpoint audit

Source commit `89db0a4703e012bc3cbc37e5e9bdd9fba9cc65df`.
Model revision `5c8be94196cb1827e820077680ac2ee46ac949bc`.
The Kaggle receipt reported PASS in a CPU session with Python 3.13.16.
It reports elapsed time, all five gates, shard size and matching expected
and observed SHA-256, tokenizer hashes, and the complete tensor counts.

- `kaggle.log` preserves all eleven records supplied by the operator.
- `model_manifest.json` transcribes the 26-field Kaggle JSON viewer output.
  The collapsed architecture array and BF16 summary were restored from the
  pinned config and audited tensor totals. The original file bytes and
  Kaggle run URL were not supplied.

The run did not disclose CPU hardware, Kaggle image version or script version
ID in the supplied evidence. Accelerator was CPU. The credential warning
shows anonymous Hugging Face access and contains no credential value.
