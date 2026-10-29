# Boundary extraction failure review001

## Conclusion

The submitted reports contain zero extracted boundaries, eighteen failures and zero RNG comparisons. Every interval remains NOT_CERTIFIED. This is not evidence that historical sampling diverged, and it is not merely a NumPy-version rejection.

Both reports match the code identities of audit source86399a638f30598cbfa9110dce435d8e70e8e2cd. Original files are preserved byte-for-byte here. The fixed HF reference and R3h identity match the intended B-H/three-seed scope.

## Confirmed implementation defect

The reader invoked `minicells_runtime.store.safe(temp_root, receipt['file'])` before copying any checkpoint into the new temporary directory. That existing helper deliberately rejects a missing file with CHECKPOINT_FILE_MISSING. It is a read/verification helper, not a destination-path constructor. Thus even correctly downloaded checkpoints could not reach Store verification.

The old failure handler retained only exception type and HTTP status, so the exact first failure stage for each of the eighteen original attempts cannot be uniquely recovered from these JSON files alone. The above code defect is independently confirmed and deterministically reproduced; it must be repaired regardless. No claim is made that every underlying request or download succeeded.

## Repair and validation

The destination name is now validated with existing `hf_storage.relative`; verified checkpoint bytes are copied into a fresh temporary directory; existing Store.verified then performs its unchanged safe-path, file-hash, identity and complete-state-fingerprint checks. No historical Store safety rule is removed.

Failures now record the operation phase plus allowlisted constant failure codes, without remote exception text or credentials. If zero boundaries are extracted, the reader exits as failure and the one-launch Notebook does not proceed to another unauthenticated candidate replay. A replay with no comparisons also uses an explicit no-boundaries status.

Two tiny synthetic CPU regression checks exercise the real original Store save/copy/verify path: all18 synthetic boundaries are extracted; altered checkpoint bytes are not accepted. They instantiate no model or optimizer. Three existing sampler checks also pass. These are engineering checks, not authenticated historical extraction evidence.

The Kaggle2.1.3 candidate trace hashes and predicted RNG states match the previously retained local1.24.4 outputs for all three seeds. This finite compatibility result does not establish the original producer NumPy version or actual historical consumption.

## Next step

Run the corrected CPU-only historical-sampling Notebook once using the existing HF_TOKEN. It reuses available HF cache entries, performs no training, and retains small boundary evidence and replay metadata. Return the new two reports. Do not rerun any scientific experiment.

Tokenizer/original asset reconstruction remain PASS. Main candidates, scientific config, original reports and Checkpoints are unchanged. Global content isolation, historical sampling certification and R3m registration remain BLOCKED.
