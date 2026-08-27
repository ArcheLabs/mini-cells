# Experiment 024 Results

This directory contains curated, reproducible outputs of the Kaggle run.
Unlisted caches and other regenerable intermediate files are intentionally excluded.

## Decision

- Status: `FIRST_BIRTH_WITHOUT_SECOND_TRAIT_GENESIS`
- Diagnosis: the first Story/Arithmetic birth is supported, but the planned Transform third capability is largely absorbed by the already-emerged computational trait. The run therefore does not establish repeated 1->2->3 genesis, and it should not be interpreted as a simple failure of the probationary birth rule.

## Diagnostic interpretation

The first committed bifurcation forms a stable Story-versus-computational division. After this 1->2 birth, the branch that improves Arithmetic also improves Transform strongly before a third trait is committed. For example, in replicate 0 at Stage E, the computational branch improves held-out Arithmetic by about 1.091 NLL relative to the frozen parent baseline and improves Transform by about 1.367 NLL. The same qualitative transfer appears in the other replicates.

This means the benchmark taxonomy `ARITHMETIC` versus `TRANSFORM` is not equivalent to the organism's natural trait boundary. The existing two-trait basis can economically absorb much of the Transform demand. Consistent with that interpretation, duplicate Arithmetic and weak Transform proposals are rejected in 3/3 replicates. A strong Transform proposal commits in only one replicate; even there, the resulting three-branch system fails the preregistered three-domain functional-identity/routing gate.

The correct scientific reading is therefore:

`task label != natural expert boundary`

and, more specifically:

`a new trait should be expected only when the current phenotype basis cannot economically absorb the new developmental demand.`

See `DIAGNOSIS.md` for the full evidence and implications. Raw CSVs and `decision.json` are unchanged.

## Key metrics

- Story-only reject: 3/3.
- Story/Arithmetic birth: 2/3 under the full preregistered scientific gate.
- Duplicate-Arithmetic reject: 3/3.
- Weak-Transform reject: 3/3.
- Strong Transform birth: 0/3 under the full preregistered scientific gate.
- Final K=3: 1/3, but that replicate fails the three-domain identity/routing requirement.

## Provenance

- Source commit: `b77441bc9cacc4626cbeb913d85ff7b87c103aaf`
- Source branch: `main`
- Kaggle script version ID: `not recorded`
- Source results directory: `results/sequential-probationary-genesis-v1`

Machine-readable provenance and SHA-256 hashes are in `metadata.json`.
The authoritative experiment decision remains `decision.json`.
