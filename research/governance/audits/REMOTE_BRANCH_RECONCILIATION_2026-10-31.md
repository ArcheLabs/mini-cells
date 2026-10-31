# Remote Branch Reconciliation — 2026-10-31

This integration branch is based on `841b16affcd6d544ce2ff21c9090e4f56ed038bb`. The source branch refs are preserved; no listed remote branch or `main` was force-updated.

## EPCLM-001 formal result snapshots

The seven E1 formal branches are cumulative snapshots of one run. Their report directories contain 13, 15, 17, 20, 22, 41, and 41 files. The 41-file snapshot from `results/epclm001-e1-formal-6f48cc1f11234947a5ce975c2cf8b801-66d8ea69988e` is used because it is the latest and most complete snapshot. Its `RUN_MANIFEST.json` remains `FORMAL_COMPLETE_REVIEW_REQUIRED`; this does not grant scientific approval.

The report files are copied without rewriting their run or publication metadata. In particular, the preserved `PUBLICATION.json` records the prior published branch `results/epclm001-e1-formal-6f48cc1f11234947a5ce975c2cf8b801-eed590936845`, while the latest branch carries the newer run status and publication index. Reviewers should verify this publication chain before accepting the report as a canonical published result.

## EPCLM-001 controlled retrain

The report from `results/epclm001-retrain-fbf1959b2a364022959ded11c1e4fea6-a9d53e3b6b50` is added as a separate run. Its recorded status is `CONTROLLED_RETRAIN_COMPLETE_E1_BENCHMARK_NOT_AUTHORIZED`. The quality file labels its checks as development-asset validation, not new scientific confirmation; the historical PPL values remain diagnostic only.

## EPCLM-002-A0 and the stale Codex branch

`codex/hybrid-clm-release-preparation-v0.1` was 601 commits behind `main`. Its EPCLM-002-A0 implementation, development artifacts, result publisher, and frozen protocol are ported here. The protocol bytes are unchanged and are stored under the current `research/governance/validations/` layout; runtime and documentation references now point to that canonical location. The governance registry records the observed development-only pilot and keeps `scientific_decision: false`.

Current-main CI, dependency bounds, release workflow, and release files are retained. The branch's retired `research/catalog.yaml`, `research/validations/`, and `research/stages/` organization and its superseded HybridCLM release-preparation changes are not copied. The Kaggle notebook and launcher retain the 7,200-second hard limit; the launcher reserves up to 600 seconds for publishing.

## Source refs

- Retrain: `591a5c2b569d8c1435d1ee2e1604d6cc9662a522`
- E1 formal b64: `c9cddf4aa26c11c503c1abd2ea9a26625f4e5aec`
- E1 formal b660: `6cb07ade92221db7aedbdb8e70c2204825ce5fe2`
- E1 formal b29a: `e7aaff04a5ae807267e3a44a41b5c9249435535a`
- E1 formal 09a3: `cba340a2c925d9e48f2d88f3ae2296e615850217`
- E1 formal e383: `e3fdeccb84f6ef5377987e67e62823880ab52134`
- E1 formal eed: `da585b5e1357bae0fd866c447cda9a34edd5bf4a`
- E1 formal 66d: `29ec1de0182c19ef74316b945e0cfc16e40d1d21`
- Codex: `ed153c6913857d3d20c834fd301d34e32bc5846e`

