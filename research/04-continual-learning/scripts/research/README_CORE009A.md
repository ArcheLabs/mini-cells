# Core 009A scripts

- `run_core_validation_009a_seed.py`: one frozen discovery/confirmation seed.
- `report_core_validation_009a.py`: partial/final phase report and discovery winner lock generation.
- `publish_core_validation_009a.py`: authenticated artifact publication and committed winner lock.
- `orchestrate_core_validation_009a.py`: resumable per-seed Kaggle orchestration.

Run discovery first. Confirmation is forbidden until a viable `winner-lock.json` is committed on the branch.
