# Native MoM 1.3B baseline audit

The clean starting checkout was `codex/minicells-t4-1b-capacity-001` at
`1cf12d789ba2d907dc499e1ec351c3708f8274e4`. Remote branch and main were
verified at that commit and `32048a03fa297cd8ab481d34ed86f115de534169` respectively.

- `apps/mom-native-cells/minicells_native/sdk.py` exposes the SDK. Its default
  340M model is resolved in `loader.py` and constructed by `backend.py`
  from pinned MoM source with Transformers loading information.
- `backend.py` hardcodes the 340M identity and dimensions. `cells.py`
  checks persisted identity and recurrent tensor coverage. Query and
  generation create independent cache streams and use native MoM forwards.
- `run_t4_1b_capacity.py` owns A–F and the measured matrix, `capacity_meter.py`
  samples CUDA and NVML, and `kaggle_t4_1b_capacity.py` publishes through the
  shared experiment publisher. The target manifest previously blocked all GPU work.
- Existing runtime, SDK, fusion and capacity tests cover software behavior.
  The T4 capacity run stopped before model loading, so it provides no limit measurement.

The implementation needs a model registry, checkpoint validation, model-derived
cache dimensions and an independent preflight before the existing capacity matrix.
