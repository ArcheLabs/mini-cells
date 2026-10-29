# GPU execution semantics

The fixed CPU functional fixture (Seed 431) is disposable and separate from the three registered Pilot Seeds. One actual two-call Connection proposal and full state/local next-forward comparison precede training automatically. Its actual work is separately reported and not represented as scientific training.

The reused optimizer legitimately permits zero gradients for constant counterfactual costs. CPU unit fixtures continue to require a nonzero signal. During actual Pilot training, finite zero-gradient steps are counted and recorded rather than treated as a synthetic unit-fixture failure. This does not replace the required nonzero preflight check or relax any scientific performance criterion. Frozen parameter groups still cannot move.

There is one CUDA event interval around training per arm, not a synchronization-based profiler on every step. Host delays are included in this interval; it is not labeled active kernel time. Existing task/optimizer scalar reductions remain part of the production algorithm. Checkpoints preserve the selected CUDA generator state in addition to the original complete state. CPU preflight is not repeatedly run as a pytest suite inside Kaggle.

Original CPU config/Notebook remain pinned and available. The new GPU entry is gpu_run.py. It fails immediately without CUDA and has no CPU fallback for scientific execution. A private CPU test mode exists only to test code with unit fixtures and cannot publish a Pilot result.
