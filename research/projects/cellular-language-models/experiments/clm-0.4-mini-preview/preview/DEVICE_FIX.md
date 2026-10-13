# CLM-0.4 Preview runtime growth device fix

A Kaggle two-T4 Preview run exposed that dynamically spawned private growth Cells were created with PyTorch's default CPU/float32 placement even when the parent model already lived on CUDA. The failure occurred before the probationary growth candidate could commit.

The fix makes every runtime-spawned private Cell inherit the device and dtype of its parent sparse layer. The regression test moves the parent model to float64 and asserts that both newly spawned growth modules inherit the same device and dtype, so the placement invariant is testable on CPU-only CI as well.

Existing base checkpoints and completed transaction checkpoints remain valid. Preview runs should update to the fixed main branch and resume from the latest checkpoint; base pretraining does not need to be repeated.
