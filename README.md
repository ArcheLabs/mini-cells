# MINI Cells

MINI Cells explores whether useful language behavior can emerge from small, local, shared neural-cell dynamics and eventually continue evolving natively on JAM. The first capability is Echo: a bounded neural field receives a short sequence and learns to reproduce it. This is an engineering copy task, not a claim that the model is alive or conscious.

## Honest project status

- **Experiment 001A — PyTorch/Adam Echo architecture: PASS.** The preserved Python/Kaggle research validates the architecture under its original floating-point training setup.
- **Rust/PVM/MiniJAM runtime: implemented; validation evidence is tracked in [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).** It includes a deterministic Q8.8 kernel, exact 4,476-parameter model, SIGN-SPSA generation pairing, explicit protocol, simulator, no_std service artifact, CLI, and browser client.
- **MiniJAM-native Echo learning: NOT YET VALIDATED.** A generation transition proves deterministic protocol execution; it does not by itself establish useful learning quality. No claim that “AI learned on MiniJAM” is made in this implementation round.

Compatibility is pinned to JAM semantics 0.7.2 and the exact MiniJAM/Jambda/toolchain refs recorded in [`service/artifacts/manifest.json`](service/artifacts/manifest.json). The V0.1 production path is direct MiniJAM plus a non-canonical Keeper; the browser is a Keeper/SSE view and has no wallet or Playground dependency. See [`docs/architecture-v0.1.md`](docs/architecture-v0.1.md), [`docs/direct-minijam.md`](docs/direct-minijam.md), [`docs/keeper.md`](docs/keeper.md), and [`docs/protocol-v1.md`](docs/protocol-v1.md).

## Validate and build

Python research requires Python 3.11 or newer with the development dependencies. Repository-owned validation is:

```bash
python -m pip install -e '.[dev]'
./tools/test_all.sh
```

Build the Rust service artifact directly with:

```bash
./tools/build_service.sh
```

For a running compatible MiniJAM stack, follow [`docs/deployment.md`](docs/deployment.md) or run the end-to-end procedure in [`docs/smoke-test.md`](docs/smoke-test.md). The browser app is under `apps/web`.

The original research commands remain available:

```bash
python scripts/train_echo.py --config configs/echo-v0.yaml
python scripts/eval_echo.py --checkpoint results/echo-v0/checkpoints/best.pt
python scripts/sample_echo.py --checkpoint results/echo-v0/checkpoints/best.pt --text "hello jam"
```

The Kaggle notebook is [`research/kaggle/experiment-001-echo.ipynb`](research/kaggle/experiment-001-echo.ipynb); reusable behavior remains in the `research/minicells` package rather than the notebook.
