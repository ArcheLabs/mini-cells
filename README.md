# MINI Cells

MINI Cells explores whether useful language behavior can emerge from small, local, shared neural-cell dynamics and eventually continue evolving natively on JAM. The first capability is Echo: a bounded neural field receives a short sequence and learns to reproduce it. This is an engineering copy task, not a claim that the model is alive or conscious.

## CLM-0.1 Research Preview

The current language-model research line has reached its first release candidate: **MiniCells CLM-0.1**. CLM-0.1 upcycles the 10M-token TextNCA checkpoint into four inherited full-width experts per recurrent stage while activating only one expert for each local state. The public release is intentionally blocked until the reproducible release build passes **Conditionality Validation 002**.

The previous Upcycling Study 001 established the release direction: copied experts preserve the pretrained function at initialization, then differentiate under route-conditioned experience; both upcycled arms beat matched dense continuation, and Dynamic routing beat matched Static and sample-Shuffled controls. CLM-0.1 freezes that mechanism into a loadable model instead of adding phenotype, growth, multimodality, or self-learning prematurely.

Release contract and reproduction instructions are in [`docs/CLM_0_1_RELEASE.md`](docs/CLM_0_1_RELEASE.md). The final Kaggle release build is [`research/kaggle/clm-0.1-release.ipynb`](research/kaggle/clm-0.1-release.ipynb).

Once the release artifacts are published, the public Python API is:

```python
from minicells import CLM

model = CLM.from_pretrained("artifacts/releases/clm-0.1")
print(model.generate("Once upon a time", max_new_tokens=48, seed=7))
```

The CLI exposes the same bundle:

```bash
minicells-clm artifacts/releases/clm-0.1 "Once upon a time" --max-new-tokens 48
```

CLM-0.1 does **not** claim sub-dense active compute, wall-clock speedup, autonomous growth, online self-learning, phenotype, multimodality, or general-purpose chat quality. It is a small TinyStories research model whose release claim is limited to function-preserving capacity expansion and causally useful local conditional computation.

## Honest project status

- **Experiment 001A — PyTorch/Adam Echo architecture: PASS.** The preserved Python/Kaggle research validates the architecture under its original floating-point training setup.
- **Rust/PVM/MiniJAM runtime: implemented; validation evidence is tracked in [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).** It includes a deterministic Q8.8 kernel, exact 4,476-parameter model, SIGN-SPSA generation pairing, explicit protocol, simulator, no_std service artifact, CLI, and browser client.
- **MiniJAM-native Echo learning: NOT YET VALIDATED.** A generation transition proves deterministic protocol execution; it does not by itself establish useful learning quality. No claim that “AI learned on MiniJAM” is made in this implementation round.

Compatibility is pinned to JAM semantics 0.7.2 and the exact MiniJAM/Jambda/toolchain refs recorded in [`service/artifacts/manifest.json`](service/artifacts/manifest.json). V0.2 keeps MiniJAM canonical and the Keeper non-canonical: wallet authentication establishes an HttpOnly session, the Keeper serves a verified finalized model, and ordinary browser inference runs locally through the Rust/WASM kernel. PVM inference remains an explicitly authenticated verification/debug path; there is no Playground dependency. See [`docs/architecture-v0.1.md`](docs/architecture-v0.1.md), [`docs/direct-minijam.md`](docs/direct-minijam.md), [`docs/keeper.md`](docs/keeper.md), and [`docs/protocol-v1.md`](docs/protocol-v1.md).

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

For a fresh checkout, bootstrap the pinned MiniJAM/Jambda sources before an
offline build, then build the browser ABI as part of the web production build:

```bash
./tools/bootstrap_deps.sh
npm --prefix apps/web run build
```

Keeper authentication uses `MINICELLS_WEB_ORIGIN`,
`MINICELLS_OPERATOR_ACCOUNT` (optional), and `MINICELLS_COOKIE_SECURE=1` for
HTTPS. The web app uses `VITE_MINICELLS_KEEPER_URL`; users connect a normal
sr25519 Polkadot extension, sign the displayed challenge bytes, and keep the
verified model only in memory. V0.2 treats the Keeper as the trusted gateway
for finalized MiniJAM state: the browser independently checks that delivered
model bytes match the hash supplied by that gateway, but does not yet verify a
direct MiniJAM proof that the hash came from canonical service storage.

For a running compatible MiniJAM stack, follow [`docs/deployment.md`](docs/deployment.md) or run the end-to-end procedure in [`docs/smoke-test.md`](docs/smoke-test.md). The browser app is under `apps/web`.

The original research commands remain available:

```bash
python scripts/train_echo.py --config configs/echo-v0.yaml
python scripts/eval_echo.py --checkpoint results/echo-v0/checkpoints/best.pt
python scripts/sample_echo.py --checkpoint results/echo-v0/checkpoints/best.pt --text "hello jam"
```

The Kaggle notebook is [`research/kaggle/experiment-001-echo.ipynb`](research/kaggle/experiment-001-echo.ipynb); reusable behavior remains in the `research/minicells` package rather than the notebook.
