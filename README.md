# MINI Cells

MINI Cells explores whether useful language behavior can emerge from small,
local, shared neural-cell dynamics and eventually continue evolving natively
on JAM.

The project begins with the simplest developmental stage: **Echo**. A small
field of neural cells receives a short sequence and learns to reproduce it.
This is a technical copy task presented publicly as a developmental metaphor;
it is not a claim that the model is alive or conscious.

Experiment 001 uses Kaggle only to validate architecture choices cheaply before
moving toward deterministic Rust/PVM execution. Kaggle is not part of the
intended canonical runtime or trust model.

See [the experiment specification](docs/experiment-001-echo.md).

## Local use

Requires Python 3.11 or newer.

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/train_echo.py --config configs/echo-v0.yaml
python scripts/eval_echo.py --checkpoint results/echo-v0/checkpoints/best.pt
python scripts/sample_echo.py --checkpoint results/echo-v0/checkpoints/best.pt --text "hello jam"
```

Open `research/kaggle/experiment-001-echo.ipynb` on Kaggle and run it from top
to bottom. Core behavior lives in the `minicells` package, not in the notebook.
