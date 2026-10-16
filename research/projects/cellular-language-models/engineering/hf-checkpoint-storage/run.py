"""Reuse the existing continuation package bootstrap; select the HF transport CLI."""

from pathlib import Path
import runpy

ENTRY = Path(__file__).resolve().parents[1] / "kaggle-continuation/run.py"
# Load its existing import-path setup without invoking its CLI.
runpy.run_path(str(ENTRY), run_name="minicells_hf_bootstrap")
from minicells_runtime.hf_run import main  # noqa: E402

if __name__ == "__main__":
    main()
