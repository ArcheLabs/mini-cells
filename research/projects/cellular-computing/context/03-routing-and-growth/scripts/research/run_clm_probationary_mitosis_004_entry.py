#!/usr/bin/env python3
"""Formal CLM-0.3d launcher entrypoint.

Delegates to the preregistered parent runner while routing worker subprocesses
through the tokenizer-interface compatibility entrypoint.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE / "run_clm_probationary_mitosis_004.py"
WORKER_ENTRY = HERE / "run_clm_probationary_mitosis_004_worker_entry.py"

spec = importlib.util.spec_from_file_location("clm_0_3d_parent_impl", PARENT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load CLM-0.3d parent implementation: {PARENT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

_original_command = module._command


def _command(args, replicate: int) -> list[str]:
    command = list(_original_command(args, replicate))
    if len(command) < 2:
        raise RuntimeError("invalid CLM-0.3d worker command")
    command[1] = str(WORKER_ENTRY)
    return command


module._command = _command


if __name__ == "__main__":
    raise SystemExit(module.main())
