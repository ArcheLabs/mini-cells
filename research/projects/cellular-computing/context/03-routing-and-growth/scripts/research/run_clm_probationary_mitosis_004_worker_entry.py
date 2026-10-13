#!/usr/bin/env python3
"""Compatibility entrypoint for the CLM-0.3d formal worker.

``prepare_scaling_corpus`` returns a tokenizer path, while the historical
``prepare_arithmetic_cache`` helper accepts a loaded tokenizer object.  Keep the
large preregistered worker unchanged and adapt that interface at process entry.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from minicells.language_data import load_tokenizer


HERE = Path(__file__).resolve().parent
WORKER = HERE / "run_clm_probationary_mitosis_004_worker.py"

spec = importlib.util.spec_from_file_location("clm_0_3d_worker_impl", WORKER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load CLM-0.3d worker implementation: {WORKER}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

_original_prepare_arithmetic_cache = module.prepare_arithmetic_cache


def _prepare_arithmetic_cache(cache_dir: Path, tokenizer_or_path: object):
    tokenizer = (
        load_tokenizer(Path(tokenizer_or_path))
        if isinstance(tokenizer_or_path, (str, Path))
        else tokenizer_or_path
    )
    return _original_prepare_arithmetic_cache(cache_dir, tokenizer)


module.prepare_arithmetic_cache = _prepare_arithmetic_cache


if __name__ == "__main__":
    raise SystemExit(module.main())
