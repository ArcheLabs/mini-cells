"""Isolated Writer and Reader subprocess entry points for EPCLM-002-A0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _install_reader_io_guard(
    allowed_paths: set[Path], *, writable_paths: set[Path] | None = None
) -> None:
    allowed = {path.resolve() for path in allowed_paths}
    writable = {path.resolve() for path in (writable_paths or set())}
    forbidden_raw = os.environ.get("EPCLM_FORBIDDEN_ROOT")
    forbidden = Path(forbidden_raw).resolve() if forbidden_raw else None
    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args:
            value = args[0]
            if isinstance(value, int):
                return
            try:
                path = Path(os.fsdecode(value)).resolve()
            except (TypeError, ValueError, OSError):
                return
            if forbidden is not None:
                try:
                    path.relative_to(forbidden)
                except ValueError:
                    pass
                else:
                    raise PermissionError(f"Reader access to removed raw-data root denied: {path}")
            if path not in allowed:
                raise PermissionError(f"Reader file access outside the explicit allowlist denied: {path}")
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else None
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            is_write = mode is not None and mode[:1] in {"w", "a", "x"}
            is_read_only = mode is not None and mode[:1] == "r"
            if mode is None:
                access_mode = flags & os.O_ACCMODE
                is_write = access_mode in {os.O_WRONLY, os.O_RDWR}
                is_read_only = access_mode == os.O_RDONLY
            if path in writable:
                if not is_write:
                    raise PermissionError(f"Reader may only write its result file: {path}")
            elif is_write or not is_read_only:
                raise PermissionError(f"Reader may only read input files: {path}")
            return
        if event in {
            "os.listdir", "os.scandir", "os.chdir", "subprocess.Popen", "os.system",
            "os.exec", "os.spawn", "os.fork", "os.forkpty", "socket.connect",
            "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
        }:
            raise PermissionError(f"Reader operation is outside the registered protocol: {event}")

    sys.addaudithook(guard)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _writer(args: argparse.Namespace) -> int:
    source = Path(args.input).resolve()
    target = Path(args.state).resolve()
    payload = _read_json(source)
    events = payload["events"]
    if args.backend == "direct-address":
        facts: dict[str, int] = {}
        for event in events:
            if event["kind"] == "fact":
                facts[str(int(event["key"]))] = int(event["value"])
            elif event["kind"] != "distractor":
                raise ValueError(f"unknown event kind: {event['kind']}")
        state = {
            "format": "minicells.epclm-002-a0.direct-state.v1",
            "episode_id": payload["episode_id"],
            "write_event_count": len(events),
            "fact_write_count": sum(event["kind"] == "fact" for event in events),
            "cells": facts,
        }
        _write_json(target, state)
    elif args.backend in {"learned-cells", "flat-recurrent"}:
        import torch

        from .data import event_tensors
        from .models import ModelConfig, build_model

        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["model_name"] != args.backend:
            raise ValueError("checkpoint backend does not match requested Writer backend")
        config = ModelConfig(**checkpoint["model_config"])
        model = build_model(args.backend, config)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        keys, values, is_fact, valid = event_tensors(events)
        with torch.no_grad():
            if args.backend == "learned-cells":
                memory = model.encode_events(keys, values, is_fact, valid)
            else:
                memory = model.write_events(keys, values, is_fact, valid)
        state_payload = {
                "format": "minicells.epclm-002-a0.learned-state.v1",
                "episode_id": payload["episode_id"],
                "event_count": len(events),
                "state": memory.cpu(),
                "logical_state_bytes": int(memory.numel() * memory.element_size())
                + (int(valid.numel()) if args.backend == "learned-cells" else 0),
            }
        if args.backend == "learned-cells":
            state_payload["valid"] = valid.cpu()
        torch.save(state_payload, target)
    else:
        raise ValueError(f"unsupported backend: {args.backend}")
    return 0


def _reader(args: argparse.Namespace) -> int:
    result_path = Path(args.result).resolve()
    query_path = Path(args.queries).resolve()
    state_path = Path(args.state).resolve() if args.state else None
    torch = None
    query_tensor = None
    ModelConfig = None
    model = None
    if args.backend in {"learned-cells", "flat-recurrent"}:
        # Initialize lazy PyTorch imports before restricting file access. These
        # warmups contain no episode data and use only in-memory tensors.
        import io

        import torch

        from .data import query_tensor
        from .models import ModelConfig, build_model

        config = ModelConfig(max_events=128, cell_dim=16)
        model = build_model(args.backend, config).eval()
        if args.backend == "learned-cells":
            dummy_state = torch.zeros((1, config.max_events, config.cell_dim))
            dummy_valid = torch.ones((1, config.max_events), dtype=torch.bool)
        else:
            dummy_state = torch.zeros((1, config.state_budget_bytes // 4))
            dummy_valid = torch.ones((1, 1), dtype=torch.bool)
        with torch.no_grad():
            model(dummy_state, dummy_valid, torch.zeros((1, config.key_bits)))
        warm_buffer = io.BytesIO()
        torch.save({"warmup": torch.zeros(1)}, warm_buffer)
        warm_buffer.seek(0)
        torch.load(warm_buffer, map_location="cpu", weights_only=True)
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    allowed = {query_path, result_path}
    if state_path is not None and state_path.is_file():
        allowed.add(state_path)
    if checkpoint_path is not None:
        allowed.add(checkpoint_path)
    _install_reader_io_guard(allowed, writable_paths={result_path})
    queries_payload = _read_json(query_path)
    query_ids = [int(value) for value in queries_payload["query_keys"]]
    predictions: list[int | None] = []
    read_cells: list[int] = []
    candidate_cells: list[int] = []
    full_scans: list[bool] = []
    file_hashes: dict[str, str] = {str(query_path): _digest(query_path)}

    if state_path is None or not state_path.is_file():
        predictions = [None for _ in query_ids]
        read_cells = [0 for _ in query_ids]
        candidate_cells = [0 for _ in query_ids]
        full_scans = [False for _ in query_ids]
        status = "abstain_no_state"
    elif args.backend == "direct-address":
        state = _read_json(state_path)
        cells = state["cells"]
        predictions = [cells.get(str(query)) for query in query_ids]
        read_cells = [1 if str(query) in cells else 0 for query in query_ids]
        candidate_cells = [len(cells) for _ in query_ids]
        full_scans = [False for _ in query_ids]
        file_hashes[str(state_path)] = _digest(state_path)
        status = "ok"
    elif args.backend in {"learned-cells", "flat-recurrent"}:
        if checkpoint_path is None:
            raise ValueError("learned Reader requires a model checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        serialized = torch.load(state_path, map_location="cpu", weights_only=True)
        config = ModelConfig(**checkpoint["model_config"])
        if config != ModelConfig(max_events=128, cell_dim=16):
            raise ValueError("checkpoint architecture differs from frozen A0 model configuration")
        model.load_state_dict(checkpoint["model_state"])
        state = serialized["state"]
        valid = serialized.get("valid")
        if valid is None:
            valid = torch.ones((1, 1), dtype=torch.bool)
        else:
            valid = valid.bool()
        with torch.no_grad():
            for query in query_ids:
                q = query_tensor(query)
                logits, _scores = model(state, valid, q)
                bits = (logits[0] >= 0).to(torch.int64).tolist()
                prediction = 0
                for bit in bits:
                    prediction = (prediction << 1) | int(bit)
                predictions.append(prediction)
                if args.backend == "learned-cells":
                    count = int(state.shape[1])
                    read_cells.append(count)
                    candidate_cells.append(count)
                    full_scans.append(True)
                else:
                    read_cells.append(1)
                    candidate_cells.append(1)
                    full_scans.append(False)
        file_hashes[str(state_path)] = _digest(state_path)
        file_hashes[str(checkpoint_path)] = _digest(checkpoint_path)
        status = "ok"
    else:
        raise ValueError(f"unsupported backend: {args.backend}")

    _write_json(
        result_path,
        {
            "format": "minicells.epclm-002-a0.reader-result.v1",
            "reader_pid": os.getpid(),
            "status": status,
            "query_episode_id": queries_payload["episode_id"],
            "predictions": predictions,
            "read_cells": read_cells,
            "candidate_cells": candidate_cells,
            "full_scan": full_scans,
            "files_read": sorted(file_hashes),
            "input_sha256": file_hashes,
            "raw_data_root_denied": bool(os.environ.get("EPCLM_FORBIDDEN_ROOT")),
            "filesystem_allowlist_enforced": True,
            "child_process_and_network_guard_enforced": True,
            "raw_history_or_kv_received": False,
            "logical_state_bytes": int(
                serialized["logical_state_bytes"]
                if state_path is not None and state_path.is_file() and args.backend in {"learned-cells", "flat-recurrent"}
                else (state_path.stat().st_size if state_path is not None and state_path.is_file() else 0)
            ),
        },
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("writer", "reader"))
    parser.add_argument("--backend", required=True)
    parser.add_argument("--input")
    parser.add_argument("--state")
    parser.add_argument("--queries")
    parser.add_argument("--result")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    code = _writer(args) if args.mode == "writer" else _reader(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
