"""Collect complete-loop actions in the same quality forward passes, not a second inference."""

import torch
from minicells_r3h.evaluation import quality
from minicells_r3e.common import fingerprint, canonical


def measured(model, runtime, data, baseline_paths=None):
    paths = []
    before = dict(model.work)

    def capture(module, inputs, result):
        paths.append(result.actions.detach())

    handle = model.register_forward_hook(capture)
    try:
        row = quality(model, runtime, data, "development")
    finally:
        handle.remove()
    row["role"] = data.get("research_role", "SYNTHETIC_ONLY")
    row["windows_sha256"] = canonical(data["windows"]["development"])
    actions = torch.cat(paths, dim=0)
    row.update(
        path_sha256=fingerprint(actions.cpu()),
        operator_rows={k: v - before.get(k, 0) for k, v in model.work.items()},
    )
    if baseline_paths is not None:
        if actions.shape != baseline_paths.shape:
            raise RuntimeError("SEARCH_WINDOW_PAIRING_CHANGED")
        row.update(
            action_change=float((actions != baseline_paths).float().mean()),
            path_change=float((actions != baseline_paths).any(-1).float().mean()),
        )
        row["effective_reorganization"] = bool((actions != baseline_paths).any())
    if model.graph:
        counts = torch.bincount(actions.reshape(-1), minlength=6)
    else:
        counts = torch.bincount(
            model.connection.pairs[actions.reshape(-1)].reshape(-1), minlength=6
        )
    row["cell_usage"] = counts.cpu().tolist()
    return row, actions
