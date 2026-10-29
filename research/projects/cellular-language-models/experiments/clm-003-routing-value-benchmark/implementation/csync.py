"""Label-free C-sync engineering adapter. No scientific launcher or quality selector."""

import copy
import hashlib
import math
import time

import routing_value as rv
import torch
from minicells_r3e.graph import GraphConnection
from minicells_r3f.evaluation import states, trust
from minicells_r3f_v2.admission import decide


def probe_identity(inputs):
    return rv.sha([x.detach().cpu().tolist() for x in inputs])


def validate_probe(probe, expected):
    if set(probe) != {"role", "inputs", "sha256"} or probe["role"] != "routing_probe":
        raise RuntimeError("CSYNC_UNLABELLED_PROBE_ONLY")
    if not expected or probe["sha256"] != expected or probe_identity(probe["inputs"]) != expected:
        raise RuntimeError("CSYNC_PROBE_IDENTITY_MISMATCH")


def deterministic(model):
    if type(model.connection) is not GraphConnection:
        raise RuntimeError("CSYNC_UNSUPPORTED_TOPOLOGY")
    if model.active_step is not None or model.training_context is not None:
        raise RuntimeError("CSYNC_DETERMINISTIC_PROBE_REQUIRED")


def direction(state, seed):
    """CPU generator independent of global, A and training RNG; no A vector argument."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = {
        k: torch.randn(v.shape, generator=g, dtype=torch.float64) for k, v in sorted(state.items())
    }
    norm = sum(v.square().sum() for v in raw.values()).sqrt()
    return {k: v / norm for k, v in raw.items()}


@torch.no_grad()
def accepted_event(before, after, *, seed, step, accepted, manifest, probe):
    """Called around A's existing update; exports geometry only, never A parameters."""
    validate_probe(probe, manifest["probe"]["identity"])
    if step not in manifest["update_steps"]:
        raise RuntimeError("CSYNC_UNREGISTERED_CHECKPOINT")
    deterministic(before)
    deterministic(after)
    event = {
        "seed": seed,
        "step": step,
        "accepted": bool(accepted),
        "attempted": True,
        "probe_sha256": probe["sha256"],
        "A_before": rv.fingerprint(before.connection.state_dict()),
        "A_after": rv.fingerprint(after.connection.state_dict()),
    }
    if accepted:
        event["geometry"] = trust(
            before, after, probe["inputs"], fixed=states(before, probe["inputs"])
        )
    elif event["A_before"] != event["A_after"]:
        raise RuntimeError("CSYNC_REJECTED_A_CHANGED_CONNECTION")
    return event


def matches(metrics, target, config):
    return all(
        math.isfinite(metrics[k])
        and math.isfinite(target[k])
        and abs(metrics[k] - target[k])
        <= max(config["absolute_tolerance"][k], config["relative_tolerance"] * abs(target[k]))
        for k in config["metrics"]
    )


@torch.no_grad()
def synchronize(engine, event, probe, manifest, *, log_path):
    """Invoke as C's pre-checkpoint failure_hook after its corresponding Cell update.

    Persist the returned event in the same checkpoint transaction as the Connection.
    This adapter deliberately offers no automatic recovery of an uncommitted event.
    """
    start = time.perf_counter()
    validate_probe(probe, manifest["probe"]["identity"])
    if (
        event["step"] != engine.step
        or event["seed"] != engine.model.seed
        or event["step"] not in manifest["update_steps"]
        or event["probe_sha256"] != probe["sha256"]
    ):
        raise RuntimeError("CSYNC_EVENT_CONTEXT_MISMATCH")
    if getattr(engine, "screen_arm", None) != "C-sync":
        raise RuntimeError("CSYNC_WRONG_ARM")
    # An event file means a prior invocation may have changed parameters. Never retry blindly.
    if log_path.exists():
        raise RuntimeError("CSYNC_EVENT_ALREADY_ATTEMPTED_RESTORE_CHECKPOINT_REQUIRED")
    model = engine.model
    deterministic(model)
    initial = copy.deepcopy(model.connection.state_dict())
    report = {
        "seed": event["seed"],
        "step": event["step"],
        "A": event,
        "before": rv.fingerprint(initial),
        "status": "STARTED",
        "attempts": [],
        "cost": {
            "optimizer_calls": 0,
            "training_tokens": 0,
            "candidate_evaluations": 0,
            "cpu_sampling_seconds": 0.0,
            "gpu_used": next(model.parameters()).is_cuda,
        },
        "topology": {"type": "GraphConnection", "actions": 6, "structure_changed": False},
    }
    rv.atomic(log_path, report)
    if not event["accepted"]:
        report["status"] = "A_REJECTED_NO_C_UPDATE"
    else:
        target = event["geometry"]
        report["target"] = {k: target[k] for k in manifest["matching"]["metrics"]}
        with engine.journal.phase("csync_geometry_search"):
            fixed = states(model, probe["inputs"])
            candidate = copy.deepcopy(model)
            found = False
            for index in range(manifest["random"]["K"]):
                # Hash-based seed avoids collisions across events and seeds.
                raw = f"csync-v1:{manifest['random']['seed_offset']}:{event['seed']}:{event['step']}:{index}"
                ds = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") % (2**63)
                tick = time.perf_counter()
                vector = direction(initial, ds)
                report["cost"]["cpu_sampling_seconds"] += time.perf_counter() - tick
                for scale in manifest["random"]["scales"]:
                    state = {k: v + scale * vector[k].to(v) for k, v in initial.items()}
                    candidate.connection.load_state_dict(state, strict=True)
                    metrics = trust(model, candidate, probe["inputs"], fixed=fixed)
                    historical = decide(metrics, engine.c, float(model.route_numeric_tolerance))
                    legal = (
                        historical["passed"]
                        and metrics["parameter_delta_norm"] > 0
                        and all(
                            metrics[k] <= manifest["trust"][k + "_max"]
                            for k in manifest["matching"]["metrics"]
                        )
                    )
                    matched = legal and matches(metrics, target, manifest["matching"])
                    row = {
                        "direction_index": index,
                        "direction_seed": ds,
                        "scale": scale,
                        "legal": legal,
                        "matched": matched,
                        "metrics": metrics,
                        "connection_sha256": rv.fingerprint(state),
                        "parameter_delta_rms": metrics["parameter_delta_norm"]
                        / math.sqrt(sum(v.numel() for v in initial.values())),
                    }
                    report["attempts"].append(row)
                    report["cost"]["candidate_evaluations"] += 1
                    if matched:
                        found = True
                        model.connection.load_state_dict(state, strict=True)
                        report["status"] = "MATCHED"
                        break
                if found:
                    break
            if not found:
                report["status"] = "UNMATCHABLE"
    report["after"] = rv.fingerprint(model.connection.state_dict())
    report["cost"]["wall_seconds"] = time.perf_counter() - start
    report["cost"]["legal_failures"] = sum(not r["legal"] for r in report["attempts"])
    report["cost"]["directions_attempted"] = len({r["direction_index"] for r in report["attempts"]})
    engine.frozen_sha = rv.fingerprint(
        {n: p for n, p in model.named_parameters() if not p.requires_grad}
    )
    engine.curve[-1]["csync"] = report
    rv.atomic(log_path, report)
    return report


def feasibility(events, manifest):
    triggered = [e for e in events if e["A"]["accepted"]]
    unmatched = sum(e["status"] == "UNMATCHABLE" for e in triggered)
    return {
        "opportunities": len(events),
        "A_accepted": len(triggered),
        "C_matched": sum(e["status"] == "MATCHED" for e in triggered),
        "C_unmatched": unmatched,
        "unmatched_fraction": unmatched / len(triggered) if triggered else None,
        "mechanism_interpretable": bool(triggered)
        and unmatched / len(triggered) <= manifest["matching"]["maximum_unmatched_fraction"],
        "scientific_go": False,
        "note": "Geometry feasibility alone never authorizes training or establishes quality effects.",
    }
