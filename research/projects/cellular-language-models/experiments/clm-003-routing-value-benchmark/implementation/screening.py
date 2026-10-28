"""Minimal ABC screening; fixed C dates, one manifest, no MoE/Dense implementation."""

import copy
import json
import math
import platform
from pathlib import Path
from statistics import mean, stdev

import routing_value as rv
import torch
from minicells_fast001 import data
from minicells_fast001.evaluation import evaluate
from minicells_fast001.parent import make_engine as original_engine
from minicells_fast002 import checkpoints, recovery
from minicells_fast002.common import historical_reports
from minicells_fast_r1 import assets
from minicells_fast_r1.stageb_engine import Engine
from minicells_r3f.evaluation import states, trust
from minicells_r3k.reuse import bind

BASE = rv.BASE


def manifest():
    m = rv.read(BASE / "SCREENING_MANIFEST.json")
    if (
        rv.digest(BASE / "SCREENING_MANIFEST.json")
        != (BASE / "SCREENING_MANIFEST.sha256").read_text().strip()
    ):
        raise RuntimeError("CLM003_SCREENING_MANIFEST_CHANGED")
    return m


def decision(m, values):
    paired = rv.paired_effects(values["A"], values["B"], values["C"])
    if set(values["A"]) != set(m["seeds"]):
        raise RuntimeError("CLM003_INCOMPLETE_SCREENING_SEEDS")
    summaries = {}
    for key in ("A_minus_B", "A_minus_C"):
        xs = [r[key] for r in paired]
        # df=2 t(0.975) = 4.3026527299; screening intervals, not multiplicity-adjusted tests.
        half = 4.3026527299 * stdev(xs) / math.sqrt(3)
        summaries[key] = {
            "mean": mean(xs),
            "low": mean(xs) - half,
            "high": mean(xs) + half,
            "all_negative": all(v < 0 for v in xs),
            "n": 3,
        }
    good = all(
        v["all_negative"] and v["mean"] <= -m["practical_effect_nats"] for v in summaries.values()
    )
    mixed = all(v["mean"] < 0 for v in summaries.values())
    return {
        "status": "PROPOSE_INDEPENDENT_CONFIRMATORY_EVALUATION"
        if good
        else "INSUFFICIENT_EVIDENCE_ADDITIONAL_SEEDS_REQUIRE_DECISION"
        if mixed
        else "STOP_BEFORE_MOE_DENSE",
        "per_seed": paired,
        "paired_differences": summaries,
        "screening_only": True,
        "automatic_expansion_authorized": False,
        "independent_confirmation_claimed": False,
        "interval_scope": "DESCRIPTIVE_UNADJUSTED_DF2_SEED_T_INTERVALS_NOT_CONFIRMATORY_SIGNIFICANCE",
    }


def random_connection(model, seed):
    # torch.manual_seed also seeds initialized CUDA generators; preserve them too.
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices):
        return rv.random_connection(model, seed)


def engine(m, parent, seed, arm, root, bundle, *, device="cpu", unit=False):
    c = copy.deepcopy(m["runtime_template"])
    c["updates"] = m["steps"]
    c["connection"]["proposal_steps"] = m["update_steps"]
    internal = "B11-H" if arm == "A" else "B10-H"
    ident = {
        "experiment_id": m["experiment_id"],
        "manifest_sha256": rv.sha(m),
        "seed": seed,
        "arm": arm,
    }
    directory = root / "private" / str(seed) / arm
    directory.mkdir(parents=True, exist_ok=True)
    e = bind(original_engine, Engine=Engine)(
        c,
        parent,
        seed,
        internal,
        ident,
        directory / "cost.jsonl",
        device,
        unit=unit,
        parent_sha=m["parent"]["checkpoint_sha256"],
    )
    e.r1_cpu_fixture = unit
    e.stage_directory, e.stage_reports = directory, root / "proposals" / str(seed)
    e.stage_identity, e.stage_refs = ident, bundle
    # Forking preserves the original Cell optimizer; only Connection values are reset.
    initial = random_connection(e.model, seed + 100000)
    e.model.connection.load_state_dict(initial, strict=True)
    e.frozen_sha = rv.fingerprint(
        {n: p for n, p in e.model.named_parameters() if not p.requires_grad}
    )
    e.screen_arm = arm
    return e


def fixed_rewire(e, m, bundle):
    if e.screen_arm != "C" or e.step not in m["update_steps"]:
        return
    with e.journal.phase("random_rewire_path_measurement"):
        old = copy.deepcopy(e.model)
        seed = e.model.seed + 200000 + e.step
        e.model.connection.load_state_dict(random_connection(e.model, seed), strict=True)
        with torch.no_grad():
            fixed = states(old, bundle["inputs"])
            metrics = trust(old, e.model, bundle["inputs"], fixed=fixed)
        e.curve[-1]["random_rewire"] = {
            "step": e.step,
            "seed": seed,
            "metrics": metrics,
            "independent_of_A_acceptance": True,
            "magnitude_matched": False,
        }
        e.frozen_sha = rv.fingerprint(
            {n: p for n, p in e.model.named_parameters() if not p.requires_grad}
        )


def prepare_data(m, folder):
    c = copy.deepcopy(m["runtime_template"])
    d = m["data"]["confirmation"]
    for key, value in [
        ("generation_seeds", d["seed"]),
        ("numeric_ranges", d["range"]),
        ("names", d["names"]),
    ]:
        c["data"][key]["confirm"] = value
    c["experiment_id"] = m["experiment_id"]
    c["data"]["confirm_windows"] = d["records_per_task"]
    generated = data.generate(c, folder / "evaluation")
    assets.training(folder / "train")
    assets.admission(c, folder / "admission")
    # Compare the new untouched pool against all known FAST001/FAST002 and R1 role hashes.
    fields = ["raw_text_sha256", "normalized_text_sha256", "window_sha256"]
    exclusion = {k: set() for k in fields}
    window_parts = {"input": set(), "target": set()}
    import numpy as np

    def exclude_arrays(path, roles):
        for task in ("Story", "Arithmetic"):
            for role in roles:
                for window in np.load(path / (task + "-" + role + ".npy"), mmap_mode="r"):
                    window_parts["input"].add(rv.sha(window[:-1].tolist()))
                    window_parts["target"].add(rv.sha(window[1:].tolist()))

    from minicells_fast001.common import configuration as old_config
    from minicells_fast002.common import config as previous_config

    for number, old in enumerate((old_config(), previous_config())):
        path = folder / ("historical-" + str(number))
        data.generate(old, path)
        exclude_arrays(path, ("train", "confirm"))
        for task in ("Story", "Arithmetic"):
            for role in ("train", "confirm"):
                for row in rv.read(path / (task + "-" + role + ".json")):
                    for key in fields:
                        exclusion[key].add(row[key])
    exclude_arrays(folder / "admission", ("admission",))
    for task in ("Story", "Arithmetic"):
        for row in rv.read(folder / "admission" / (task + "-admission.json")):
            for key, alias in zip(fields, ["raw", "normalized", "window"]):
                exclusion[key].add(row["hashes"][alias])
    for task in ("Story", "Arithmetic"):
        for row in rv.read(folder / "evaluation" / (task + "-confirm.json")):
            if any(row[k] in exclusion[k] for k in fields):
                raise RuntimeError("CLM003_CONFIRMATION_CONTENT_CONFLICT_NO_RESELECTION")
    for task in ("Story", "Arithmetic"):
        for window in np.load(folder / "evaluation" / (task + "-confirm.npy"), mmap_mode="r"):
            if (
                rv.sha(window[:-1].tolist()) in window_parts["input"]
                or rv.sha(window[1:].tolist()) in window_parts["target"]
            ):
                raise RuntimeError("CLM003_CONFIRMATION_INPUT_TARGET_CONFLICT_NO_RESELECTION")
    generated["screening_exclusion"] = {
        "raw_normalized_window_input_target_overlap": 0,
        "coverage": "FAST001_FAST002_TRAIN_CONFIRM_AND_R1_ADMISSION",
        "global_history_coverage_claimed": False,
    }
    return c, generated


def run(args):
    m = manifest()
    if args.publish_existing:
        return bind(rv.transport.previous.old.transport.publish_existing, publish=publish)(
            args.output
        )
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity = {
        "experiment_id": rv.ID,
        "screening_id": m["experiment_id"],
        "source_sha": rv.source(),
        "manifest_sha256": rv.digest(BASE / "SCREENING_MANIFEST.json"),
    }
    identity["run_id"] = rv.sha(identity)[:32]
    root = args.output / identity["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    run_manifest = dict(
        identity,
        identity=identity,
        scope="CLM003_ABC_SCREENING",
        run_status="PREPARING",
        scientific_optimizer_updates=0,
        gpu_used=False,
        environment={
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
    )
    rv.atomic(root / "RUN_MANIFEST.json", run_manifest)
    failure = None
    try:
        bind(rv.transport.previous.old.transport.preflight, _checkout=rv.checkout)(root)
        if not torch.cuda.is_available():
            raise RuntimeError("CLM003_SCREENING_GPU_REQUIRED")
        run_manifest["gpu_used"] = True
        run_manifest["environment"]["gpu"] = torch.cuda.get_device_name(0)
        c, ds = prepare_data(m, args.assets)
        rv.atomic(root / "REGISTRATION.json", {"manifest": m, "data": ds})
        reports = historical_reports(c, args.assets / "historical-reports")
        # Exactly one common A500 parent, not one different Cell state per Seed.
        selection = copy.deepcopy(c)
        selection["seeds"] = [2026200004]
        selection["arms"] = ["A-H"]
        paths, proof = recovery.resolve(
            selection, reports, [], args.assets / "kaggle-output", version=116
        )
        record = rv.read(reports / "jobs/2026200004-A-H.json")
        if record["checkpoint_sha256"] != m["parent"]["checkpoint_sha256"]:
            raise RuntimeError("CLM003_SHARED_PARENT_IDENTITY_MISMATCH")
        parent = checkpoints.load(paths["2026200004-A-H"], record)
        rv.atomic(
            root / "PARENT.json",
            {
                "record_identity": record["identity"],
                "sha256": record["checkpoint_sha256"],
                "cells_previously_coadapted": True,
                "recovery": proof,
            },
        )
        bundle = assets.refs(c, args.assets / "admission", "cuda:0")
        values = {a: {} for a in "ABC"}
        for seed in m["seeds"]:
            for arm in "ABC":
                e = engine(m, parent, seed, arm, root, bundle, device="cuda:0")
                initialization = {
                    "cell_state_sha256": rv.fingerprint(e.model.cells.state_dict()),
                    "connection_sha256": rv.fingerprint(e.model.connection.state_dict()),
                    "cell_optimizer_sha256": rv.fingerprint(e.cell_optimizer.state_dict()),
                    "parent_cells_previously_coadapted": True,
                }
                directory = e.stage_directory
                latest = directory / "latest.pt"
                done = directory / "COMPLETE.json"
                if done.exists():
                    result = rv.read(done)
                    if result["manifest_sha256"] != rv.sha(m) or result[
                        "checkpoint_sha256"
                    ] != rv.digest(latest):
                        raise RuntimeError("CLM003_COMPLETED_JOB_IDENTITY_CHANGED")
                else:
                    stream = data.Stream(e.fast, args.assets / "train", seed, "H", "B", "cuda:0")
                    if latest.exists():
                        # C intentionally changes frozen Connection; restore its registered frozen mask fingerprint first.
                        saved = torch.load(latest, map_location="cpu", weights_only=False)
                        if arm == "C":
                            e.model.connection.load_state_dict(
                                {
                                    k[len("connection.") :]: v
                                    for k, v in saved["state"]["model"].items()
                                    if k.startswith("connection.")
                                }
                            )
                            e.frozen_sha = rv.fingerprint(
                                {n: p for n, p in e.model.named_parameters() if not p.requires_grad}
                            )
                        e.restore(latest, stream)
                    elif len(e.journal.events) != 1:
                        raise RuntimeError("CLM003_UNCERTAIN_WORK_WITHOUT_CHECKPOINT")
                    else:
                        e.save(latest, stream)
                    while e.step < m["steps"]:
                        e.update(
                            stream, latest, [], failure_hook=lambda tr: fixed_rewire(tr, m, bundle)
                        )
                        if e.step % 25 == 0:
                            print("[CLM003]", seed, arm, "STEP", e.step, flush=True)
                    with e.journal.phase("confirm_final"):
                        evaluation = evaluate(
                            e.model, c, args.assets / "evaluation", mechanism=False
                        )
                    e.save(latest, stream)
                    result = {
                        "seed": seed,
                        "arm": arm,
                        "manifest_sha256": rv.sha(m),
                        "initialization": initialization,
                        "checkpoint_sha256": rv.digest(latest),
                        "evaluation": evaluation,
                        "curve": e.curve,
                        "budget": e.journal.summary(),
                        "training_seconds": e.training_seconds,
                    }
                    rv.atomic(done, result)
                rv.atomic(root / "jobs" / (str(seed) + "-" + arm + ".json"), result)
                values[arm][seed] = result["evaluation"]["L"]
                for handle in e._work_handles:
                    handle.remove()
                del e
                torch.cuda.empty_cache()
        rv.atomic(root / "SCREENING_RESULTS.json", decision(m, values))
        run_manifest["run_status"] = "SCREENING_COMPLETE_REVIEW_REQUIRED"
    except Exception as error:  # noqa: BLE001 -- publish partial evidence without secrets
        from minicells_fast_r1.runner import safe_failure

        failure = error
        run_manifest["run_status"] = "SCREENING_BLOCKED_OR_PARTIAL"
        evidence = safe_failure(error)
        if str(error).startswith("CLM003_") and str(error).replace("_", "").isalnum():
            evidence["code"] = str(error)
        rv.atomic(root / "FAILURE.json", evidence)
    finally:
        from minicells_jct001.cost import Journal

        logs = []
        for path in (root / "private").glob("*/*/cost.jsonl"):
            logs.append(
                Journal(path, json.loads(path.read_text().splitlines()[0])["data"]).summary()
            )
        candidates = []
        for path in (root / "private").glob("*/*/proposals/*/cost.jsonl"):
            candidates.append(
                Journal(path, json.loads(path.read_text().splitlines()[0])["data"]).summary()
            )
        cost = {
            "main_updates": sum(x["logical_updates"] for x in logs),
            "cell_optimizer_calls": sum(x["optimizer_calls_by_group"].get("cell", 0) for x in logs),
            "candidate_optimizer_calls": sum(
                x["optimizer_calls_by_group"].get("connection", 0) for x in candidates
            ),
            "uncertain_optimizer_calls": sum(
                x["uncertain_optimizer_calls"] for x in logs + candidates
            ),
            "phase_wall_seconds": [
                {k: v for k, v in x["cpu_seconds_by_phase"].items()} for x in logs
            ],
            "operator_work": [x["executed_operator_work"] for x in logs],
            "gpu_kernel_seconds": None,
        }
        rv.atomic(root / "COST_LEDGER.json", cost)
        run_manifest["scientific_optimizer_updates"] = (
            cost["cell_optimizer_calls"] + cost["candidate_optimizer_calls"]
        )
        run_manifest["artifact_files"] = {
            p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": rv.digest(p)}
            for p in root.rglob("*.json")
            if allowed(p.relative_to(root).as_posix())
        }
        rv.atomic(root / "RUN_MANIFEST.json", run_manifest)
        receipt = publish(root)
    print(
        json.dumps(
            {"status": run_manifest["run_status"], "output": str(root), "publication": receipt}
        ),
        flush=True,
    )
    if failure:
        raise RuntimeError("CLM003_SCREENING_STOPPED_SEE_REPORTS") from None


def allowed(name):
    p = Path(name).parts
    if len(p) == 4 and p[0] == "proposals" and p[1].isdigit() and p[2].replace("-", "").isalnum():
        return p[3] in {"ADMISSION_DECISION.json", "CANDIDATE.json"}
    return name in {
        "REGISTRATION.json",
        "PARENT.json",
        "FAILURE.json",
        "COST_LEDGER.json",
        "SCREENING_RESULTS.json",
    } or (len(p) == 2 and p[0] == "jobs" and p[1].endswith(".json"))


def publish(root):
    def payload(directory):
        # Reuse checked metadata serialization with the screening scope explicitly bound here.
        m = rv.read(directory / "RUN_MANIFEST.json")
        if m["scope"] != "CLM003_ABC_SCREENING":
            raise RuntimeError("CLM003_SCREENING_SCOPE_MISMATCH")
        files = {}
        for n, e in m["artifact_files"].items():
            p = directory / n
            if (
                not allowed(n)
                or p.is_symlink()
                or rv.digest(p) != e["sha256"]
                or p.stat().st_size != e["bytes"]
            ):
                raise RuntimeError("CLM003_SCREENING_REPORT_CHANGED")
            files[n] = p.read_bytes()
            if len(files[n]) > 4_000_000:
                raise RuntimeError("CLM003_REPORT_TOO_LARGE")
        if sum(map(len, files.values())) > 24_000_000:
            raise RuntimeError("CLM003_PAYLOAD_TOO_LARGE")
        files["RUN_MANIFEST.json"] = (json.dumps(m, sort_keys=True, indent=2) + "\n").encode()
        import hashlib

        index = {
            n: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for n, v in files.items()
        }
        return m, files, index, rv.sha(index)

    return bind(
        rv.transport.previous.old.transport.publish,
        payload=payload,
        _checkout=rv.checkout,
        REPORTS=rv.REPORTS,
        EXPERIMENT_ID=rv.ID,
        BRANCH_PREFIX="results/clm003-screen-",
    )(root)
