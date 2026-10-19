"""One-launch read-only audit, frozen parity, Calibration budget and resumable CF work."""

from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError
from clm001_r3.data import load_assets
from clm001_r3.oracle import Snapshot
from clm001_r3b import runner, full_analysis as frozen, instrumentation as obs
from clm001_r3b.analysis_runner import fingerprint
from kaggle_posthoc.kaggle_source import discover, locate_training, fetch, relative
from .transport import RecoverableKaggleOutputAPI as KaggleOutputAPI
from .common import ENGINE_SHA, EXPERIMENT, POLICIES, configuration, protocol_identity
from .preservation import audit_old
from .validation import load_model, baseline_parity
from .inference import census, rollout, statistics, reconstruct
from .sampling import sample
from .counterfactual import GreedySnapshot, audit_state
from .storage import Chunks
from .reporting import interventions, cf1_summary, census_summary, findings
from .publication import prepare, publish


class SegmentComplete(Exception):
    """Cooperative boundary after a committed chunk; never interrupts an operator."""


def restore_saved(api, inventories, cache, store):
    restored = 0
    for inv in sorted(inventories, key=lambda i: i["version_number"], reverse=True):
        names = [r["name"] for r in inv["files"] if r["name"].endswith("cf-snapshot.json")]
        for name in names:
            target = Path(cache) / ("cf-prior-" + str(inv["version_number"]))
            fetch(api, inv, [name], target)
            snap = read(target / name)
            if (
                snap.get("schema") != "minicells.cf001.snapshot.v1"
                or snap.get("identity") != store.identity
            ):
                continue
            parent = Path(name).parent
            declared = [(parent / relative(n)).as_posix() for n in snap["files"]]
            available = {r["name"] for r in inv["files"]}
            if not set(declared) <= available:
                raise ContinuationError("CF_SAVED_SNAPSHOT_INCOMPLETE")
            fetch(api, inv, declared, target)
            for rel, sha in snap["files"].items():
                if digest(target / parent / relative(rel)) != sha:
                    raise ContinuationError("CF_SAVED_SNAPSHOT_HASH_MISMATCH")
            restored += store.import_saved(
                target / parent,
                {
                    "kernel": "bingjiangli/minicells",
                    "version": inv["version_number"],
                    "inventory_sha256": inv["inventory_sha256"],
                },
            )
    return restored


def window(data, c, role, index, device):
    meta = data["windows"][role]["evaluation"]
    start = meta["starts"][index]
    length = c["training"]["sequence_length"]
    x = torch.tensor(
        np.array(data[role][start : start + length], dtype="int64")[None], device=device
    )
    y = torch.tensor(
        np.array(data[role][start + 1 : start + length + 1], dtype="int64")[None], device=device
    )
    group = meta["groups"][index]
    wid = canonical(
        {
            "role": role,
            "start": start,
            "group": group,
            "tokens": x.cpu().tolist(),
            "targets": y.cpu().tolist(),
        }
    )
    return x, y, str(group), wid


def clock(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def choose_budget(seconds, c):
    projections = {str(n): float(seconds * n) for n in c["cf1_state_candidates"]}
    valid = [n for n in c["cf1_state_candidates"] if projections[str(n)] <= c["cf1_target_seconds"]]
    chosen = max(valid) if valid else min(c["cf1_state_candidates"])
    return {
        "selected_states": chosen,
        "mc_samples": c["mc_samples"],
        "seconds_per_state": seconds,
        "projected_cf1_seconds": projections,
        "selection_data": "CALIBRATION_ONLY",
        "development_outcomes_used": False,
        "target_is_estimate_not_termination": True,
    }


def run(args, source):
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    if (
        output.resolve() == cache.resolve()
        or output.resolve() in cache.resolve().parents
        or cache.resolve() in output.resolve().parents
        or Path("/kaggle/working") in cache.resolve().parents
    ):
        raise ContinuationError("CF_INPUT_OUTPUT_ISOLATION_REQUIRED")
    c = configuration()
    timing = {}
    phase = started
    stage_start = started
    reports = {}
    prepared = None
    api = None
    publication_phase = False
    store = None
    native = {}
    policies = {p: {} for p in POLICIES[2:]}
    states = []
    records = []
    run_id = {
        "experiment_id": EXPERIMENT,
        "source_sha": source,
        "new_analysis_protocol_sha256": protocol_identity(c),
        "engine_sha": ENGINE_SHA,
        "checkpoint_sha256": c["checkpoint_sha256"]["QC-R3b"],
        "training_updates": 0,
    }
    model = None
    input_files = {}
    before = None
    registry_before = None

    def enter_stage(name):
        nonlocal stage_start
        timing["stage"] = name
        stage_start = time.perf_counter()

    def finish(status, reason=None, allow_publication=True):
        nonlocal reports, publication_phase
        if before is not None and (
            fingerprint(model.state_dict()) != before
            or model.registry.fingerprint != registry_before
        ):
            raise ContinuationError("CF_MODEL_OR_REGISTRY_MUTATED")
        for file, sha in input_files.items():
            if digest(file) != sha:
                raise ContinuationError("CF_FROZEN_INPUT_MUTATED")
        if store:
            store.snapshot(output)
            covered = list(store.index.values())
            durable = sum(r["cross_session_persistence_verified"] for r in covered)
        else:
            covered = []
            durable = 0
        timing["session_wall_seconds_before_publication"] = time.perf_counter() - started
        timing["classification"] = "MEASURED_UNLESS_MARKED"
        timing["saved_version_retention_seconds"] = {
            "value": None,
            "classification": "NOT_MEASURED",
        }
        timing["gpu_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(args.device)
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else None
        )
        timing["per_task_gpu_time"] = (
            "CUDA event milliseconds in private chunk receipts; CPU runs report null"
        )
        timing["hf_commits"] = 0
        if api is not None and hasattr(api, "network_metrics"):
            timing["network_requests"] = api.network_metrics
        timing["training_updates"] = 0
        reports.update(
            {
                "experiment-manifest.json": {
                    **run_id,
                    "status": status,
                    "reason": reason,
                    "last_execution_stage": timing.get("stage", "BOOTSTRAP"),
                    "confirmation": "SEALED",
                    "scientific_decision": False,
                    "formal_confirmation": False,
                    "output_saved_version": "OUTPUT_VERSION_PENDING_SAVE",
                    "new_chunks": len(covered),
                    "saved_output_readback_verified_chunks": durable,
                    "historical_raw_trace": "NOT_RECORDED",
                },
                "coverage-ledger.json": {
                    "baseline_stories": len(native),
                    "expected_baseline_stories": c["stories"],
                    "intervention_stories": {p: len(v) for p, v in policies.items()},
                    "cf1_states": len(states),
                    "chunk_count": len(covered),
                    "cross_session_verified_chunks": durable,
                    "private_results_in_github": False,
                },
                "resource-ledger.json": timing,
            }
        )
        for name in (
            "cf1-stop-value-calibration.json",
            "cf2-forced-stop.json",
            "cf3-repeat-constraint.json",
            "sampling-summary.json",
            "baseline-parity.json",
            "checkpoint-validation.json",
        ):
            reports.setdefault(name, {"status": "NOT_MEASURED"})
        for name, value in reports.items():
            atomic(output / "public" / name, value)
        (output / "public/findings.md").write_text(findings(reports), encoding="utf-8")
        from .plots import render

        render(reports, output / "public/figures")
        if args.publish_github and allow_publication:
            publication_phase = True
            active = prepared if prepared else prepare(cache, run_id)
            publish(active, reports, run_id, output)
            publication_phase = False
        result = {
            "status": status,
            "reason": reason,
            "training_updates": 0,
            "scientific_decision": False,
            "NEXT_RUN_REQUIRED": status != "COMPUTED_AWAITING_SAVE" or durable < len(covered),
            "private_output": str(output),
            "output_saved_version": "OUTPUT_VERSION_PENDING_SAVE",
        }
        atomic(output / "result.json", result)
        print(result, flush=True)
        return result

    try:
        enter_stage("KAGGLE_DISCOVERY")
        api = KaggleOutputAPI()
        if hasattr(api, "connectivity_probe"):
            enter_stage("EARLY_STORAGE_CONNECTIVITY")
            timing["early_storage_probe"] = api.connectivity_probe(c["training_output_version"])
        enter_stage("KAGGLE_DISCOVERY")
        phase = time.perf_counter()
        inventories, discovery = discover(
            api, c["training_output_version"], minimum_version=c["training_output_version"]
        )
        atomic(
            output / "private-version-inventory.json",
            {"versions": inventories, "discovery": discovery},
        )
        timing["discovery_seconds"] = time.perf_counter() - phase
        phase = time.perf_counter()
        enter_stage("OLD59_SAVED_READBACK")

        def preserve_progress(value):
            reports["old-analysis-preservation.json"] = value
            atomic(output / "public/old-analysis-preservation.json", value)

        audit, reference = audit_old(
            api, inventories, cache, args.orchestrator_root, on_progress=preserve_progress
        )
        reports["old-analysis-preservation.json"] = audit
        timing["old59_readback_seconds"] = time.perf_counter() - phase
        if audit["status"] != "OLD_ANALYSIS_PERSISTENCE_VERIFIED" or reference is None:
            raise ContinuationError("CF_OLD_ANALYSIS_NOT_VERIFIED")
        engine = SimpleNamespace(
            REPORTS=frozen.REPORTS,
            SOURCE=frozen.SOURCE,
            INTERNAL_RUN=frozen.INTERNAL_RUN,
            ENGINE_SHA=ENGINE_SHA,
            verify=runner.verify,
            validate_input=frozen.validate_input,
        )
        checkpoints = read(frozen.REPORTS / "checkpoint-index.json")["checkpoints"]
        timing["expected_checkpoint_transfer"] = {
            "records": len(checkpoints),
            "bytes": sum(r["bytes"] for r in checkpoints),
            "largest_file_bytes": max(r["bytes"] for r in checkpoints),
            "scope": "frozen checkpoint records only; excludes assets, banks and reused files",
            "classification": "DERIVED_FROM_FROZEN_INDEX",
        }
        print({"stage": "CF_INPUT_SIZE_PLAN", **timing["expected_checkpoint_transfer"]}, flush=True)
        phase = time.perf_counter()
        enter_stage("FROZEN_TRAINING_INPUT_DOWNLOAD_VERIFY")
        print({"stage": timing["stage"], "version": c["training_output_version"]}, flush=True)
        inv, root, meta, transfer = locate_training(
            api,
            [i for i in inventories if i["version_number"] == c["training_output_version"]],
            cache,
            engine,
        )
        timing["input_verify_seconds"] = time.perf_counter() - phase
        timing["training_input_transfer"] = transfer
        enter_stage("MODEL_AND_ASSET_RESTORE")
        model, runtime, check = load_model(root, c, args.device)
        reports["checkpoint-validation.json"] = check
        data = load_assets(runtime, root / "assets")
        development = data["windows"]["development"]["evaluation"]
        if len(development["starts"]) != 128 or len(set(development["groups"])) != 128:
            raise ContinuationError("CF_FROZEN_STORY_COUNT_MISMATCH")
        if reference["identity"]["dataset_window_sha256"] != canonical(
            data["windows"]["development"]
        ):
            raise ContinuationError("CF_HISTORICAL_WINDOW_IDENTITY_MISMATCH")
        old = reference["result"]["quality"]["greedy"]
        if old["groups"] != development["groups"]:
            raise ContinuationError("CF_HISTORICAL_WINDOW_GROUP_MISMATCH")
        # Exact pairing identity is inherited from the original evaluator, not user-entered metadata.
        from clm001_r3.common import canonical_sha256

        pairing = canonical_sha256(
            {
                "role": "development",
                "windows": development,
                "seed": runtime["evaluation"]["seed"],
                "budget": 16,
                "policy": "greedy",
                "cost_lambda": 0.01,
            }
        )
        if old.get("original_policy_identity", old["paired_identity"]) != pairing:
            raise ContinuationError("CF_HISTORICAL_WINDOW_PAIRING_MISMATCH")
        run_id["development_windows_sha256"] = canonical(development)
        run_id["asset_sha256"] = data["asset_sha256"]
        store = Chunks(output / "chunks", run_id)
        enter_stage("PRIOR_CF_OUTPUT_RESTORE")
        restored = restore_saved(api, inventories, cache, store)
        timing["restored_chunks"] = restored
        if args.publish_github:
            enter_stage("GITHUB_PREFLIGHT")
            publication_phase = True
            prepared = prepare(cache, run_id)
            publication_phase = False
            timing["github_preflight"] = prepared[2]
        before = fingerprint(model.state_dict())
        registry_before = model.registry.fingerprint
        original_files = {
            name: digest(root / name)
            for name in ("runtime-config.json", "continuation.json", "experiment.lock.json")
        }
        input_files = {root / name: sha for name, sha in original_files.items()}
        selected = (
            __import__("minicells_runtime.store", fromlist=["MountedInputStore"])
            .MountedInputStore(root)
            .select("QC-R3b", 500, 0)
        )
        input_files[root / selected["checkpoint"]] = c["checkpoint_sha256"]["QC-R3b"]
        sampled = Snapshot(model)
        greedy = GreedySnapshot(model)

        def boundary():
            # Cooperative packaging only, no signal/timeout/kill or loss of completed work.
            if time.perf_counter() - started >= args.segment_seconds:
                raise SegmentComplete()

        def task(key, compute):
            existing = store.get(key)
            if existing is not None:
                return existing
            boundary()
            enter_stage(key[0])
            begin = clock(args.device)
            versions = tuple(
                (name, p._version, p.data_ptr(), tuple(p.shape), str(p.dtype))
                for name, p in model.named_parameters()
            )
            registry = model.registry.fingerprint
            rng_before = fingerprint(
                {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                }
            )
            events = (
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                if str(args.device).startswith("cuda")
                else None
            )
            if events:
                events[0].record()
            with obs.isolated(model), torch.no_grad():
                value = compute()
            elapsed = clock(args.device) - begin
            if events:
                events[1].record()
                events[1].synchronize()
            if (
                versions
                != tuple(
                    (name, p._version, p.data_ptr(), tuple(p.shape), str(p.dtype))
                    for name, p in model.named_parameters()
                )
                or registry != model.registry.fingerprint
            ):
                raise ContinuationError("CF_TASK_MUTATED_PARAMETERS_OR_REGISTRY")
            rng_after = fingerprint(
                {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                }
            )
            if rng_before != rng_after:
                raise ContinuationError("CF_TASK_MUTATED_GLOBAL_RNG")
            value = {
                **value,
                "measured_seconds": elapsed,
                "cuda_elapsed_ms": events[0].elapsed_time(events[1]) if events else None,
                "read_only_task_guard": "ANCHORED_PARAMETER_VERSION_POINTER_SHAPE_DTYPE_AND_REGISTRY_PLUS_EXACT_RNG",
            }
            store.put(key, value)
            print(
                {
                    "stage": key[0],
                    "item": key[1:],
                    "seconds": round(elapsed, 3),
                    "training_updates": 0,
                },
                flush=True,
            )
            return value

        # The budget is committed from Calibration before any Development predictions.
        if store.get(["calibration-budget"]) is None:

            def pilot():
                x, y, story, wid = window(data, runtime, "calibration", 0, args.device)
                out, rs = census(model, x, y, story, wid)
                eligible = [r for r in rs if r["stop_legal"] and not r["actual_stop"]]
                if not eligible:
                    raise ContinuationError("CF_CALIBRATION_HAS_NO_ELIGIBLE_STATE")
                begin = clock(args.device)
                for r in eligible[: c["pilot_states"]]:
                    r = {**r, "design_weight": 1.0, "inclusion_probability": 1.0}
                    state = reconstruct(
                        model, x, out.actions.reshape(-1, model.config["budget"]).cpu().tolist(), r
                    )
                    audit_state(greedy, sampled, state, y.reshape(-1)[r["token_position"]], r, c)
                each = (clock(args.device) - begin) / min(c["pilot_states"], len(eligible))
                return choose_budget(each, c)

            task(["calibration-budget"], pilot)
        budget = store.get(["calibration-budget"])
        reports["sampling-summary.json"] = {"calibration_budget": budget}
        noops = {}
        for i in range(c["stories"]):

            def base(i=i):
                x, y, story, wid = window(data, runtime, "development", i, args.device)
                out, rs = census(model, x, y, story, wid)
                a = statistics(out, y, model.registry)
                b = statistics(rollout(model, x), y, model.registry)
                return {
                    "base": {**a, "window_index": i, "window_id": wid, "story_id": story},
                    "noop": {**b, "window_index": i, "window_id": wid, "story_id": story},
                    "census": rs,
                }

            row = task(["baseline", i], base)
            story = row["base"]["story_id"]
            native[story] = row["base"]
            noops[story] = row["noop"]
            records.extend(row["census"])
        reports["baseline-parity.json"] = baseline_parity(native, noops, reference, c)
        chosen, sampling = sample(records, budget["selected_states"], c["sampling_seed"])
        task(
            ["sampling-freeze"], lambda: {"sample_sha256": canonical(chosen), "manifest": sampling}
        )
        if store.get(["sampling-freeze"])["sample_sha256"] != canonical(chosen):
            raise ContinuationError("CF_SAMPLING_CHANGED")
        reports["sampling-summary.json"] = {"calibration_budget": budget, **sampling}
        for i in range(c["stories"]):

            def intervened(i=i):
                x, y, story, wid = window(data, runtime, "development", i, args.device)
                return {
                    "story_id": story,
                    "policies": {
                        p: {
                            **statistics(rollout(model, x, p), y, model.registry),
                            "window_index": i,
                            "window_id": wid,
                            "story_id": story,
                        }
                        for p in POLICIES[2:]
                    },
                }

            row = task(["interventions", i], intervened)
            for p, value in row["policies"].items():
                policies[p][row["story_id"]] = value
        comparison = interventions(native, policies, c)
        reports["cf2-forced-stop.json"] = {
            **comparison,
            "comparisons": {
                p: v for p, v in comparison["comparisons"].items() if p.startswith("FORCE_STOP")
            },
        }
        reports["cf3-repeat-constraint.json"] = {
            **comparison,
            "comparisons": {
                p: v for p, v in comparison["comparisons"].items() if not p.startswith("FORCE_STOP")
            },
        }
        for n, r in enumerate(chosen):

            def state_audit(r=r):
                baseline = native[r["story_id"]]
                x, y, _, _ = window(
                    data, runtime, "development", baseline["window_index"], args.device
                )
                state = reconstruct(model, x, baseline["actions"], r)
                from clm001_r3.common import tensor_hash

                if tensor_hash(y.reshape(-1)[r["token_position"]]) != r["target_identity"]:
                    raise ContinuationError("CF_TARGET_PAIRING_CHANGED")
                return audit_state(greedy, sampled, state, y.reshape(-1)[r["token_position"]], r, c)

            states.append(task(["cf1", n, canonical(r)], state_audit))
        reports["cf1-stop-value-calibration.json"] = {
            **cf1_summary(states, c, list(native)),
            "census": census_summary(records),
        }
        sampled.assert_integrity()
        sampled.assert_current(model)
        greedy.assert_integrity()
        greedy.assert_current(model)
        if any(digest(root / name) != sha for name, sha in original_files.items()):
            raise ContinuationError("CF_FROZEN_INPUT_MUTATED")
        return finish("COMPUTED_AWAITING_SAVE")
    except SegmentComplete:
        if native and states:
            reports["cf1-stop-value-calibration.json"] = cf1_summary(states, c, list(native))
        if any(policies.values()):
            partial = interventions(native, policies, c)
            reports["cf2-forced-stop.json"] = {"status": "PARTIAL", **partial}
            reports["cf3-repeat-constraint.json"] = {"status": "PARTIAL", **partial}
        return finish("PARTIAL_RESUMABLE")
    except Exception as error:
        from clm001_r3b.common import safe_error

        timing["failed_stage_wall_seconds"] = time.perf_counter() - stage_start
        atomic(
            output / "failure.json",
            {
                **safe_error(error),
                "stage": timing.get("stage", "BOOTSTRAP"),
                "old59_verified": reports.get("old-analysis-preservation.json", {}).get(
                    "verified_saved_results", 0
                ),
                "training_updates": 0,
            },
        )
        finish(
            "BLOCKED", safe_error(error)["failure_code"], allow_publication=not publication_phase
        )
        raise ContinuationError(safe_error(error)["failure_code"]) from None
    finally:
        if api is not None and hasattr(api, "close"):
            api.close()
        if before is not None and (
            fingerprint(model.state_dict()) != before
            or model.registry.fingerprint != registry_before
        ):
            atomic(
                output / "READ_ONLY_GUARD_FAILED.json", {"status": "FAIL", "training_updates": 0}
            )
            raise ContinuationError("CF_MODEL_OR_REGISTRY_MUTATED")
