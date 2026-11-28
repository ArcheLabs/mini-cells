"""A3.4: measure the functional effect of A3.3 execution histories."""
import argparse
from contextlib import contextmanager
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time
from unittest.mock import patch
from urllib.request import urlopen

from common import publishable, write
from m1_bridge import ARMS, DATASET_SHA256 as CANONICAL_SHA256
from m1_continuation_isolation import BOUNDS_SHA256, frozen_bounds
from m1_full_state_bridge import BASELINE, DATASET_SHA256, dataset, load_json, scored
from m1_reference_capacity import digest_file
from r1_equivalence import ensure_readonly, live

import m1_process_history as history

FORMAT = "minicells.m1.full-state.functional-impact.v1"
A33_COMMIT = "91e8c1af3ff580f1e90e9ddb493c3d70fd8a7de2"
FIXTURE = (f"https://raw.githubusercontent.com/ArcheLabs/mini-cells/{A33_COMMIT}/"
           "artifacts/experiments/minicells-m1-002a-r1-a3-3-process-history/20eed975ba41/bridge_dataset.json")
FIXED = dict(BT=32, num_warps=4, num_ctas=1, num_stages=3)
LANES = ("P0-A", "P0-E", "A1-E", "A2-E", "B1-E", "B2-E")
CANDIDATE_COUNTS = dict(reference=2, anchor=2, encoding=2, candidates=20, wording=2, combined=20)


class InvalidProtocol(ValueError):
    """The workload or runtime failed a frozen identity check."""


def fixture(out):
    try:
        with urlopen(FIXTURE, timeout=30) as response:
            (out / "bridge_dataset.json").write_bytes(response.read())
        items = dataset(out)
        if len(items) != 12 or len({item["index"] for item in items}) != 12:
            raise InvalidProtocol("Expected 12 unique frozen episodes")
        for item in items:
            if set(item["arms"]) != set(ARMS):
                raise InvalidProtocol("Frozen episode lacks one of the six arms")
            for name, arm in item["arms"].items():
                labels = arm["labels"]
                if len(labels) != 2 or set(labels) != arm["texts"].keys() or len(arm["wordings"]) != 2:
                    raise InvalidProtocol("Expected two write variants and two queries")
                for query in arm["wordings"]:
                    choices = query["candidates"]
                    if len(choices) != CANDIDATE_COUNTS[name] or not set(labels) <= choices.keys():
                        raise InvalidProtocol("Frozen candidate set is incomplete")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InvalidProtocol(f"Frozen Bridge dataset: {exc}") from exc
    write(out / "fixture_manifest.json", dict(
        dataset_url=FIXTURE, dataset_sha256=DATASET_SHA256, canonical_dataset_sha256=CANONICAL_SHA256,
        historical_frozen_bounds_sha256=BOUNDS_SHA256, a33_commit=A33_COMMIT,
        historical_m1_002a=BASELINE, arms=ARMS, episodes=[item["index"] for item in items],
        model_profile="t4-fp16-recurrent-v1", fixed_l2norm=FIXED))


@contextmanager
def segment_label(segments, name):
    segments.append(name)
    try:
        yield
    finally:
        segments.pop()


def bridge_case(mc, item, paths, segments):
    """Keep the original R1 write grouping, full-state restoration and scoring."""
    backbone = mc._backbone
    groups = {}
    rows = []
    for name, arm in item["arms"].items():
        groups.setdefault(tuple(arm["texts"].values()), []).append((name, arm))
    for cases in groups.values():
        first = cases[0][1]
        for label in first["labels"]:
            text = first["texts"][label]
            tokens = backbone.tokenizer.encode(text, add_special_tokens=False)
            cell = mc.cells.create(f"full-{item['index']}-{label}", full_state=True)
            try:
                with segment_label(segments, "bridge/sdk_write"):
                    mc.cells.write(cell.id, text, resume=False)
                state = mc.cells._state(cell.id)

                def live_cache():
                    with segment_label(segments, "bridge/live_prefill"):
                        return live(backbone, [tokens])

                def restored_cache():
                    return state.restore(backbone)

                factories = {"A": live_cache, "E": restored_cache}

                def evaluate_queries():
                    for name, arm in cases:
                        for path in paths:
                            with segment_label(segments, f"bridge/query/{path}"):
                                results = scored(backbone, factories[path], arm, label)
                            for wording, (query, result) in enumerate(zip(arm["wordings"], results, strict=True)):
                                logp = result["logp"]
                                order = list(query["candidates"])
                                if order != list(logp):
                                    raise InvalidProtocol("Candidate scoring changed the frozen order")
                                best = max((key for key in order if key != label), key=logp.__getitem__)
                                rows.append(dict(
                                    episode=item["index"], arm=name, variant=label, wording=wording, path=path,
                                    gold_logp=logp[label], best_distractor=best, best_distractor_logp=logp[best],
                                    correct=result["margin"] > 0, candidate_order=order, **result))
                ensure_readonly(mc, cell, evaluate_queries)
            finally:
                mc.cells.delete(cell.id)
    return rows


def worker(out, label):
    segments = []
    bridge_launches = []
    run_probe = history.trial
    pin_norm = history.fixed_forward
    check_runtime = history.check_environment
    paths = ("A", "E") if label == "P0" else ("E",)
    active_calls = None

    @contextmanager
    def pin(module, calls, segment):
        nonlocal active_calls
        active_calls = calls
        with pin_norm(module, calls, lambda: segments[-1] if segments else segment()) as chosen:
            if chosen != FIXED:
                raise InvalidProtocol("Fixed L2 Norm policy differs from A3.3")
            yield chosen

    def check(mc):
        try:
            env = check_runtime(mc)
        except RuntimeError as exc:
            raise InvalidProtocol(str(exc)) from exc
        if len(mc._backbone.model.model.layers) != 24:
            raise InvalidProtocol("Frozen backbone must have 24 layers")
        return env

    def probe_then_bridge(mc, root, trial_label, *args):
        run_probe(mc, root, trial_label, *args)
        probe = load_json(root / "work" / f"{trial_label}.json")
        if probe["start_failures"] or not probe.get("bundle_unchanged"):
            raise InvalidProtocol(f"{label}: A3.3 probe start or Bundle integrity mismatch")
        with (root / "work" / f"cases-{label}.jsonl").open("w", encoding="utf-8") as stream:
            for item in dataset(root):
                rows = bridge_case(mc, item, paths, segments)
                for row in rows:
                    stream.write(json.dumps(dict(worker=label, lane=f"{label}-{row['path']}", **row),
                                            allow_nan=False) + "\n")
                stream.flush()
                print(f"A3_4_EPISODE_DONE {label} {item['index']}", flush=True)
        # Leave A3.3's own coverage check intact. Retain the additional Bridge launches separately.
        for key in sorted(active_calls):
            if key[0].startswith("bridge/"):
                segment, d, t, nb, bd, bt, warps, stages, ctas = key
                bridge_launches.append(dict(segment=segment, D=d, T=t, NB=nb, BD=bd, BT=bt,
                                            num_warps=warps, num_stages=stages, num_ctas=ctas,
                                            calls=active_calls.pop(key)))

    with (patch.object(history, "trial", probe_then_bridge),
          patch.object(history, "fixed_forward", pin),
          patch.object(history, "check_environment", check)):
        history.worker(out, label)
    expected = {"bridge/sdk_write", *(f"bridge/query/{path}" for path in paths)}
    if label == "P0":
        expected.add("bridge/live_prefill")
    if {row["segment"] for row in bridge_launches} != expected:
        raise RuntimeError(f"{label}: incomplete Bridge L2 Norm launch coverage")
    policy = load_json(out / f"policy-{label}.json")
    write(out / f"worker-{label}.json", dict(
        worker=label, pid=os.getpid(), lanes=[f"{label}-{path}" for path in paths],
        expected_query_count=288 * len(paths), history=policy, bridge_launches=bridge_launches))


def index_cases(out):
    items = dataset(out)
    expected = {(item["index"], arm, label, wording): query
                for item in items for arm, values in item["arms"].items()
                for label in values["labels"] for wording, query in enumerate(values["wordings"])}
    lanes = {lane: {} for lane in LANES}
    bundle = out / "functional_bridge_cases.jsonl"
    with bundle.open("w", encoding="utf-8") as target:
        for worker_label in history.ORDER:
            receipt = load_json(out / f"worker-{worker_label}.json")
            expected_lanes = {lane for lane in LANES if lane.startswith(worker_label + "-")}
            if set(receipt["lanes"]) != expected_lanes or receipt["pid"] != receipt["history"]["pid"]:
                raise InvalidProtocol(f"{worker_label}: Bridge and probe did not share one process")
            count = 0
            with (out / "work" / f"cases-{worker_label}.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    key = tuple(row[k] for k in ("episode", "arm", "variant", "wording"))
                    lane = row["lane"]
                    if key not in expected or lane not in expected_lanes or key in lanes[lane]:
                        raise InvalidProtocol(f"Unexpected or duplicated functional row: {lane} {key}")
                    query = expected[key]
                    if (row["candidate_order"] != list(query["candidates"]) or row["query"] != query["text"]
                            or row["path"] != lane[-1] or row["worker"] != worker_label):
                        raise InvalidProtocol(f"Functional row disagrees with the frozen query: {key}")
                    lanes[lane][key] = row
                    count += 1
                    target.write(line)
            if count != receipt["expected_query_count"]:
                raise InvalidProtocol(f"{worker_label}: unexpected query count")
    if any(rows.keys() != expected.keys() for rows in lanes.values()):
        raise InvalidProtocol("Incomplete functional matrix")
    return lanes


def gold_order(row):
    return tuple((row["gold_logp"] > row["logp"][key]) - (row["gold_logp"] < row["logp"][key])
                 for key in row["candidate_order"] if key != row["variant"])


def summarize(rows):
    arms = {}
    for arm in ARMS:
        queries = [row for key, row in rows.items() if key[1] == arm]
        episodes = {str(ep): all(row["correct"] for row in queries if row["episode"] == ep)
                    for ep in sorted({row["episode"] for row in queries})}
        arms[arm] = dict(episodes=episodes, successes=sum(episodes.values()), n=len(episodes),
                         queries_correct=sum(row["correct"] for row in queries), queries_total=len(queries),
                         mean_margin=mean(row["margin"] for row in queries),
                         mean_gold_nll=mean(row["gold_nll"] for row in queries))
    return arms


def compare(left, right, summary_left, summary_right):
    changed_order = 0
    cases = []
    for key, a in left.items():
        b = right[key]
        changed = gold_order(a) != gold_order(b)
        changed_order += changed
        cases.append(dict(episode=key[0], arm=key[1], variant=key[2], wording=key[3],
                          left_correct=a["correct"], right_correct=b["correct"],
                          gold_order_changed=changed, margin_delta=b["margin"] - a["margin"],
                          max_candidate_logp_delta=max(abs(b["logp"][candidate] - score)
                                                        for candidate, score in a["logp"].items()),
                          gold_nll_delta=b["gold_nll"] - a["gold_nll"]))
    arms = {}
    for arm in ARMS:
        a, b = summary_left[arm]["episodes"], summary_right[arm]["episodes"]
        rows = [row for row in cases if row["arm"] == arm]
        arms[arm] = dict(episode_gains=[key for key in a if not a[key] and b[key]],
                         episode_losses=[key for key in a if a[key] and not b[key]],
                         query_gains=sum(not q["left_correct"] and q["right_correct"] for q in rows),
                         query_losses=sum(q["left_correct"] and not q["right_correct"] for q in rows),
                         gold_order_changes=sum(q["gold_order_changed"] for q in rows))
    return dict(arms=arms, gold_order_changes=changed_order, cases=cases)


def functional_decision(pairs):
    def differs(a, b):
        return bool(pairs[f"{a}_vs_{b}"]["gold_order_changes"])

    unstable = [arm for arm in ("A", "B") if differs(f"{arm}1-E", f"{arm}2-E")]
    between = differs("A1-E", "B1-E")
    live_changed = differs("P0-A", "P0-E")
    effects = any(p["gold_order_changes"] for p in pairs.values())
    history_status = ("WITHIN_ARM_INSTABILITY" if unstable else
                      "STABLE_BETWEEN_ARM_DIFFERENCE" if between else "ARMS_FUNCTIONALLY_MATCH")
    status = ("FUNCTIONAL_INSTABILITY_PRESENT" if unstable else
              "FUNCTIONAL_EFFECT_PRESENT" if effects else "NO_FUNCTIONAL_EFFECT_OBSERVED")
    return dict(functional_status=status, process_history_status=history_status, unstable_arms=unstable,
                live_vs_restored_status="DIFFERENT" if live_changed else "MATCHED")


def report(out, result, stats, pairs):
    lines = ["# A3.4 Functional Impact", "", f"**{result['functional_status']}**", "",
             result["diagnosis"], "", "## Episode accuracy", "",
             "| Arm | " + " | ".join(LANES) + " |",
             "|---|" + "---:|" * len(LANES)]
    for arm in ARMS:
        lines.append("| " + arm + " | " + " | ".join(
            f"{stats[lane][arm]['successes']}/{stats[lane][arm]['n']}" for lane in LANES) + " |")
    lines += ["", "## Paired outcome changes", "",
              "Direction is left to right. Episode success requires all four strict positive margins.", "",
              "| Comparison | Affected arm | Episode gains | Episode losses | Query gains/losses | Gold order changes |",
              "|---|---|---|---|---:|---:|"]
    changes = 0
    for key, pair in pairs.items():
        for arm, values in pair["arms"].items():
            if not values["gold_order_changes"]:
                continue
            changes += 1
            gains = ", ".join(values["episode_gains"]) or "—"
            losses = ", ".join(values["episode_losses"]) or "—"
            lines.append(f"| {key} | {arm} | {gains} | {losses} | "
                         f"{values['query_gains']}/{values['query_losses']} | {values['gold_order_changes']} |")
    if not changes:
        lines.append("| All | All | 0 | 0 | 0/0 | 0 |")
    lines += ["", f"- Process history — {result['process_history_status']}",
              f"- P0 Live versus restored — {result['live_vs_restored_status']}",
              f"- Historical-bound numerical probe — {result['numerical_probe_status']}",
              "- Scope — frozen 12-episode development benchmark. Historical M1-002A Cell and A3.4 Full-State are distinct paths.",
              "", result["next_action"]]
    (out / "functional_impact_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(out, result):
    probe = {}
    try:
        history.evaluate(out, probe)
    except RuntimeError as exc:
        if "mismatch" in str(exc) or "integrity gate" in str(exc):
            raise InvalidProtocol(str(exc)) from exc
        raise
    lanes = index_cases(out)
    stats = {lane: summarize(rows) for lane, rows in lanes.items()}
    pairs = {f"{a}_vs_{b}": compare(lanes[a], lanes[b], stats[a], stats[b])
             for a, b in combinations(LANES, 2)}
    result.update(functional_decision(pairs))
    result["numerical_probe_status"] = probe["state_equivalence"]
    numeric = load_json(out / "process_history.json")
    write(out / "functional_bridge_statistics.json", dict(lanes=stats, historical_m1_002a=BASELINE))
    write(out / "functional_process_history.json", dict(status=result["process_history_status"],
          unstable_arms=result["unstable_arms"], numeric_probe=numeric,
          comparisons={name: value for name, value in pairs.items() if "P0-A" not in name}))
    write(out / "functional_path_comparison.json", dict(
        live_vs_restored_status=result["live_vs_restored_status"], comparisons=pairs))
    drift_seen = (any(not pair["exact"] for pair in numeric["pairs"].values())
                  or any(case["max_candidate_logp_delta"] > 0
                         for pair in pairs.values() for case in pair["cases"]))
    result["numeric_difference_observed"] = drift_seen
    if not drift_seen:
        result["functional_status"] = "INCONCLUSIVE"
    status = result["functional_status"]
    if status == "INCONCLUSIVE":
        diagnosis = "All outputs matched and this run did not reproduce a numerical difference."
        action = "Keep the functional effect of prior numerical drift open."
    elif status == "FUNCTIONAL_INSTABILITY_PRESENT":
        arms = ", ".join(result["unstable_arms"])
        diagnosis = f"Matched histories changed the gold answer's relative order within arm(s) {arms}."
        action = f"Repeat only the unstable arm(s) {arms} once before assigning a mechanism."
    elif status == "FUNCTIONAL_EFFECT_PRESENT":
        diagnosis = "Gold-answer order differed across at least one matched pair of execution paths."
        action = "Review the identified path and affected query margins before a targeted diagnostic."
    else:
        diagnosis = "Every matched query retained its gold-answer order and strict correctness across all six lanes."
        action = "Pause broad numerical tracing for this benchmark and investigate the observed generalization limits."
    result.update(status="PASS", engineering_status="PASS", diagnosis=diagnosis, next_action=action)
    report(out, result, stats, pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", choices=history.ORDER)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        try:
            worker(out, args.worker)
        except Exception as exc:
            write(out / f"worker-error-{args.worker}.json", dict(
                engineering_status="INVALID" if isinstance(exc, InvalidProtocol) else "FAIL",
                error=f"{type(exc).__name__}: {exc}"))
            raise
        return
    (out / "work").mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="A3.4 Functional Impact", status="FAIL", engineering_status="FAIL",
                  functional_status="INCONCLUSIVE", scientific_status="DEVELOPMENT_ONLY",
                  process_history_status="NOT_RUN", live_vs_restored_status="NOT_RUN",
                  capacity_status="NOT_RUN_BY_SCOPE", dataset_sha256=DATASET_SHA256,
                  canonical_dataset_sha256=CANONICAL_SHA256)
    try:
        fixture(out)
        frozen_bounds(out)
        for label in history.ORDER:
            child = subprocess.run([sys.executable, __file__, str(out), "--worker", label])
            if child.returncode:
                path = out / f"worker-error-{label}.json"
                error = load_json(path) if path.exists() else {}
                exception = InvalidProtocol if error.get("engineering_status") == "INVALID" else RuntimeError
                raise exception(f"{label}: {error.get('error', child.returncode)}")
        evaluate(out, result)
    except Exception as exc:
        if isinstance(exc, InvalidProtocol):
            result.update(status="INVALID", engineering_status="INVALID")
        result.update(functional_status="INCONCLUSIVE", diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)
        decision = load_json(out / "decision.json")
        write(out / "decision.json", dict(decision, **{
            key: result[key] for key in ("experiment", "functional_status", "scientific_status",
                                         "dataset_sha256", "canonical_dataset_sha256", "process_history_status",
                                         "live_vs_restored_status", "capacity_status")}))
        if not (out / "functional_impact_report.md").exists():
            (out / "functional_impact_report.md").write_text(
                f"# A3.4 Functional Impact\n\n**INCONCLUSIVE**\n\n{result['diagnosis']}\n", encoding="utf-8")
        write(out / "manifest.json", {"artifacts_sha256": {
            path.name: digest_file(path) for path in out.iterdir()
            if path.is_file() and path.name != "manifest.json" and path.suffix != ".tmp"}})


if __name__ == "__main__":
    main()
