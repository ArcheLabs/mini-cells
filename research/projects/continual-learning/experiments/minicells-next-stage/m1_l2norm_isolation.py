"""Recover the A3 Bundle, isolate Q/K L2 normalization and cross-replay its inputs."""
import argparse
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen

import torch
from safetensors.torch import load_file, save_file

from a31_norm import Boundary, FIELDS, config, describe, describe_norm, instrument, materialize, metrics, restore, tensor_bytes
from common import canonical_hash, publishable, write
from m1_continuation_isolation import BOUNDS_FILE, compare_pair, fingerprint, frozen_bounds, imported, start_differences
from m1_full_state_bridge import baseline, check_environment, dataset, load_json, probe_inputs
from m1_reference_capacity import digest_file
from minicells_native import MiniCells
from minicells_native.full_state import FullState
from r1_equivalence import ensure_readonly, live, trace

FORMAT = "minicells.m1.full-state.l2norm-isolation.v1"
A3_COMMIT = "818c9625f46f9450c49500cfab54a4c9dbc71101"
A3_ARTIFACT = (f"https://raw.githubusercontent.com/ArcheLabs/mini-cells/{A3_COMMIT}/"
               "artifacts/experiments/minicells-m1-002a-r1-a3-live-vs-restored/1349bcc11ee1/")
BUNDLE_SHA256 = "b07fcb2f0fe5efe9836608fa1ff54faae2bd81eb0402a35b75ae1d0c2c345804"
PROCESSES = ("P0", "P2", "P3")
REPEATS = 10
CAPTURE_REPEATS = 3
KAGGLE_DOWNLOAD_VERSION = "2.2.5"


class BundleUnavailable(RuntimeError):
    pass


def select_bundle(paths, expected=BUNDLE_SHA256):
    return next((path for path in paths if path.is_file() and digest_file(path) == expected), None)


def kaggle_download_token():
    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("KAGGLE_API_TOKEN").strip()
    except Exception:
        raise BundleUnavailable("Kaggle download requires the attached KAGGLE_API_TOKEN Secret.") from None
    if not token:
        raise BundleUnavailable("KAGGLE_API_TOKEN Secret is empty.")
    return token


def download_historical_output(work, version):
    token = kaggle_download_token()
    target = work / "historical-output"
    target.mkdir()
    package = work / "kaggle-download-cli"
    # Keep credentials out of the installer, command line and published artifacts.
    env = {key: val for key, val in os.environ.items()
           if key not in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "GITHUB_TOKEN", "HF_TOKEN")}
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet",
                        "--target", str(package), f"kaggle=={KAGGLE_DOWNLOAD_VERSION}", "six==1.17.0"], env=env,
                       check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BundleUnavailable(f"Could not prepare version-aware Kaggle download CLI: {type(exc).__name__}") from None
    env.update(KAGGLE_API_TOKEN=token, PYTHONPATH=str(package))
    command = [sys.executable, "-c", "from kaggle.cli import main; main()", "kernels", "output", version,
               "--file-pattern", r"(?:.*/)?prefix[.]cell$", "-q", "-p", str(target)]
    try:
        subprocess.run(command, env=env, check=True, timeout=240, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # Report a classified HTTP status only. Server responses may include signed URLs.
        error = re.search(r"\b(401|403|404|429)\b", str(exc.stderr))
        status = f"HTTP {error.group(1)}" if error else f"exit {exc.returncode}"
        raise BundleUnavailable(f"Kaggle historical output download failed ({status}).") from None
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BundleUnavailable(f"Kaggle historical output download failed: {type(exc).__name__}.") from None
    return target


def recover_bundle(out, local_path, version, *, mounted_root=Path("/kaggle/input")):
    if local_path:
        candidates = [Path(local_path)]
        source = "explicit_path"
    else:
        candidates = sorted(mounted_root.rglob("prefix.cell"))
        source = "mounted_saved_output"
    bundle = select_bundle(candidates)
    if bundle is None and not local_path and version:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[1-9][0-9]*", version):
            raise BundleUnavailable("Use a fixed Kaggle owner/notebook/version number")
        print("A3_BUNDLE_RECOVERY versioned Kaggle Saved Output", flush=True)
        target = download_historical_output(out / "work", version)
        candidates = sorted(target.rglob("prefix.cell"))
        bundle = select_bundle(candidates)
        source = "versioned_cli_output"
    if bundle is None:
        if local_path and not candidates[0].is_file():
            raise BundleUnavailable("A3_BUNDLE_PATH does not name a readable file.")
        if candidates:
            raise BundleUnavailable(
                f"A3 Bundle SHA-256 mismatch in {len(candidates)} candidate file(s). Use A3 run 1349bcc11ee1.")
        if source == "versioned_cli_output":
            raise BundleUnavailable("The requested Kaggle Saved Output does not contain prefix.cell.")
        raise BundleUnavailable(
            "No mounted A3 prefix.cell found. Set A3_KAGGLE_REF=owner/notebook/version to download with the Secret.")
    write(out / "bundle_provenance.json",
          dict(a3_commit=A3_COMMIT, sha256=BUNDLE_SHA256, recovery_method=source,
               kaggle_version=version if source == "versioned_cli_output" else None,
               kaggle_cli_version=KAGGLE_DOWNLOAD_VERSION if source == "versioned_cli_output" else None))
    write(out / "work" / "archive.json", dict(path=str(bundle.resolve())))


def get_a3(out):
    def read(name):
        with urlopen(A3_ARTIFACT + name, timeout=30) as response:
            return response.read()

    expected = json.loads(read("manifest.json"))["artifacts_sha256"]
    data = read("initial_state_comparison.json")
    if sha256(data).hexdigest() != expected["initial_state_comparison.json"]:
        raise ValueError("A3 receipt checksum mismatch")
    write(out / "work" / "a3_reference.json", json.loads(data)["trials"]["P0"])


def validated_start(mc, archive, expected):
    if digest_file(archive) != BUNDLE_SHA256:
        raise ValueError("Original Bundle bytes changed")
    cell, cache = imported(mc, archive)
    if digest_file(mc.cells._store.path(cell.id)) != BUNDLE_SHA256:
        raise ValueError("Imported Bundle bytes changed")
    state = FullState.capture(mc._backbone, cache)
    mismatches = start_differences(expected, fingerprint(state.structure, state.tensors))
    if mismatches:
        raise RuntimeError(f"A3 initial State differs: {mismatches[:5]}")
    return cell, cache


def capture_trials(mc, out, role, fn, module):
    backbone = mc._backbone
    work = out / "work"
    archive = Path(load_json(work / "archive.json")["path"])
    expected = load_json(work / "a3_reference.json")
    _, writes, query = probe_inputs(dataset(out), backbone)
    primary = query["tokens"] + next(iter(query["candidates"].values()))[:-1]
    if (writes[0] != expected["inputs"]["write_token_ids"]
            or list(query["candidates"]) != expected["inputs"]["candidate_order"]
            or query["candidates"] != expected["inputs"]["candidates"]
            or [primary] != expected["inputs"]["primary_chunks"]):
        raise ValueError("A3 query IDs or chunk boundaries differ")
    warm = live(backbone, writes[:1])
    backbone.step(primary, [warm], [1.0])
    del warm
    all_tensors, trials, validation = {}, {}, {}
    bounds = load_json(out / BOUNDS_FILE)
    for i in range(1, CAPTURE_REPEATS + 1):
        runs = []
        observer = Boundary(fn, module)
        for enabled in (False, True):
            cell, cache = validated_start(mc, archive, expected["state"])
            pristine = deepcopy(cache)

            def execute():
                return trace(backbone, lambda: deepcopy(pristine), query, primary_cache=cache)

            if enabled:
                with instrument(backbone, observer):
                    runs.append(ensure_readonly(mc, cell, execute))
            else:
                runs.append(ensure_readonly(mc, cell, execute))
            mc.cells.delete(cell.id)
        tensors, metadata = observer.export()
        all_tensors.update({f"{i}.{name}": value for name, value in tensors.items()})
        trials[str(i)] = dict(tensors=metadata, calls=observer.calls)
        summary, errors = compare_pair(*runs, bounds)
        validation[str(i)] = dict(summary=summary, fields=errors)
        print(f"A31_CAPTURE {role} {i}/{CAPTURE_REPEATS}", flush=True)
    save_file(all_tensors, out / f"boundary-{role}.safetensors")
    write(out / f"boundary-{role}.json", trials)
    write(out / f"instrumentation-{role}.json", validation)


def run_matrix(out, executor, fn, module):
    metadata = {role: load_json(out / f"boundary-{role}.json")["1"] for role in PROCESSES}
    inputs = {role: load_file(out / f"boundary-{role}.safetensors") for role in PROCESSES}
    output, records, profiles = {}, [], {}
    for origin in PROCESSES:
        for axis in "QK":
            name = f"{axis}_PRE_NORM"
            receipt = metadata[origin]["tensors"][name]
            source = inputs[origin][f"1.{name}"]
            params = dict(metadata[origin]["calls"][axis]["arguments"])
            if params.get("output_dtype"):
                params["output_dtype"] = getattr(torch, params["output_dtype"].removeprefix("torch."))
            exemplar = restore(source, receipt, "cuda:0")
            scratch = exemplar.clone()
            fn(scratch, **params)
            del scratch, exemplar
            reference = inputs[origin][f"1.{axis}_POST_NORM"]
            for repetition in range(1, REPEATS + 1):
                view = restore(source, receipt, "cuda:0")
                before = describe(view)
                y = fn(view, **params)
                actual_kernel = config(module, view)
                if actual_kernel["autocast"] != metadata[origin]["calls"][axis]["kernel"]["autocast"]:
                    raise RuntimeError("Replay autocast differs from the original call")
                if sha256(tensor_bytes(view)).hexdigest() != receipt["sha256"]:
                    raise RuntimeError("Normalization mutated its replay input")
                value, conf = materialize(y, describe(y))
                key = f"{origin}.{axis}.{repetition}"
                output[key] = value
                records.append(dict(executor=executor, origin=origin, axis=axis, repetition=repetition,
                                    pid=os.getpid(), input_sha256=receipt["sha256"], input_metadata=before,
                                    output_sha256=conf["sha256"], output_metadata=conf,
                                    kernel=actual_kernel, vs_native=metrics(reference, value)))
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    fn(view, **params)
                    torch.cuda.synchronize()
                profiles[f"{origin}.{axis}"] = sorted({e.name for e in profiler.events()
                                                      if str(e.device_type).endswith("CUDA")})
            except (OSError, RuntimeError) as exc:
                profiles[f"{origin}.{axis}"] = {"error": str(exc)}
    save_file(output, out / "work" / f"replay-{executor}.safetensors")
    write(out / f"replay-{executor}.json", dict(records=records, separate_profile=profiles))


@torch.inference_mode()
def worker(out, role):
    torch.set_num_threads(4)
    with MiniCells.open_preview(out / "work" / f"registry-{role}") as mc:
        env = check_environment(mc)
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not uuid.startswith("GPU-") or "," in uuid:
            raise RuntimeError("An exact physical GPU UUID is required")
        from fla.modules import l2norm
        fn = mc._backbone._recurrent_execution.ops.l2_norm
        if fn is not l2norm.l2_norm:
            raise RuntimeError("MoM does not resolve to the pinned FLA l2_norm")
        write(out / f"environment-{role}.json",
              dict(env=env, gpu_uuid=uuid, pid=os.getpid(), implementation=describe_norm(fn, l2norm)))
        capture_trials(mc, out, role, fn, l2norm)
    torch.cuda.empty_cache()
    work = out / "work"
    (work / f"{role}.ready").touch()
    while not (work / f"{role}.replay").exists():
        if (work / "stop").exists():
            return
        time.sleep(.2)
    run_matrix(out, role, fn, l2norm)
    (work / f"{role}.done").touch()


def wait_ready(proc, flag, out, role):
    while not flag.exists():
        if proc.poll() is not None:
            file = out / f"worker_error-{role}.json"
            raise RuntimeError(f"{role}: {load_json(file) if file.exists() else proc.returncode}")
        time.sleep(.2)


def match_boundaries(out):
    tensors = {role: load_file(out / f"boundary-{role}.safetensors") for role in PROCESSES}
    receipts = {role: load_json(out / f"boundary-{role}.json") for role in PROCESSES}
    keys = [(role, str(i)) for role in PROCESSES for i in range(1, CAPTURE_REPEATS + 1)]
    rows = {}
    for (a, i), (b, j) in combinations(keys, 2):
        comparison = {}
        for axis in "QK":
            left_call, right_call = receipts[a][i]["calls"][axis], receipts[b][j]["calls"][axis]
            slot = left_call["first_recurrent_routed_slot"]
            comparison[axis] = dict(arguments_equal=left_call["arguments"] == right_call["arguments"],
                                    same_first_routed_slot=slot is not None and
                                    slot == right_call["first_recurrent_routed_slot"],
                                    kernel_config_equal=left_call["kernel"] == right_call["kernel"])
        for name in FIELDS:
            x, y = tensors[a][f"{i}.{name}"], tensors[b][f"{j}.{name}"]
            left = receipts[a][i]["tensors"][name]
            right = receipts[b][j]["tensors"][name]
            if (x.shape != y.shape or x.dtype != y.dtype):
                comparison[name] = {"bitwise_equal": False, "shape_or_dtype_mismatch": True}
            else:
                comparison[name] = dict(metrics(x, y), layout_equal=all(left[k] == right[k] for k in
                                        ("shape", "dtype", "stride", "storage_offset", "memory_format")))
        rows[f"{a}-{i}_vs_{b}-{j}"] = comparison
    write(out / "boundary_capture.json", rows)
    write(out / "input_hashes.json", {role: {axis: receipts[role]["1"]["tensors"][f"{axis}_PRE_NORM"]["sha256"]
                                              for axis in "QK"} for role in PROCESSES})
    return rows


def summarize(out, boundary):
    replayed = {role: load_file(out / "work" / f"replay-{role}.safetensors") for role in PROCESSES}
    records = {role: load_json(out / f"replay-{role}.json")["records"] for role in PROCESSES}
    matrix = {}
    with (out / "replay_results.jsonl").open("w", encoding="utf-8") as stream:
        for origin in PROCESSES:
            for axis in "QK":
                group = [row for rows in records.values() for row in rows
                         if (row["origin"], row["axis"]) == (origin, axis)]
                if len(group) != len(PROCESSES) * REPEATS or len({r["input_sha256"] for r in group}) != 1:
                    raise RuntimeError("Replay matrix is incomplete or selected inputs differ")
                key = f"{origin}.{axis}"
                reference = replayed[PROCESSES[0]][key + ".1"]
                group_results = []
                for row in group:
                    output = replayed[row["executor"]][f"{key}.{row['repetition']}"]
                    report = dict(row, vs_p0_first=metrics(reference, output))
                    group_results.append(report)
                    stream.write(json.dumps(report, allow_nan=False) + "\n")
                matrix[key] = dict(repetitions=len(group), cross_process_bitwise_equal=all(
                    row["vs_p0_first"]["bitwise_equal"] for row in group_results),
                    output_hashes_by_process={role: sorted({r["output_sha256"] for r in group if r["executor"] == role})
                                              for role in PROCESSES})
    write(out / "replay_matrix.json", matrix)
    evidence = []
    for pair, shapes in boundary.items():
        for axis in "QK":
            pre, post, recurrent = (shapes[f"{axis}_{stage}"] for stage in ("PRE_NORM", "POST_NORM", "RECURRENT"))
            if not pre["bitwise_equal"]:
                tag = "UPSTREAM_DIVERGENCE_FOUND"
            elif not pre.get("layout_equal", False) or not shapes[axis]["arguments_equal"]:
                tag = "INCONCLUSIVE"
            elif not post["bitwise_equal"]:
                # Require both replay inputs corresponding to the observed pair to diverge.
                roles = [side.split("-")[0] for side in pair.split("_vs_")]
                reproduced = pair.count("-1") == 2 and all(
                    not matrix[f"{role}.{axis}"]["cross_process_bitwise_equal"] for role in roles)
                tag = "ROOT_CAUSE_LOCALIZED_TO_L2_NORM" if reproduced else "INCONCLUSIVE"
            elif not recurrent["bitwise_equal"]:
                tag = ("POST_NORM_DIVERGENCE_FOUND" if shapes[axis]["same_first_routed_slot"]
                       else "INCONCLUSIVE")
            else:
                continue
            evidence.append(dict(pair=pair, axis=axis, status=tag, pre=pre, post=post, recurrent=recurrent))
    statuses = {v["status"] for v in evidence} - {"INCONCLUSIVE"}
    status = next(iter(statuses)) if len(statuses) == 1 else "INCONCLUSIVE"
    write(out / "first_divergence.json", dict(status=status, evidence=evidence))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", choices=PROCESSES)
    parser.add_argument("--bundle", default=os.getenv("A3_BUNDLE_PATH"))
    parser.add_argument("--kaggle-version", default=os.getenv("A3_KAGGLE_REF"))
    args = parser.parse_args()
    out = args.output
    if args.worker:
        try:
            worker(out, args.worker)
        except Exception as exc:
            write(out / f"worker_error-{args.worker}.json", dict(type=type(exc).__name__, message=str(exc)))
            raise
        return
    work = out / "work"
    work.mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="M1-002A-R1-A3.1 Q/K L2 Norm Isolation", status="INCONCLUSIVE",
                  engineering_status="FAIL", state_equivalence="NOT_RUN_BY_SCOPE",
                  bridge_status="NOT_RUN_BY_SCOPE", capacity_status="NOT_RUN_BY_SCOPE")
    workers = {}
    try:
        recover_bundle(out, args.bundle, args.kaggle_version)
        get_a3(out)
        baseline(out)
        frozen_bounds(out)
        for role in PROCESSES:
            workers[role] = subprocess.Popen([sys.executable, __file__, str(out), "--worker", role])
            wait_ready(workers[role], work / f"{role}.ready", out, role)
        env = {role: load_json(out / f"environment-{role}.json") for role in PROCESSES}
        if (len({r["gpu_uuid"] for r in env.values()}) != 1 or len({r["pid"] for r in env.values()}) != 3
                or len({canonical_hash(r["env"]) for r in env.values()}) != 1):
            raise RuntimeError("Workers did not have an identical runtime and physical GPU with distinct PIDs")
        write(out / "environment.json", env)
        write(out / "l2norm_source.json", {role: env[role]["implementation"] for role in PROCESSES})
        validation = {role: load_json(out / f"instrumentation-{role}.json") for role in PROCESSES}
        write(out / "instrumentation_validation.json", validation)
        valid = all(trial["summary"]["passed"] for r in validation.values() for trial in r.values())
        if not valid:
            result.update(status="INVALID_INSTRUMENTATION", engineering_status="PASS",
                          diagnosis="Observed and unobserved runs exceeded the frozen R1 envelope.")
            return
        boundary = match_boundaries(out)
        for role in PROCESSES:
            (work / f"{role}.replay").touch()
            wait_ready(workers[role], work / f"{role}.done", out, role)
        status = summarize(out, boundary)
        result.update(status=status, engineering_status="PASS", diagnosis=status)
    except BundleUnavailable as exc:
        result.update(status="BLOCKED", diagnosis=str(exc))
    except Exception as exc:
        result.update(diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (work / "stop").touch()
        for proc in workers.values():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait()
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
