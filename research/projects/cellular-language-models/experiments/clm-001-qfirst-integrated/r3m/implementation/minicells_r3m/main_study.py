"""Prospective Main execution. Pilot review and all scientific freezes are mandatory inputs."""

import copy
import itertools
from pathlib import Path
import numpy as np
from minicells_r3e.common import read, atomic, fingerprint, digest
from minicells_r3h.routing import Model
from minicells_r3h.trainer import Trainer
from minicells_r3j.storage import Store, save
from minicells_r3l.storage import recover_latest
from minicells_r3m import anchor, data
from minicells_r3m.engine import run_arm
from minicells_r3m.training import restore
from minicells_r3m.evaluation import repeated, attribution, specialization
from minicells_r3m.protocol import main_gate

FILES = (
    "MAIN_FROZEN_PROTOCOL.json",
    "PRESSURE_TRIGGER_EVENTS.json",
    "CONNECTION_TRANSACTIONS.json",
    "ENVIRONMENT_INTERACTION.json",
    "LONG_HORIZON_RETENTION.json",
    "CAPABILITY_SPECIALIZATION.json",
    "CELL_CONNECTION_ATTRIBUTION.json",
    "COMPUTE_COST.json",
    "STATISTICAL_ANALYSIS.json",
    "SCIENTIFIC-FINDINGS.md",
    "RUN_MANIFEST.json",
)


def configure(c, pilot_manifest):
    main_gate(c, read(pilot_manifest)["decision"], digest(pilot_manifest))
    result = copy.deepcopy(c)
    result["phase"] = "MAIN"
    return result


def phase_i(c, backend, output, seed, device, historical, ident):
    runtime = anchor.runtime(c)
    root = Path(output) / "private-checkpoints/phase-I" / str(seed)
    aid = dict(ident, seed=seed, kind="COMMON_PHASE_I")
    previous = recover_latest(root, aid)
    if previous is None:
        spec = c["main_initialization"][str(seed)]
        if spec["step"] != 0 or spec["parent_seed"] not in c["pilot_seeds"]:
            raise RuntimeError("MAIN_INITIALIZATION_MUST_NAME_VERIFIED_R3H_STEP0")
        initial, evidence = anchor.load(
            c, backend, Path(output) / "parent-r3h", spec["parent_seed"], device, step=0
        )
        m = Model(runtime, seed, "B-H").to(device)
        m.load_state_dict(initial.model.state_dict(), strict=True)
        m.temperature = initial.temperature
        tr = Trainer(m, runtime, "B-H")
        tr.parent_identity = dict(main_initialization=spec, receipt=evidence)
        atomic(
            root / "INITIALIZATION.json",
            dict(
                specification=spec,
                source=evidence,
                new_training_seed=seed,
                initialization="EXPLICIT_FROZEN_PRETRAINED_START_NOT_AN_UNDECLARED_NEW_BACKBONE",
                model_sha256=fingerprint(m.state_dict()),
            ),
        )
    else:
        tr = restore(runtime, previous, seed, device)
    while tr.step < c["anchor_step"]:
        tr.update(historical)
        if tr.step % 50 == 0:
            save(root, aid, tr.state(), tr.step)
            print("[R3M_SNAPSHOT]", seed, "COMMON_PHASE_I", tr.step, flush=True)
    if tr.data_cursor != c["anchor_step"] * c["tokens_per_step"]:
        raise RuntimeError("MAIN_PHASE_I_BUDGET_CHANGED")
    save(root, aid, tr.state(), tr.step)
    return tr


def random_schedule(seed):
    rng = np.random.default_rng(seed + 153301)
    return [int(rng.choice([80, 100, 120, 140, 160])), int(rng.choice([260, 280, 300, 320, 340]))]


def train_job(c, backend, output, folder, job, device, ident):
    seed, env, arm = job["seed"], job["environment"], job["arm"]
    runtime = anchor.runtime(c)
    historical = data.historical_assets(runtime, backend, Path(folder) / "historical", ("train",))
    research, _ = data.load(folder, c)
    base = phase_i(c, backend, output, seed, device, historical, ident)
    root = Path(output) / "private-checkpoints/main" / ("%s-%s-%s" % (seed, env, arm))
    aid = dict(ident, seed=seed, environment=env, arm=arm, random_schedule=random_schedule(seed))
    run_arm(c, runtime, base, aid, root, historical, research, env, arm, device)


def job_root(output, seed, env, arm):
    return Path(output) / "private-checkpoints/main" / ("%s-%s-%s" % (seed, env, arm))


def evaluate_job(c, output, research, job, device, tokenizer, ident):
    seed, env, arm = job["seed"], job["environment"], job["arm"]
    root = job_root(output, seed, env, arm)
    if (root / "EVALUATION.json").exists():
        return
    runtime = anchor.runtime(c)
    values = {}
    grids = {}
    spec = {}

    def at(which, t):
        aid = dict(
            ident, seed=seed, environment=env, arm=which, random_schedule=random_schedule(seed)
        )
        store = Store(job_root(output, seed, env, which), aid)
        return store.verified(read(store.receipt_file(t)))

    for t in c["barriers"]:
        payload = at(arm, t)
        tr = restore(runtime, payload["trainer"], seed, device)
        values[str(t)] = repeated(tr.model, research, "main_report", tokenizer)
        if t in (400, 600):
            old = restore(runtime, at("S0", t)["trainer"], seed, device)
            grids[str(t)] = attribution(old.model, tr.model, research, "main_report", tokenizer)
            spec[str(t)] = specialization(tr.model, research, "main_report", tokenizer)
    aux = at(arm, 600)["aux"]
    atomic(
        root / "EVALUATION.json",
        dict(
            job=job,
            evaluations=values,
            attribution=grids,
            specialization=spec,
            pressure=aux["pressure_events"],
            transactions=aux["transactions"],
            curve=aux["curve"],
            cost={
                k: aux[k]
                for k in (
                    "main_updates",
                    "shadow_updates",
                    "discarded_shadow_updates",
                    "proposal_optimizer_updates",
                )
            },
        ),
    )


def statistics(seed_comparisons):
    result = []
    for name, values in seed_comparisons.items():
        values = np.asarray(values, dtype=float)
        if len(values) != 5:
            result.append(dict(comparison=name, status="INCOMPLETE"))
            continue
        observed = abs(float(values.mean()))
        perm = [
            abs(float((values * np.asarray(sign)).mean()))
            for sign in itertools.product((-1, 1), repeat=5)
        ]
        result.append(
            dict(
                comparison=name,
                seed_values=values.tolist(),
                mean=float(values.mean()),
                p_value=sum(v >= observed - 1e-15 for v in perm) / len(perm),
                unit="INDEPENDENT_SEED_NOT_STORY",
            )
        )
    tested = sorted([r for r in result if "p_value" in r], key=lambda r: r["p_value"])
    previous = 0.0
    for i, row in enumerate(tested):
        previous = max(previous, min(1.0, row["p_value"] * (len(tested) - i)))
        row["holm_p_value"] = previous
    return result


def aggregate(c, output, ident, reference, failures):
    records = []
    for s in c["main_seeds"]:
        for e in c["environments"]:
            for a in c["main_arms"]:
                path = job_root(output, s, e, a) / "EVALUATION.json"
                if path.exists():
                    records.append(read(path))
    complete = len(records) == 50 and not failures
    index = {(r["job"]["seed"], r["job"]["environment"], r["job"]["arm"]): r for r in records}
    comparisons = {
        k: [] for k in ("S2_S0_C", "S2_S1_C", "S2_S4_C", "S0_S3_N", "ENVIRONMENT_INTERACTION")
    }
    interactions = []
    if complete:
        for seed in c["main_seeds"]:

            def utility(env, arm):
                values = index[(seed, env, arm)]["evaluations"]
                initial = values["0"]
                last = values["600"]
                if not initial["story_nll"] > 0 or not initial["answer_nll"] > 0:
                    raise RuntimeError("UTILITY_ANCHOR_DENOMINATOR_INVALID")
                return -0.5 * (
                    last["story_nll"] / initial["story_nll"]
                    + last["answer_nll"] / initial["answer_nll"]
                )

            for key, other in [("S2_S0_C", "S0"), ("S2_S1_C", "S1"), ("S2_S4_C", "S4")]:
                comparisons[key].append(utility("C", "S2") - utility("C", other))
            comparisons["S0_S3_N"].append(utility("N", "S0") - utility("N", "S3"))
            interaction = (utility("C", "S2") - utility("C", "S0")) - (
                utility("N", "S2") - utility("N", "S0")
            )
            comparisons["ENVIRONMENT_INTERACTION"].append(interaction)
            interactions.append(dict(seed=seed, value=interaction))
    dest = Path(output) / "reports"
    dest.mkdir(parents=True, exist_ok=True)
    documents = {
        "MAIN_FROZEN_PROTOCOL.json": c,
        "PRESSURE_TRIGGER_EVENTS.json": [dict(job=r["job"], events=r["pressure"]) for r in records],
        "CONNECTION_TRANSACTIONS.json": [
            dict(job=r["job"], transactions=r["transactions"]) for r in records
        ],
        "ENVIRONMENT_INTERACTION.json": interactions,
        "LONG_HORIZON_RETENTION.json": [
            dict(job=r["job"], quality=r["evaluations"]) for r in records
        ],
        "CAPABILITY_SPECIALIZATION.json": [
            dict(job=r["job"], interventions=r["specialization"]) for r in records
        ],
        "CELL_CONNECTION_ATTRIBUTION.json": [
            dict(job=r["job"], four_grid=r["attribution"]) for r in records
        ],
        "COMPUTE_COST.json": dict(
            shards=[dict(job=r["job"], cost=r["cost"]) for r in records],
            strict_equal_total_compute=False,
        ),
        "STATISTICAL_ANALYSIS.json": dict(
            seed_comparisons=statistics(comparisons),
            story_bootstrap="DESCRIPTIVE_ONLY_NOT_SEED_REPLICATION",
            confirmation="SEALED",
        ),
    }
    for name, value in documents.items():
        atomic(dest / name, value)
    (dest / "SCIENTIFIC-FINDINGS.md").write_text(
        "# R3m-M\n\n"
        + (
            "Complete exploratory matrix; independent mechanism review required."
            if complete
            else "INCONCLUSIVE: incomplete matrix."
        )
        + "\nNo independent confirmation.\n"
    )
    manifest = dict(
        identity=ident,
        status="MAIN_COMPLETE_EXPLORATORY" if complete else "PARTIAL",
        completed_shards=len(records),
        expected_shards=50,
        hf_reference=reference,
        failures=failures,
        confirmation="SEALED",
        scientific_decision=False,
        artifacts=[
            dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
            for p in dest.iterdir()
            if p.name in FILES and p.name != "RUN_MANIFEST.json"
        ],
    )
    atomic(dest / "RUN_MANIFEST.json", manifest)
    return manifest


def worker(c, output, folder, index, count, phase):
    from minicells_artifacts.hf_storage import HubBackend
    from tokenizers import Tokenizer
    from .runner import identity

    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    ident = identity(c)
    jobs = read(Path(output) / "SHARD_PLAN.json")["jobs"]
    if phase == "main-initialize":
        if index == 0:
            historical = data.historical_assets(
                anchor.runtime(c), backend, Path(folder) / "historical", ("train",)
            )
            for seed in sorted({j["seed"] for j in jobs}):
                phase_i(c, backend, output, seed, "cuda:0", historical, ident)
        return
    research, _ = data.load(folder, c)
    tokenizer = Tokenizer.from_file(str(Path(folder) / "historical/tokenizer.json"))
    for job in jobs[index::count]:
        if phase == "main-train":
            train_job(c, backend, output, folder, job, "cuda:%s" % index, ident)
        elif phase == "main-evaluate":
            evaluate_job(c, output, research, job, "cuda:%s" % index, tokenizer, ident)


def automatic(c, path, output, folder):
    import uuid
    import inspect
    import torch
    from minicells_artifacts.hf_storage import HubBackend
    from minicells_r3e.runner import secrets
    from minicells_r3h import runner as coordinator
    from .runner import identity
    from .storage import Publisher, restore as restore_private
    from minicells_r3e.storage import snapshot_lock

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ident = identity(c)
    session = uuid.uuid4().hex
    secrets()
    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    backend.check_write()
    revision = backend.info().sha
    pointer = backend.optional("r3m/" + ident["run_id"] + "/latest.json", revision)
    prior = dict(pointer, hf_repo=backend.repo, hf_revision=revision) if pointer else None
    private = output / "private-checkpoints"
    if prior:
        restore_private(backend, prior, private, output, ident, session, c["barriers"])
    publisher = Publisher(backend, private, output, ident, session, prior)

    def publish(force):
        with snapshot_lock(private):
            publisher.publish(force)

    failures = []
    try:
        if torch.cuda.device_count() != 2:
            raise RuntimeError("TWO_GPUS_REQUIRED")
        proof = c["main_release"].get("cross_session_proof")
        if not proof or proof["status"] != "CROSS_SESSION_RESTORE_PASS":
            raise RuntimeError("CROSS_SESSION_RESTORE_INVALID")
        if not (Path(folder) / "manifest.json").exists():
            data.prepare(c, anchor.runtime(c), backend, folder, main=True)
        data.load(folder, c)
        jobs = [
            dict(seed=s, environment=e, arm=a)
            for s in c["main_seeds"]
            for e in c["environments"]
            for a in c["main_arms"]
        ]
        pending = [
            j
            for j in jobs
            if not (
                job_root(output, j["seed"], j["environment"], j["arm"]) / "COMPLETE.json"
            ).exists()
        ]
        evaluation_pending = [
            j
            for j in jobs
            if not (
                job_root(output, j["seed"], j["environment"], j["arm"]) / "EVALUATION.json"
            ).exists()
        ]
        selected = pending[:2] if pending else evaluation_pending[:2]
        atomic(
            output / "SHARD_PLAN.json",
            dict(jobs=selected, identity=ident, remaining_shards=max(0, len(pending) - 2)),
        )
        namespace = dict(coordinator.workers.__globals__, BASE=Path(__file__).resolve().parents[2])
        exec(
            inspect.getsource(coordinator.workers).replace("[R3H_SNAPSHOT]", "[R3M_SNAPSHOT]"),
            namespace,
        )
        pool = namespace["workers"]
        if pending:
            for phase in ("main-initialize", "main-train"):
                pool(anchor.runtime(c), path, output, folder, phase, publish)
        if not pending:
            # Main Report is evaluated only after all fifty fixed-budget training branches complete.
            pool(anchor.runtime(c), path, output, folder, "main-evaluate", publish)
        publish(True)
    except Exception as error:
        failures.append(dict(type=type(error).__name__, code=str(error)[:120]))
    report = aggregate(c, output, ident, publisher.prior, failures)
    from .publication import publish as github_publish

    try:
        report["publication"] = github_publish(output)
    except Exception as error:
        report["publication"] = dict(status="LOCAL_RETAINED", type=type(error).__name__)
    atomic(output / "RUN_STATUS.json", report)
    print(report, flush=True)
    if failures:
        raise RuntimeError("MAIN_PARTIAL_SEE_RETAINED_STATUS")
