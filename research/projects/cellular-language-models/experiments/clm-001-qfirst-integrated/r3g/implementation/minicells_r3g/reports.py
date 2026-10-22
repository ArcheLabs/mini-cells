"""Exploratory attribution, explicit partial evidence, no confirmation claims."""

from pathlib import Path
from datetime import datetime, timezone
import sys
from .common import atomic, digest

MAPPING = {
    "INPUT_VERIFICATION.json": "input",
    "BRANCH_RESTORE.json": "branches",
    "CROSSOVER_MATRIX.json": "crossover",
    "CELL_PARAMETER_DIFF.json": "parameters",
    "ROUTING_DISTRIBUTION.json": "routing",
    "TRAINING_TRAJECTORY_AUDIT.json": "trajectory",
    "ATTRIBUTION_SUMMARY.json": "attribution",
    "COMPUTE_COST.json": "cost",
}


def write(output, identity, arms, env, publication, started, inputs):
    output = Path(output)
    root = output / "reports"
    root.mkdir(parents=True, exist_ok=True)
    valid = all(r.get("valid", False) for r in arms.values()) and len(arms) == 2
    status = "READONLY_ATTRIBUTION_AUDIT_COMPLETE" if valid else "INVALID_OR_INCOMPLETE_AUDIT"
    for name, key in MAPPING.items():
        atomic(
            root / name,
            {
                a: r.get(key, {"status": "NOT_AVAILABLE", "failure": r.get("failure")})
                for a, r in arms.items()
            },
        )
    atomic(root / "ENVIRONMENT.json", env)
    summary = {
        "identity": identity,
        "status": status,
        "optimizer_updates": 0,
        "scientific_decision": False,
        "development_accessed": False,
        "confirmation": "SEALED",
        "arms": {
            a: {
                "valid": r.get("valid", False),
                "failure": r.get("failure"),
                "attribution": r.get(
                    "attribution", {"attribution_status": "INVALID_OR_INCOMPLETE"}
                ),
            }
            for a, r in arms.items()
        },
    }
    atomic(root / "METRICS.json", summary)
    lines = [
        "# R3g read-only Cell–Connection attribution",
        "",
        status,
        "",
        "One training seed. Calibration was used by the parent to select an organization; it is not independent validation.",
        "No optimizer updates, new candidates, new adaptation or Monte Carlo training replay. Development unread; Confirmation sealed.",
        "",
    ]
    for arm, row in arms.items():
        lines += ["## " + arm, ""]
        if not row.get("valid", False):
            lines += ["Incomplete evidence: " + str(row.get("failure")), ""]
            continue
        cross = row["crossover"]
        at = row["attribution"]
        lines += ["| Quantity | Value |", "|---|---:|"]
        for key in [
            "L_0O",
            "L_0N",
            "L_OO",
            "L_ON",
            "L_NO",
            "L_NN",
            "route_effect_old_cells",
            "cell_effect_new_connection",
            "interaction",
            "numerical_resolution",
        ]:
            lines.append("| " + key + " | " + repr(cross[key]) + " |")
        lines += [
            "",
            "Attribution: " + str(at["attribution_status"]),
            "Cell parameters bitwise equal: " + str(row["parameters"]["all_cells_bitwise_equal"]),
            "Training/inference distribution assessment: "
            + str(at.get("train_inference_distribution_assessment")),
            "Next research recommendation: " + str(at.get("next_research_recommendation")),
            "Training path difference steps: " + str(row["trajectory"]["path_difference_steps"]),
            "Training-distribution entropy and hard-action match probabilities are fixed-state descriptive measures, not closed-loop path agreement.",
            "No functional specialization or general convergence claim follows from an interaction term.",
            "",
        ]
    lines += [
        "## Next research decision",
        "",
        "If route-only benefit dominates, prioritize a separately versioned train/inference routing-alignment experiment.",
        "If Cell adaptation or favorable interaction is resolved above numerical error, design multi-seed replication before claiming general benefit.",
        "If evidence is incomplete or historical NLL does not reproduce, resolve the input/evaluation discrepancy first.",
        "This audit never starts that next experiment.",
    ]
    (root / "SCIENTIFIC-FINDINGS.md").write_text("\n".join(lines) + "\n")
    files = [
        {"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.iterdir())
        if p.is_file() and p.name != "RUN_MANIFEST.json"
    ]
    manifest = {
        **identity,
        "run_status": status,
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_updates": 0,
        "exit_code": 0 if valid else 1,
        "command": sys.argv,
        "output_directory": str(output.resolve()),
        "publication_record_scope": "PUBLIC_PRE_PUSH_SNAPSHOT; LOCAL_MANIFEST_UPDATED_AFTER_REMOTE_VERIFICATION",
        "data_roles": ["calibration"],
        "input_reference": None if inputs is None else inputs.reference,
        "input_files": [] if inputs is None else inputs.files,
        "artifact_files": files,
        "publication_status": publication["status"],
        "scope": "READ_ONLY_EXPLORATORY",
    }
    if publication.get("status") == "PUBLISHED":
        manifest["remote_commit_sha"] = publication["remote_commit_sha"]
    atomic(root / "RUN_MANIFEST.json", manifest)
    return summary
