"""Public report attestations, not a claim of private Kaggle Output readback."""

from .common import ROOT, canonical, digest, source_sha

BASELINE_SHA = "d796eb815fb77549a594713a0d3bc5001bebbe4f"
CF_BRANCH = "results/r3b-cf001-474b7cfc13030107"
PUBLIC_EVIDENCE = {
    "reviewed_on": "2026-10-20",
    "method": "GitHub connector public JSON file reads",
    "cf_branch": CF_BRANCH,
    "manifest_git_blob": "bed741d51ab498b782ca271f5e600d882abf1fdc",
    "coverage_git_blob": "9ab412dfdc0e4623c6c991f4fc7072e2df9adf4a",
    "old_preservation_git_blob": "33aab573277895cd409f5965c7391250672a04e4",
    "old_analysis": {
        "public_report_status": "OLD_ANALYSIS_PERSISTENCE_VERIFIED",
        "verified_tasks": 59,
        "kaggle_version": 69,
        "new_private_readback_by_this_task": False,
    },
    "cf001": {
        "computed_chunks": 2306,
        "saved_output_readback_verified_chunks": 0,
        "public_status": "COMPUTED_AWAITING_SAVE",
        "status": "BLOCKED_PERSISTENCE",
    },
    "private_data_read": False,
    "blocks_independent_synthetic_engineering": False,
    "dependent_private_analysis_authorized": False,
}


def provenance(c):
    relative = "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated"
    files = [
        relative + "/r3b/config/proposed.json",
        relative + "/r3b/reports/runs/33734fdd1558/checkpoint-index.json",
        relative + "/r3/implementation/clm001_r3/model.py",
        relative + "/r3/implementation/clm001_r3/training.py",
    ]
    return {
        "source_sha": source_sha(),
        "branch_base_sha": BASELINE_SHA,
        "config_sha256": canonical(c),
        "reference_file_sha256": {name: digest(ROOT / name) for name in files},
        "input": c["input"],
        "phase0_public_review": PUBLIC_EVIDENCE,
        "confirmation": "SEALED",
        "scientific_decision": False,
    }
