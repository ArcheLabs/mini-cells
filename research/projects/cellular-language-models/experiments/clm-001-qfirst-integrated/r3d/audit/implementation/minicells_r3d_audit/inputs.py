"""Restore only requested final objects. Private storage needs read access only."""

from pathlib import Path
import os
import shutil
from minicells_artifacts.hf_storage import HubBackend, relative
from minicells_runtime.store import safe, ContinuationError
from minicells_r3c_moe_bridge.checkpoint import Store
from minicells_r3c_moe_bridge.development import verified_manifest
from .data import load as load_assets, restore as private_asset_subset
from minicells_r3d.connection import PathModel
from .common import atomic, digest, read, checkpoint_identity, run_identity, fingerprint


def restore_object(source, root, row):
    """Validate a destination before creation; verify bytes using the existing Store afterwards."""
    target = Path(root) / relative(row["file"])
    for part in (target,) + tuple(target.parents):
        if part.is_symlink():
            raise ContinuationError("UNSAFE_PATH", row["file"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != row["sha256"]:
            raise RuntimeError("CHECKPOINT_CACHE_COLLISION")
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    return safe(root, row["file"])


def secret(name, required=False):
    value = os.environ.get(name, "").strip()
    if not value:
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret(name).strip()
        except Exception:
            pass
    if not value and name == "HF_TOKEN":
        try:
            from huggingface_hub import get_token

            value = get_token() or ""
        except ImportError:
            pass
    if not value and required:
        raise RuntimeError("READ_ONLY_INPUT_UNAVAILABLE_ENABLE_EXISTING_" + name)
    if value:
        os.environ[name] = value
    return value


class Inputs:
    def __init__(self, c, original, historical, output, local=None):
        self.c = c
        self.original = original
        self.historical = historical
        self.output = Path(output)
        self.local = Path(local or c["runtime"]["local_training_output"])
        self.backend = None
        self.reference = None
        self.manifest = None
        self.descriptors = {}
        self.hub_error = None
        self.manifest_error = None
        self.rejected = []
        self.expected = run_identity(c, original)

    def hub(self):
        if self.hub_error is not None:
            raise self.hub_error
        if self.backend is None:
            try:
                secret("HF_TOKEN", required=True)
            except RuntimeError as error:
                self.hub_error = error
                raise
            self.backend = HubBackend(self.original["parent"]["hf_reference"]["hf_repo"])
        return self.backend

    def resolve_manifest(self):
        if self.manifest_error is not None:
            raise self.manifest_error
        try:
            return self._resolve_manifest()
        except Exception as error:
            self.manifest_error = error
            raise

    def _resolve_manifest(self):
        backend = self.hub()
        candidates = []
        for path in (
            self.local / "hf-reference.json",
            self.output / "private/input-reference.json",
        ):
            if path.exists():
                candidates.append(read(path))
        if self.historical.get("hf_reference"):
            candidates.append(self.historical["hf_reference"])
        for reference in candidates:
            try:
                manifest = verified_manifest(backend, reference, self.expected)
            except Exception as error:
                self.rejected.append(
                    {
                        "manifest_path": reference.get("manifest_path"),
                        "reason": type(error).__name__,
                    }
                )
                continue
            self.reference = reference
            self.manifest = manifest
            break
        if self.manifest is None:
            revision = backend.info().sha
            pointer = "r3d/" + self.expected["run_id"] + "/latest.json"
            reference = backend.optional(pointer, revision)
            if not reference:
                raise RuntimeError("R3D_FINAL_MANIFEST_NOT_FOUND_FOR_RECORDED_IDENTITY")
            reference = {**reference, "hf_repo": backend.repo, "hf_revision": revision}
            self.manifest = verified_manifest(backend, reference, self.expected)
            self.reference = reference
        atomic(self.output / "private/input-reference.json", self.reference)

    def object(self, name):
        row = self.manifest["files"].get(name)
        if row is None:
            raise RuntimeError("R3D_MANIFEST_OBJECT_MISSING: " + name)
        path = self.hub().get(row["object"], self.reference["hf_revision"])
        if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise RuntimeError("R3D_INPUT_OBJECT_HASH_MISMATCH: " + name)
        return path

    def descriptor(self, seed, arm):
        key = str(seed) + "/" + arm
        if key in self.descriptors:
            return self.descriptors[key]
        name = "seed-%s-arm-%s" % (seed, arm)
        directory = self.local / "private-checkpoints/jobs" / name
        target = self.c["checkpoint_steps"][arm]
        ident = checkpoint_identity(self.c, self.original, seed, arm)
        receipts = [read(p) for p in (directory / "receipts").glob("*.json")]
        matches = [r for r in receipts if r.get("step") == target and r.get("identity") == ident]
        if matches:
            row = matches[0]
            try:
                path = safe(directory, row["file"])
            except ContinuationError as error:
                if error.code != "CHECKPOINT_FILE_MISSING":
                    raise
                matches = []
        if matches:
            origin = "EXISTING_KAGGLE_OUTPUT"
            root = directory
        else:
            if self.manifest is None:
                self.resolve_manifest()
            prefix = "jobs/" + name + "/"
            row = read(self.object(prefix + "latest.json"))
            if row["identity"] != ident or row["step"] != target:
                raise RuntimeError("FINAL_CHECKPOINT_IDENTITY_OR_STEP_MISMATCH: " + key)
            receipt = Store(Path(prefix), ident).receipt_file(target).as_posix()
            if receipt not in self.manifest["files"]:
                raise RuntimeError("FINAL_CHECKPOINT_RECEIPT_MISSING: " + key)
            if read(self.object(receipt)) != row:
                raise RuntimeError("LATEST_RECEIPT_CONFLICT: " + key)
            root = self.output / "private/checkpoint-cache" / name
            # Object restoration is lazy: a completed audit receipt can reuse its previous result.
            path = None
            origin = "PINNED_HF_OBJECT"
        if row.get("identity") != ident or row.get("step") != target:
            raise RuntimeError("CHECKPOINT_IDENTITY_MISMATCH: " + key)
        d = {"row": row, "root": root, "source_file": path, "origin": origin, "name": name}
        self.descriptors[key] = d
        return d

    def model(self, seed, arm, device):
        d = self.descriptor(seed, arm)
        row = d["row"]
        root = d["root"]
        if d["source_file"] is None:
            prefix = "jobs/" + d["name"] + "/"
            source = self.object(prefix + row["file"])
            restore_object(source, root, row)
        state = Store(root, row["identity"]).verified(row)
        expected_tokens = row["step"] * self.original["training"]["train_tokens_per_step"]
        if state["step"] != row["step"] or state["data_cursor"] != expected_tokens:
            raise RuntimeError("CHECKPOINT_CURSOR_MISMATCH")
        # All complete-state components must exist, but no optimizer or training RNG is restored.
        fields = {
            "model",
            "cell_optimizer",
            "connection_optimizer",
            "data_rng",
            "route_rng",
            "cf_rng",
            "global_cpu_rng",
            "global_cuda_rng",
            "python_rng",
            "numpy_rng",
            "optimizer_identity",
            "frozen_sha256",
            "policy_updates",
        }
        if not fields.issubset(state):
            raise RuntimeError("INCOMPLETE_TRAINING_STATE")
        if (state["cell_optimizer"] is not None) != (arm in ("F", "P", "J")) or (
            state["connection_optimizer"] is not None
        ) != (arm in ("L", "M", "J")):
            raise RuntimeError("CHECKPOINT_ARM_STATE_MISMATCH")
        model = PathModel(self.original, seed)
        model.load_state_dict(state["model"], strict=True)
        model.force_original = arm in ("F0", "F")
        import itertools

        expected_pairs = [list(p) for p in itertools.combinations(range(6), 2)]
        if model.connection.pairs.tolist() != expected_pairs:
            raise RuntimeError("CHECKPOINT_ACTION_SPACE_MISMATCH")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model = model.to(device)
        evidence = {
            "seed": seed,
            "arm": arm,
            "step": row["step"],
            "checkpoint_sha256": row["sha256"],
            "state_fingerprint": row["state_fingerprint"],
            "source_sha": row["identity"]["source_sha"],
            "config_sha256": row["identity"]["config_sha256"],
            "restore_origin": d["origin"],
            "full_state_hash_verified": True,
            "model_structure_verified": True,
            "action_pairs": model.connection.pairs.cpu().tolist(),
            "component_fingerprints": {
                kind: fingerprint({k: v for k, v in state["model"].items() if predicate(k)})
                for kind, predicate in {
                    "cells": lambda k: k.startswith("cells."),
                    "gate": lambda k: k.startswith("gate."),
                    "correction": lambda k: k.startswith("connection."),
                    "shared": lambda k: not k.startswith(("cells.", "gate.", "connection.")),
                }.items()
            },
            "optimizer_created": False,
            "cross_session_verified": False,
            "session_proof": "Successful object restoration is recorded; independent Kaggle version provenance not inferred from a new UUID.",
        }
        return model, evidence

    def assets(self, explicit=None):
        sha = self.original["engine"]["input"]["assets_manifest_sha256"][:12]
        paths = (
            [Path(explicit)]
            if explicit
            else [
                Path("/kaggle/working/r3c-pilot-assets-" + sha),
                Path("/kaggle/working/r3c-assets-" + sha),
                self.output / "private/assets",
            ]
        )
        for path in paths:
            if (path / "manifest.json").exists():
                return load_assets(self.original, path), path
        destination = paths[-1]
        data = private_asset_subset(
            self.hub(),
            self.original["engine"]["input"]["asset_bundle_ref"],
            destination,
            self.original,
        )
        return data, destination

    def provenance(self):
        return {
            "expected_training_identity": self.expected,
            "hf_reference": self.reference,
            "rejected_references": self.rejected,
            "hf_access": "READ_ONLY; no auth_check(write) or create_commit",
            "cross_session_verified": False,
        }
