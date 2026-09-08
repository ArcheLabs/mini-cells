from __future__ import annotations

import json

import pytest

from minicells import CellMutation
from minicells.hybrid.errors import ArtifactValidationError


def test_malformed_artifact_is_rejected(tmp_path) -> None:
    root = tmp_path / "broken"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ArtifactValidationError):
        CellMutation.from_pretrained(root)

