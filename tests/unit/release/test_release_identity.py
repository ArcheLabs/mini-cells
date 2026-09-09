from pathlib import Path

import pytest

from scripts.release.check_release_identity import check_identity


def test_release_tag_matches_package_version() -> None:
    result = check_identity("v0.2.0a1", Path("pyproject.toml"))
    assert result["package_version"] == "0.2.0a1"


def test_release_tag_mismatch_is_hard_failure() -> None:
    with pytest.raises(ValueError, match="RELEASE_IDENTITY_MISMATCH"):
        check_identity("v0.2.0a2", Path("pyproject.toml"))
