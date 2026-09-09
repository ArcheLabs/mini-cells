from pathlib import Path

from tools.check_release_notebook import check


def test_hybrid_release_notebook_is_safe_orchestration() -> None:
    report = check(Path("notebooks/kaggle/hybrid_clm_release_v0_2_0a1.ipynb"))
    assert report["status"] == "NOTEBOOK_CONTRACT_VALID"
