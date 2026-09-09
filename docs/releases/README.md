# MiniCells release flow

The software and HybridCLM artifact releases are independent trust domains.

## Software release

Review and freeze the commit, then create and push the matching tag:

```bash
git checkout main
git pull --ff-only
git tag -s v0.2.0a1 -m "MiniCells v0.2.0a1 — HybridCLM Research Preview"
git push origin v0.2.0a1
```

The tag-driven `release.yml` workflow validates tag/version identity, builds
and tests the wheel/sdist, publishes through PyPI OIDC, and creates a GitHub
prerelease.

## HybridCLM artifact release

Open `notebooks/kaggle/hybrid_clm_release_v0_2_0a1.ipynb`, enable Internet and
the `HF_TOKEN` Kaggle secret, review the dry run, then set `PUBLISH=True`.
Run All. The notebook checks out the immutable tag, exports and validates the
engineering mutation, renders the Model/Space assets, and publishes only after
the `READY_FOR_PUBLICATION` gate.

The notebook never writes GitHub or formal-seed state. Formal validation stays
separate, and locality remains reported as unresolved under the frozen gate.
