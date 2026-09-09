# PyPI Trusted Publishing setup

MiniCells publishes tagged software releases through PyPI Trusted Publishing;
no long-lived PyPI token belongs in repository secrets.

One-time administrator steps:

1. In PyPI, add a Trusted Publisher for owner `ArcheLabs`, repository
   `mini-cells`, workflow `release.yml`, and environment `pypi`.
2. In GitHub, create the protected environment named `pypi` and optionally
   require an approval before publishing.
3. Confirm the package project is owned by the ArcheLabs PyPI account.

Per release, a maintainer reviews the immutable commit, creates and pushes a
`v<package-version>` tag, and approves the environment if required. The
workflow validates the tag, builds the wheel and sdist, runs `twine check`,
performs a clean-wheel smoke test, publishes to PyPI, and creates a GitHub
prerelease.
