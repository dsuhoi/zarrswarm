# Contributing to ZarrSwarm

Install the locked development environment using the instructions in [README.md](README.md).
Run `python -m pytest -q tests` from that environment before opening a pull request.

For documentation changes, run:

```bash
uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs build --strict
```

Implementation changes belong in `zarr_torrent/`; `zarrswarm/` exposes the public package API.
Keep the existing commands, link schemes and configuration compatible, or document a migration.
Add a focused regression check for a behavior change. Use small, generated fixtures for local tests.

When dependencies change, update `pyproject.toml` and run `uv lock`.
For build changes, run `uv build` and check the installed wheel from outside the source checkout.

Research measurements belong under `bench/` and `sim/`. Preserve existing result files and measured
source snapshots. Pass a separate `--out` path for a new run, and record the source version, input hashes,
configuration, completion checks and comparison scope. Label simulated and local results accordingly.
The [experiment guide](docs/experiments.md) describes the existing workflows.

For a bug report, include the package/Python/Zarr versions, a small reproducer, the node role and relevant
logs. Remove private network invitations and credentials before posting them in a public issue.
