# ZarrSwarm

ZarrSwarm shares Zarr arrays between sites and opens compatible replicas as one
`xarray.Dataset`. Nodes may hold different time ranges, variables, chunk shapes,
codecs or Zarr versions. The system discovers holders, chooses sources for a
request and verifies received chunks.

For example, one node may store hourly temperature maps, another may store
chunks suited to time series, and a third may extend the archive into the next
month. Clients use one `zt://...` link. Coordinates, sample times and value
verification determine whether copies can be combined; see
[network access and trust](network.md).

## Try it locally

Clone the [repository](https://github.com/dsuhoi/zarrswarm), install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run from its root:

```bash
uv sync --frozen --extra test --extra netcdf --inexact
.venv/bin/python -m pytest -q tests/test_e2e.py tests/test_scenarios.py tests/test_cli.py
```

These tests create small datasets and loopback nodes. They check transfers
between processes, replica merging, xarray reads, local Dask computation and
export. The [user guide](user.md) explains how to test your own data.

## Share your data

1. [Start a node](operator.md) and join a network.
2. Register a store with `zarrswarm seed /path/to/data.zarr`.
3. Pass the resulting link to Python:

```python
import zarrswarm as zs

ds = zs.open_dataset("zt://<grid>", pushdown=False)
sample = ds.isel(time=slice(0, 24)).load()
```

`open_dataset` needs a running local node. `pushdown=False` reads whole source
chunks and verifies them fully. The P2P view is read-only; save a result
separately with `sample.to_zarr(...)`. Virtual chunking, Dask and prefetch are
covered in the [xarray section](user.md#xarray).

## Guides

| Task | Guide |
|---|---|
| Find data, download a subset and use xarray | [User guide](user.md) |
| Run bootstrap nodes, relays and holders | [Operator guide](operator.md) |
| Control access and configure trusted publishers | [Network and trust](network.md) |
| Change node and client settings | [Configuration](config.md) |
| Verify data using source packing parameters | [Source contracts](source-contracts.md) |
| Reproduce the paper's measurements | [Experiments](experiments.md) |
| Preview and publish the website | [Documentation publishing](publishing.md) |
