# User guide

Start a local node with `zarrswarm node`; see the [operator guide](operator.md).
Commands and Python clients contact its control API at `127.0.0.1:7882`.
Set `ZT_CTL` to use another local port.

## Local checks

From the repository root:

```bash
uv sync --frozen --extra test --extra netcdf --inexact
.venv/bin/python -m pytest -q tests/test_e2e.py tests/test_scenarios.py tests/test_cli.py
```

The tests generate small Zarr stores and loopback nodes with separate state
directories and ports. The CLI test runs two separate processes in a private
network. Other scenarios use real HTTP connections between nodes in one process.
No large dataset or external service is required, and the tests stop their nodes
when they finish.

The subset covers mixed Zarr v2/v3 layouts, merged replicas, relays, xarray,
local Dask, prefetch, search and NetCDF/Zarr export. It contains 18 tests.
Run the full suite with `.venv/bin/python -m pytest -q tests`.

To try your own store, start a node, run
`.venv/bin/zarrswarm seed /path/to/data.zarr` and pass its `zt://...` link to
`zarrswarm.open_dataset`.

## Terminal interface

```bash
zarrswarm-tui
zarrswarm-tui http://127.0.0.1:7892
```

The dataset list shows local and available chunk counts, progress, status,
transfer rates, peers and estimated time remaining. Filters select all datasets,
active downloads, seeded datasets, completed downloads or errors. The status bar
shows aggregate rates, DHT contacts, network access mode and whether the node is
public or behind NAT.

Select a dataset to inspect its metadata, tasks, holder list, log and piece map.
The piece map marks chunks held locally, by several peers, by one peer or by none.

| Key | Action |
|---|---|
| `a` | Add a download; select variables, time, region, strategy and optional export or follow window |
| `s` | Seed a directory after previewing its grids, variables, layouts and size |
| `/` | Search the metadata index; Enter opens the selected result |
| `p` | Pause or resume a download |
| `Del` | Stop a download or unseed a store, with confirmation |
| `o` | Edit node settings; restart the node to apply them |
| `q` | Quit the interface |

## Dataset links

| Kind | Example | Meaning |
|---|---|---|
| Grid | `zt://8df9...` | Copies with compatible dimensions, coordinates and a common time grid |
| Multiple grids | `zt://8df9...+41aa...` | Variables on different grids, such as surface and pressure-level fields |
| Publisher name | `zt://era5@<pubkey>` | A signed, updateable name published with `zarrswarm name` |

A grid link identifies compatible array coordinates. Time ranges, variable sets,
chunk shapes, codecs and Zarr v2/v3 encoding may differ between its holders.
Availability and agreement of values are checked separately.

## Find data

```bash
zarrswarm search 2m_temperature
zarrswarm peers 'zt://<grid>'
```

Search uses variable names, `standard_name` and `long_name`.
Replace `<grid>` with an actual result or the link printed by `seed`.

## Download a subset

```bash
zarrswarm get 'zt://<grid>' --vars t2m --time 2020-01-01:2020-03-01 --out t2m.zarr
zarrswarm get 'zt://<grid>' --vars t2m --sel lat=40:60,lon=0:30 --out eu.nc
zarrswarm get 'zt://<grid>' --vars t2m --chunking time=8760,lat=1,lon=1 --out ts.zarr
zarrswarm get 'zt://<grid>' --vars t2m --progressive
```

Use your store's variable names, coordinates and dates. NetCDF output needs the
`netcdf` extra, for example `python -m pip install -e '.[netcdf]'`.

The planner compares layouts and holders using estimated transfer time,
receiver capacity and shared relays. It rounds the assignment to whole chunks
and replaces failed or slow holders during transfer. JLPS is a heuristic;
neither global optimality nor a speedup over the minimum-byte cover is guaranteed.
Received whole chunks are checked against their byte and value identities.

## Follow a stream

```bash
zarrswarm follow 'zt://era5@<pubkey>' --vars t2m,tp --last 7d --sel lat=40:70,lon=20:60
zarrswarm follows
zarrswarm unfollow <id>
```

Every `follow_every_s` seconds, the node fetches a window ending at the newest
sample available in the swarm. The default interval is five minutes. This works
for historical archives as well as live streams. A named link is resolved again
on each update, so a subscription follows the publisher's new target.
Subscriptions survive node restarts. Set `cache_max_gb` to limit retained data.

## Pause and resume

The TUI's `p` key uses `POST /api/pause/<job>` and `POST /api/resume/<job>`.
Pausing stops work after current batches and keeps downloaded chunks.
Resuming restarts the same job and uses its cached data.

## xarray

```python
import zarrswarm as zs

ds = zs.open_dataset("zt://<grid>", chunks={}, pushdown=False)
subset = ds["t2m"].sel(time=slice("2020-01", "2020-02"))
zs.prefetch(subset)
result = subset.mean("time").compute()

series = zs.open_dataset(
    "zt://<grid>", chunking={"time": -1, "lat": 1, "lon": 1},
    pushdown=False,
)
```

`open_dataset` returns a normal `xarray.Dataset`. By default, reads are lazy
without Dask; `chunks={}` enables local Dask computation. `chunking=` sets the
virtual Zarr chunks exposed to xarray, while `chunks=` controls Dask.
Virtual rechunking does not rewrite holder stores.

Whole-chunk reads transfer the full source chunk. A small `.isel()` or `.sel()`
alone does not guarantee less network traffic. Optional pushdown requests signed
slices and checks a sample against whole chunks. Set `pushdown=False` for full
whole-chunk validation.

`prefetch` performs a dry read to identify chunks and may allocate memory the
size of its input. Call it on the selected subset. Sequential time reads can
prefetch upcoming view chunks with `ZT_READAHEAD`.

The view is read-only. Save results separately with `to_zarr` or `to_netcdf`.
An open Dataset contains a metadata snapshot; reopen it after source metadata
changes. Use `zs.open_dataset` for these links: a native
`xr.open_dataset("zt://...")` backend is not registered.
Multi-machine Dask execution has not been validated.

## Approximate means

```bash
zarrswarm mean 'zt://<grid>' t2m --time 2020-01-01:2020-12-31 --rel-err 0.001
```

```python
estimate = zs.progressive_mean_vas(ds, "t2m", rel_err=0.001)
```

VAS samples time strata and allocates additional chunks using estimated
variance. It stops when the estimated confidence interval meets the requested
tolerance. `--method vdc` uses a van der Corput order without stratification.
These are approximate estimates; the confidence interval relies on the
sampling assumptions.

## Share data

```bash
zarrswarm seed /data/my_run.zarr
zarrswarm name my-run 'zt://<grid>'
```

Seeding reads the original store without copying it. Compatible replicas can
extend the same grid link with different time ranges or chunk shapes. For
different time strides, sample alignment is checked before pooling coverage.
See [source contracts](source-contracts.md) when you have trusted source packing
parameters.
