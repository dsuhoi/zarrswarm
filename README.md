# ZarrSwarm

Verified peer-to-peer sharing of Zarr arrays, with an xarray interface.

ZarrSwarm lets independent sites share compatible gridded data even when their copies use different
chunk shapes, codecs, time ranges or Zarr versions. A local node discovers holders through a Kademlia
DHT, chooses chunks for a request, verifies downloads and exposes the combined data as an `xarray.Dataset`.
Nodes behind NAT can serve data through a relay.

[Documentation](https://dsuhoi.github.io/zarrswarm/) · [User guide](docs/user.md) ·
[Operator guide](docs/operator.md) · [Configuration](docs/config.md) · [Experiments](docs/experiments.md) ·
[Manuscript](paper/ieee_access/main.pdf)

## Install from source

Python 3.12 is the tested environment. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/dsuhoi/zarrswarm.git
cd zarrswarm
uv sync --frozen --extra test --extra netcdf
source .venv/bin/activate
```

The `test` extra includes pytest and Dask; `netcdf` enables NetCDF export. For a minimal installation
from the checkout, use `python -m pip install -e .` in a virtual environment.

The distribution, implementation package and main command are named `zarrswarm`.
`python -m zarrswarm` also runs the CLI; `zarrswarm-tui` opens the terminal interface.
Short commands `zt` and `zt-tui`, `zt://` links, `ZT_*` settings and `~/.zt` node state remain supported.

## Test on one machine

```bash
python -m pytest -q tests/test_e2e.py tests/test_scenarios.py tests/test_cli.py
```

These tests create small Zarr stores and temporary nodes on loopback. The CLI test runs two separate
processes in a private network. The remaining scenarios use real HTTP connections between node instances
in one process. They cover mixed Zarr v2/v3 layouts, merged replicas, relays, xarray reads, local Dask,
prefetch and NetCDF/Zarr export. They need no large dataset or external service and stop their nodes on exit.

Run the complete suite with `python -m pytest -q tests`.
GitHub Actions checks the package entry points, test suite and distribution build.

## Start a network

On a machine with a public DNS name and an open TCP data port, start a bootstrap and relay:

```bash
zarrswarm init --bootstrap-node --public-host data.example.org --private
zarrswarm node
```

Replace `data.example.org` with your reachable host. `zarrswarm init` prints an invitation such as
`ztnet://<node-id>@data.example.org:7881?k=<network-key>`. Pass it to the other participants.
The invitation key grants access to the entire network. Omitting `--private` creates an open network.

On another machine, or in another terminal with a separate state directory:

```bash
ZT_HOME="$HOME/.zt-client" zarrswarm init --join 'ztnet://<node-id>@data.example.org:7881?k=<network-key>' --port 7891
ZT_HOME="$HOME/.zt-client" zarrswarm node
```

In a new client terminal, set `ZT_CTL=http://127.0.0.1:7892` before using the client commands or Python.
The default node uses data port 7881 and local control port 7882; `--port 7891` sets control port 7892.
Keep the control API bound to loopback. NAT clients need outbound connectivity to the bootstrap and relay.

For persistent services, public holders, multiple bootstrap nodes and Docker, see the [operator guide](docs/operator.md).

## Share and download data

With a running local node:

```bash
zarrswarm seed /path/to/data.zarr                     # prints zt://<grid>, or a multi-grid link
zarrswarm search 2m_temperature
zarrswarm peers 'zt://<grid>'
zarrswarm get 'zt://<grid>' --vars t2m --time 2020-01-01:2020-01-02 --out subset.zarr
zarrswarm name my-data 'zt://<grid>'                  # signed, updateable name
zarrswarm-tui
```

Use your store's variable names and time range. Seeding reads the original store rather than copying it.
The node scans decoded chunks to build identities; registration costs depend on data size and decoding.
Stores can be Zarr v2 or v3. Currently supported inputs use flat groups, standard calendars and regular time axes.

To seed on every restart, add a store path under `[[seed]]` in `~/.zt/config.toml`.
The node polls for appended or revised data; [configuration](docs/config.md) describes scan and follow intervals.

## Read through xarray

```python
import zarrswarm as zs

# Requires a running local node. Use the link printed by `zarrswarm seed` or `zarrswarm search`.
ds = zs.open_dataset("zt://<grid>", chunks={}, pushdown=False)
subset = ds["t2m"].isel(time=slice(0, 24))
zs.prefetch(subset)
result = subset.mean("time").compute()
```

Replace `t2m` with your variable. `open_dataset` returns a normal `xarray.Dataset`; pass `ctl=` or set
`ZT_CTL` to use a different local node. `chunks={}` enables local Dask computation. The separate `chunking=`
option changes the virtual Zarr view without rewriting holder data, for example `chunking={"time": -1, "lat": 1, "lon": 1}`.

`pushdown=False` downloads whole source chunks for full validation. A small `.isel()` alone does not guarantee
less network traffic. Optional pushdown uses signed slices and sampled whole-chunk audits.
`prefetch` performs a dry read to identify chunks and may allocate memory the size of its input;
use it on the selected subset.

The P2P view is read-only. Save results with xarray's `to_zarr` or `to_netcdf`, and reopen the Dataset
after holder metadata changes. Use `zs.open_dataset` to open `zt://` links; a native
`xr.open_dataset("zt://...")` backend is not registered. Multi-machine Dask execution has not been validated.

## Verification and trust

Byte hashes check transferred chunks. Value identities determine which compatible copies may serve the same
request. Exact mode preserves decoded values bit for bit. Source-contract mode reconstructs source integer
codes using publisher-signed packing parameters and a valid decoder-error bound; see the
[source-contract guide](docs/source-contracts.md). Without those parameters, lattice fitting
is a heuristic and can merge some changed sparse fields.

Signatures authenticate announcements rather than the scientific truth of a measurement. Configure trusted
publisher keys when majority voting is insufficient. Private networks authenticate HTTP requests with HMAC;
they do not encrypt payloads. Use a VPN when channel confidentiality is required.

JLPS selects layouts and holders using estimated transfer costs, receiver limits and shared relays.
It is a heuristic; global optimality and a speedup on every query are not guaranteed.
The [network guide](docs/network.md) describes access, supported data and current restrictions.

## Reproduce the research

[docs/experiments.md](docs/experiments.md) maps the manuscript results to drivers and result files.
[bench/revalidation/README.md](bench/revalidation/README.md) explains measured source snapshots,
corrections and completed runs. Raw source archives and SHA256 manifests distinguish measured versions.
Some Internet results use earlier code than the current source-contract implementation; keep those versions separate.

The architecture and wire protocol are described in [DESIGN.md](DESIGN.md) and [PROTOCOL.md](PROTOCOL.md).
Large provider datasets are fetched separately by the experiment drivers. Local tests verify functionality;
they do not measure independent station deployments or establish forecast quality.

## Build the documentation

```bash
uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs serve
uv tool run --python 3.12 --with-requirements docs/requirements.txt mkdocs build --strict
```

The English website includes search and guides for users, operators and experiments. GitHub Actions checks
its build and publishes from `main` after GitHub Pages is enabled with **Source → GitHub Actions**.
[Publishing instructions](docs/publishing.md) cover local preview and Pages configuration.

## License

ZarrSwarm code is distributed under the [MIT license](LICENSE).
Provider data, third-party paper templates and fonts retain their original terms.
