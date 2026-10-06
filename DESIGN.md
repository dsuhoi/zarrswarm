# ZarrSwarm architecture

ZarrSwarm joins compatible gridded datasets held at independent sites. A query can use copies with different
chunk shapes, codecs, time ranges and Zarr versions. The client chooses which chunks and holders to use,
checks the received data and presents the result through a read-only xarray view.

This document describes the current implementation. The [experiment guide](docs/experiments.md) and
[measured snapshots](bench/revalidation/README.md) identify the versions used for each published result.

## From a local store to a shared view

1. A holder scans a Zarr store, normalizes its coordinates and time units, and identifies its grid.
2. It records each physical layout separately, including chunk boundaries and time stride.
3. It computes byte and value identifiers for chunks and signs a manifest.
4. The DHT advertises the holder under the grid identifier and searchable metadata tags.
5. A receiving node fetches holder manifests and resolves compatible fields and chunk identities.
6. It selects a complete cover for the requested samples, assigns transfers, verifies data and caches chunks.

Discovery does not copy the dataset. Holders serve their original stores; receivers cache the chunks they use.
Copies need compatible coordinates and field definitions. Re-encoding unrelated measurements does not make them
interchangeable.

## Grid, field, layout and chunk identities

These identities answer different questions:

| Identity | What it identifies | Main implementation |
|---|---|---|
| Grid | Coordinates, dimensions and a common time quantum | `zarrswarm/scan.py` |
| Field | Variable name, dimensions, units and compatible value representation | `zarrswarm/scan.py`, `zarrswarm/node.py` |
| Layout | Chunk shape and phase, time stride and offset, Zarr encoding metadata | `zarrswarm/scan.py` |
| Byte identifier (`cid`) | One encoded chunk payload | `zarrswarm/codec.py` |
| Value identifier (`vcid`) | Decoded values or source integer codes, under the selected identity mode | `zarrswarm/codec.py` |

The grid's time quantum is the largest member of one day, one hour, one minute or one second compatible
with the store's regular time stride. Aligned hourly and six-hourly copies can therefore share a grid.
Their layouts still describe which samples each copy actually contains. Chunk keys use global time indices,
so a local replica's starting date does not shift every key.

For a request with time stride `S` and offset `O`, a layout with stride `r` and offset `o` can cover the
requested lattice when `r` divides `S` and the offsets agree. The planner also checks spatial coverage and
actual available chunks. A matching grid identifier alone does not establish a complete answer.

Input support currently covers flat Zarr v2/v3 groups, standard calendars and regular time axes. General
nested groups, irregular timelines and alternate calendars are outside the tested ingest path.

## Value comparison

### Exact mode

Exact identity hashes canonical decoded values, including their dtype and shape. Integer data and
non-lattice fallbacks use exact comparison. Byte identity instead groups only identical encoded payloads;
rechunking or a codec change can split a byte swarm even if decoded values are identical.

### Fitting mode

The default floating-point path fits a lattice to each time slice independently. The current estimator is
called `lattice-v5`; its serialized identifier starts with `L3:`. It records a hash and per-slice fitted
level, step and radius. Relative integer codes, shape and time-axis placement enter the hash. Finite,
NaN and positive/negative infinity masks are treated separately. Canonical byte order is little-endian.

Comparison checks compatible hashes and the per-slice fit parameters. It is not a universal guarantee that
one original source count cannot change. In the retained sparse-field control, fitting accepts 13 of 20
changed cases. Dense provider controls exercise a different distribution; they do not remove that boundary.

### Source-contract mode

A publisher can supply the source quantizer's `step`, `origin` and `max_error`, either once for a field or
per timestamp. The decoder-error bound must be positive and strictly below half a source step. A node can
then recover absolute source integer codes and verify reconstruction against that bound. Different decoder
dtypes may pool when they recover the same codes under the same authenticated field contract.

The contract binds the grid, variable and units. A publisher signs it; mirrors forward the signature.
The receiver must trust that publisher for the contract to anchor a field family. Conflicting trusted field
contracts leave the field unresolved. Code equality still depends on correct source parameters and the
claimed decoder-error bound; signatures authenticate their origin, not their scientific validity.

See the [source-contract guide](docs/source-contracts.md) for operator commands, controls and restrictions.
Arbitrary Zarr files do not automatically retain or reveal their original GRIB packing provenance.

## Manifest merging and trust

A manifest maps `variable@layout/chunk-index` to a byte identifier, value identifier, encoded size and valid
prefix information. It also carries array metadata and coordinate documents. Signed announcements let a
receiver attribute each claim to a node key.

The receiving node first resolves field families, then groups compatible holders for each chunk. Trusted
publishers and explicitly seeded local sources can establish authority; merely caching a received chunk
does not turn the cache into a new authoritative measurement. Ordinary copies provide support for a value
candidate. An unresolved tie remains unserved rather than becoming an arbitrary answer.

An append can enlarge the valid prefix of a chunk. A receiver checks both content identity and valid sample
coverage, and invalidates stale decoded data when the byte identifier or decoder metadata changes. The same
rules apply to cached chunks, transport re-encoding and pushed-down slices.

The DHT is a discovery mechanism, not a consensus service. Majority support alone provides no protection
against a sufficiently large colluding group. Trusted publisher keys are needed when that distinction matters.

## Selecting chunks and holders

A request may be covered by several physical layouts. Whole-map chunks suit map queries; small spatial tiles
or long time chunks can reduce transfer for a point series. Minimum-byte selection chooses a complete cover
with the least estimated remote payload, but ignores where those bytes must come from.

Joint Layout and Peer Selection (JLPS) considers both the cover and the transfer assignment. It uses a
dynamic-programming cover oracle, iteratively changes prices for loaded resources, and evaluates candidate
covers with a shared-capacity assignment model. The minimum-byte cover is always among the candidates.
Cached chunks have no remote transfer cost. Receiver limits and shared relay bandwidth participate in the model.

The implementation uses estimated rates, finite iterations and assignment rounding. Its bound applies to
that cost model and its assumptions. It does not establish global optimality, a fixed approximation ratio
for the joint problem, or a speedup on every observed network query. Small exhaustive controls and network
measurements are reported separately in [bench/revalidation](bench/revalidation/README.md).

Transfers probe new holders, update rate estimates and reassign unfinished work after failures. Endgame
requests can fetch the same remaining chunk from two holders. The client checks chunks before adding them
to verified coverage. If the chosen cover proves incomplete, it refreshes the catalogue once and replans;
that work is part of query time. Genuine gaps remain partial answers.

Implementation: `zarrswarm/jlps.py`, `zarrswarm/plan.py`, `zarrswarm/node.py`.

## Transport, relays and caching

Nodes use HTTP for DHT messages, manifests and chunks. A NAT holder can attach an outbound WebSocket to a
public relay; the relay forwards requests to that holder. Shared relay capacity is a planner resource.
The relay streams framed replies, while the holder's local HTTP response is buffered before framing.
This is not a fully streaming store-to-receiver path.

Private networks authenticate requests with an HMAC over the timestamp, method and path. This does not
encrypt HTTP payloads. The control API binds to loopback by default. Public deployments need appropriate
network isolation when data confidentiality or control access matters.

The receiver keeps encoded chunks on disk and caches decoded arrays by byte identity plus decoder metadata.
Metadata snapshots and signed source contracts are retained across cache restarts. An xarray Dataset opened
before metadata changes is a snapshot; reopening it refreshes the virtual view.

## Metadata updates

Chunk-Addressed Paged Trees (CAPT) divide a manifest into hash-addressed pages. A signed head contains array
metadata, coordinates, page root, count and version information. Receivers fetch changed pages instead of
reloading the entire chunk list. Page addressing is stable across appends.

Heads carry a monotonic sequence and previous-head reference. Nodes reject observed rollback and conflicting
heads. Locally published versions persist across restart; received-head tracking is kept in memory.
Legacy heads without sequence metadata remain accepted, so these checks are not a complete anti-rollback
protocol across every receiver restart.

Implementation: `zarrswarm/capt.py`, `zarrswarm/node.py`.

## Optional transfer and analysis paths

### Temporal transport encoding

A holder may send an `xt1` temporal-XOR, bitshuffle and Zstd representation on slow paths. The receiver
verifies decoded values and re-encodes its local chunk when needed. Exact round trips preserve signed zero,
NaN payloads and byte order. This wire representation does not change the stored layout.

### Value-level parity

Reed–Solomon parity operates on compatible canonical tiles rather than original compressed chunk bytes.
Cauchy coding over GF(256) allows recovery from `k` independent compatible data/parity members. Stripes
interleave times so a contiguous outage need not erase a whole stripe.

Source-contract fields use recovered absolute codes. Other fields use a lossless integer representation
when available or an exact value-byte fallback. Each stripe requires one field definition and compatible
spatial tile. Recovered data still pass content checks before entering the cache.

Implementation: `zarrswarm/parity.py`, `zarrswarm/gf.py`.

### Slices and approximate means

Optional hyperslab pushdown returns signed slices and audits an initial set and a sample of later replies
against whole chunks. It can save bytes but provides weaker immediate checking than downloading every chunk.
A discovered mismatch rolls back the current batch and marks previously accepted results from that holder
as tainted. Applications must handle that state; an earlier optimistic result cannot be silently made correct.

Approximate means use a pilot sample and stratified allocation based on observed variance. Reported intervals
use a Student/Satterthwaite estimate. Their nominal confidence level is not a proof of coverage for every field
or adaptive workload. Whole-data reads remain available when sampling assumptions do not fit the task.

Implementation: `zarrswarm/aqp.py`, `zarrswarm/node.py`.

## xarray interface

```python
import zarrswarm as zs

ds = zs.open_dataset("zs://<grid>", chunks={}, pushdown=False)
subset = ds["t2m"].isel(time=slice(0, 24))
zs.prefetch(subset)
result = subset.mean("time").compute()
```

The local node exposes a virtual read-only Zarr store. `chunks={}` enables local Dask; `chunking=` controls
the virtual Zarr layout. Neither rewrites holder stores. With pushdown disabled, selecting a small slice
still fetches whole source chunks. `prefetch` identifies chunks through a dry read and may allocate an array
the size of the selected input. Multi-machine Dask execution and a native `xr.open_dataset("zs://...")`
backend have not been validated or registered.

Implementation: `zarrswarm/store.py`. See the [user guide](docs/user.md) for exports and examples.
