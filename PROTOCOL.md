# ZarrSwarm wire protocol

This is an implementation reference, not a stable interoperability specification. Wire changes should be
checked against the existing clients and tests. The package and executable are named `zarrswarm`; link schemes,
`ZT_*` configuration and existing node state retain their current formats.

## Node identity and discovery

Each node has an Ed25519 keypair. Its node identifier is a 160-bit BLAKE2b digest of the public key. The HTTP
Kademlia DHT uses bucket size 8 and lookup concurrency 3. Its record types announce peers under a grid,
index metadata tags and publish signed names that point to dataset links.

A grid describes dimensions, coordinate values and a common time quantum. It is independent of chunk shape,
codec, Zarr version and local time range. Regular aligned hourly and six-hourly layouts can share a grid.
Their stride and offset still determine which global samples they contain. A holder record includes its
address, role and announced upload rate; these are discovery claims, not measured transfer guarantees.

| Route | Purpose |
|---|---|
| `/dht` | DHT messages and record lookup/storage |
| `/pex` | Peer exchange |
| `/whoami` | Address discovery |
| `/probe` | First-contact transfer-rate probe |
| `/m` | Full signed manifest |
| `/mh` | Signed paged-manifest head |
| `/mp` | Hash-addressed manifest pages |
| `/cb` | Encoded chunk batch |
| `/qb` | Selected decoded slices with signed receipts |
| `/relay/attach` | NAT holder's outbound relay attachment |
| `/r/<node-id>/...` | Request forwarded through a relay |

The local `/api/...` control routes are separate from peer transport. They submit queries, read completed
payloads and manage node state. Keep their listener private; it is not a public user API.

## Coordinates, layouts and chunk keys

The time quantum `q` is the largest compatible value in `{86400, 3600, 60, 1}` seconds. Global time indices
are derived from canonical timestamps and the grid phase. A layout records chunk shape and phase, plus a
time stride `r` and offset `o` on that grid. A request at stride `S` and offset `O` can use the layout when
`r` divides `S` and the offsets agree, subject to actual coverage.

A chunk key has the form `variable@layout/c0.c1...`. Chunk coordinates are tied to the global grid rather
than a replica's first local timestamp. The manifest entry is `[cid, vcid, size, nvalid]`: byte identity,
value identity, encoded size and valid-prefix information. Array metadata describes dtype, shape, fill
values, codecs and the Zarr representation. Coordinate documents allow a receiver to reconstruct its view.

Two arrays with the same variable name are not automatically the same field. Units, dimensions and field
identity are checked before their chunks become interchangeable.

## Content identities

`cid` hashes the encoded chunk bytes. A receiver verifies those bytes before accepting a native payload.
Alternative transport encodings and reconstructed chunks must also pass decoded-content checks.

Exact value identity hashes canonical decoded data with shape and dtype. Floating-point fitting mode uses
an `L3:<hash>:<parameters>` identifier; the current estimator label is `lattice-v5`. Parameters are JSON
records of fitted level, step and radius for individual time slices, with null entries where appropriate.
The hash covers canonical relative codes, array shape, time-axis placement and finite/NaN/infinity masks.
Numeric bytes use little-endian order. Constant and non-floating values have exact fallbacks.

The receiver compares both hashes and compatible fitting parameters. Matching fits do not prove equality of
original source counts. Sparse one-count changes can be accepted when both copies infer a coarser lattice.
The [source-contract controls](docs/source-contracts.md) quantify this boundary.

## Signed source contracts

A contract identifies a grid and field, binds its variable name and units, and supplies a source `step`,
`origin` and `max_error`. Parameters may be constant or a catalogue indexed by canonical integer UTC-second
strings. The error bound must satisfy `0 < max_error < step / 2`.

The original publisher signs the contract. Mirrors carry that signature with the field metadata. A receiver
uses it as an authoritative field anchor only when it trusts the signing key. Copies may have different
floating dtypes if they reconstruct the same absolute source codes within the authenticated bound.
Conflicting trusted field definitions are left unresolved.

The code hash includes the quantizer, shape and masks. An error budget determines admissible decoding; it
is not a substitute for the source step or origin. The implementation rejects non-finite parameters, invalid
bounds, reconstruction failures and integer overflow. This format requires supplied source metadata;
there is no general automatic provenance recovery for arbitrary Zarr inputs.

## Manifests, pages and version checks

A full manifest is signed by its holder. Paged manifests use a signed head with grid, array metadata,
coordinate documents, page root and count. Chunk entries are assigned to stable hash-addressed pages;
receivers retrieve only new pages when a head changes. The current page partition uses 64 key-hash buckets.
A manifest range request can limit the relevant chunk keys.

A published head includes `seq` and `prev`. The node advances its version beyond both its previous value
and the current millisecond timestamp, and persists its local versions. Receivers reject observed sequence
rollback and conflicting heads. Received-head tracking is in memory, and legacy heads without sequence
metadata remain supported. These limits matter when reasoning about restart or adversarial replay.

Appending data changes metadata and the affected final chunk. `nvalid` prevents uninitialized tail samples
from counting as coverage. Revisions replace byte identities; decoded caches also key on decoder metadata,
so new content cannot reuse an old decoded array solely because its chunk location is unchanged.

## Holder selection and verification

The receiver merges compatible field families and evaluates support for each chunk value. Trusted keys or
an explicitly seeded local source can resolve authority. A received cache does not create new measurement
authority. If equally supported candidates cannot be resolved, the receiver leaves the chunk unavailable.

The client chooses layouts and holders for the requested samples. It measures first-contact rates and
updates estimates during transfers. JLPS includes receiver and shared-relay capacities, local cached data
and the minimum-byte cover among its candidates. It is a heuristic under a transfer-cost model.

A complete response requires verified chunks covering every requested sample. Coverage is kept without
rounding. A selected cover that proves incomplete triggers one catalogue refresh and replan, included in
query time. Timeouts and true gaps remain partial responses; elapsed time alone does not establish success.

## Relay framing

A NAT node opens an outbound WebSocket attachment to a reachable relay. The relay forwards a request with
a request identifier `rid`. The holder returns status and stream metadata followed by frames containing
`rid` and data, then an end marker. The default frame limit is 4096 KiB.

The relay can forward frames as they arrive. The NAT holder currently buffers its local HTTP response before
sending frames, so the complete path is not store-level streaming. A relay's shared uplink must be accounted
for once across attached holders, rather than treating every holder as an independent public link.

## Private-network authentication

When `ZT_NETWORK_KEY` is set, HTTP requests carry HMAC authentication over timestamp, method and path,
including the query string. The allowed timestamp skew is 120 seconds. Nodes therefore need reasonably
synchronized clocks. Possession of an invitation key grants access to the network.

HMAC does not encrypt data. Ed25519 signatures authenticate announcements and receipts; neither establishes
scientific correctness without a valid source and trust policy. Use network isolation or an encrypted tunnel
when the deployment requires confidential transport.

## Optional temporal transport encoding

For a slow path, a client can request `xt1`: temporal XOR, bitshuffle and Zstd. The current default threshold
is 30 MB/s. The receiver checks decoded content and writes a compatible local encoding when necessary.
Byte order is normalized on the wire, and exact round trips preserve signed zero and NaN payloads.
Stored Zarr codecs and layout identities remain independent of this negotiated transport representation.

## Selected slices and audit receipts

With pushdown enabled, `/qb` replies contain selected values and signed receipts binding the grid, chunk key,
byte identifier, selection and returned-data hash. The client downloads whole chunks for an initial audit
(default: three chunks), then samples later replies (default: five percent); explicitly trusted holders can
skip audits. A receipt attributes a reply to a signer, but does not by itself prove the selection matches a
scientifically correct source.

If an audit finds a mismatch, the node rolls back the current batch and marks earlier accepted results from
that holder as tainted. Proof checking needs the relevant signed claim and verified chunk data. Applications
using optimistic pushed-down results must inspect their verification state. Use whole-chunk reads when every
accepted result must receive immediate content validation.

## Value-level parity

Parity stripes operate on compatible canonical field tiles using Cauchy Reed–Solomon coding over GF(256).
Recovery requires `k` independent compatible data/parity members, with distinct parity rows. Several parity
rows are supported. A stripe keeps one field definition and spatial tile.

For `N` time samples, stripe width `k` and interleave spacing `D = ceil(N / k)`, a member time is
`c = o + q*k*D + i*D + s`. Interleaving spreads contiguous outages across stripes. Source-contract fields use
absolute source codes. Other fields use a lossless code representation where available or exact canonical
value bytes. Reconstructed content is verified before it is exposed to the caller.

## Reference implementation and checks

The protocol is implemented in `zarrswarm/common.py`, `dht.py`, `scan.py`, `codec.py`, `capt.py`, `node.py`,
`parity.py` and `gf.py`. The local xarray store is in `zarrswarm/store.py`.

```bash
python -m pytest -q tests/test_identity.py tests/test_security.py tests/test_packing.py tests/test_parity.py
python -m pytest -q tests/test_e2e.py tests/test_cli.py
```

These checks use generated inputs and local connections. They do not establish WAN performance or an
independent station deployment; measured runs and their source versions are listed in the
[experiment guide](docs/experiments.md).
