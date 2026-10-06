# Network access and trust

This guide explains who can join a network, how stores become discoverable and
which limits apply. See the [operator guide](operator.md) for setup and the
[user guide](user.md) for downloads.

## Access scope

| Capability | Open network | Private network |
|---|---|---|
| Join | Anyone with a node address | Holders of the network key |
| Search the metadata index | All participants | All participants |
| Download seeded chunks | All participants | All participants |
| Requests without the network key | Accepted subject to normal checks | HTTP 403 on every data-port endpoint |

Access is granted to the whole network. Every participant can see its metadata
and download its seeded data. Use separate networks for separate access groups.
Dataset-specific permissions and per-user quotas are not implemented.

All networks verify signed announcements and chunk identities. DHT addresses,
metadata index records and publisher names are signed with Ed25519 keys.
Only the key owner can update `zs://name@<pubkey>`.
The control API is local and checks `X-Zs-Client` and `Host`.

Private-network requests carry timestamped HMAC authentication. HTTP payloads
and metadata are unencrypted, and anyone who obtains the network key can join.
Use a VPN when channel confidentiality is required.

## Create and join a network

```bash
zarrswarm init --bootstrap-node --public-host data.example.org
zarrswarm init --bootstrap-node --public-host data.example.org --private
```

Initialization prints an invitation:
`zsnet://<id>@<host>:<port>`, with `?k=<network-key>` for a private network.
Share a private invitation only with its participants.

```bash
zarrswarm init --join 'zsnet://<id>@data.example.org:7881?k=<network-key>'
zarrswarm node
```

The key is saved as `network_key` in `~/.zs/config.toml`, created with mode
`0600`. `ZS_NETWORK_KEY` can supply it instead. Check that `zarrswarm status`
reports DHT contacts.

Any reachable public node can provide entry to the DHT. To add one, join with
`--listen-public`, set `relay_server = true` if it should relay NAT nodes, and
add its URL to other nodes' `bootstrap = [...]`. The network key stays the same.

## Node addresses

| Item | Source | Form |
|---|---|---|
| Node ID | Hash of the public key in `node.key` | 40 hexadecimal characters |
| Public key | `zarrswarm status` | 64 hexadecimal characters |
| Public address | `--public-host` or bootstrap reachability check | `http://HOST:7881` |
| NAT address | Assigned through the relay | `http://RELAY:7881/r/<id>` |

Nodes publish their signed address at startup and every ten minutes by default.
DHT records expire after one hour. Restart a node after its public address
changes so it announces the new address.

### Multiple networks on one host

Run a separate node with its own state directory and ports for each network:

```bash
ZS_HOME="$HOME/.zs-project" zarrswarm init --join 'zsnet://<id>@HOST:7881?k=<network-key>' --port 7891
ZS_HOME="$HOME/.zs-project" zarrswarm node
```

In another terminal, select it with
`ZS_CTL=http://127.0.0.1:7892 zarrswarm search t2m`.

## Register stores

Registering a store starts seeding it. The node scans the Zarr store, computes
its grid links and announces the data through the DHT.

```bash
zarrswarm seed /data/era5.zarr
```

To seed on every startup, add `[[seed]]` entries to `config.toml`.

| Published item | Location | Contents |
|---|---|---|
| Grid link | Computed from coordinates | Dimensions, normalized coordinates and time grid |
| Holder record | DHT under the grid ID | Address and bandwidth hint |
| Search index | DHT under metadata tags | Variable names, attributes, time range and dimensions |
| Manifest | Served by the holder | Chunk identities, sizes, layouts, codecs and signed metadata |
| Chunks | Served on request | Stored bytes or a supported transfer encoding |

Local filesystem paths are not included in public manifests. Seeding reads
stores in place without copying or rewriting them.

Compatible stores can differ in time coverage, variables, chunking, codecs,
Zarr version and aligned sample strides. Variables on different coordinate
grids produce multiple links, joined as `zs://g1+g2`.

### Publisher names

```bash
zarrswarm name era5 'zs://<grid>'
```

The resulting `zs://era5@<pubkey>` name can be updated to another grid.
The node republishes it while running. Publishers can use the same short name;
their public keys distinguish the links.

### Trusted publishers

```toml
trust = ["<publisher-public-key>"]
```

Trusted publishers take precedence when holders disagree about values.
Pushdown responses from trusted holders skip sampled audits, so this setting
also changes the client's trust assumptions.

[Source contracts](source-contracts.md) bind source packing parameters to a
field, grid and units. Accepting one requires its publisher's trusted key.
A cached mirror retains the original signature and does not become a new
authority merely by serving a downloaded copy.

### Stop seeding

```bash
zarrswarm unseed /data/era5.zarr
```

The node stops serving the registered store. DHT records expire within an hour.
Remove a matching `[[seed]]` entry too, or startup will register it again.
Copies already downloaded by other participants remain with them.

## Resource limits

| Owner setting | Control | Default |
|---|---|---|
| Upload rate | `upload_mbps` or `--upload-mbps` | Unlimited |
| Stores to serve | `seed` and `[[seed]]` entries | Explicitly registered stores |
| Network membership | `--private` and network key | Open |
| Relay operation | `relay_server` | Enabled for bootstrap nodes |
| Download cache | `cache_max_gb` | Unlimited |
| Repair storage | `parity --k K` | No parity retained unless requested |

There are no daily per-user byte quotas or per-participant download accounting.

| Built-in limit | Value |
|---|---|
| Node request body | 64 MB |
| Chunk request batch | 64 chunks or 32 MB |
| Uncompressed manifest | 1 GB |
| NAT nodes per relay | 2,000 |
| Concurrent relay requests | 32 per attached node, 512 total |
| DHT publishers per key | 2,000 |
| DHT keys per node | 100,000 |
| DHT record lifetime | One hour |
| Future timestamp allowance for DHT records | Five minutes |

A rejected request fails independently; clients may retry another holder.

## Supported inputs and verification limits

Holders need local or mounted Zarr v2/v3 stores. NFS, Lustre and mounted object
storage can work; direct S3 URLs are not seed inputs. NetCDF is an export format,
not a directly seedable input. Current support uses flat groups, standard
calendars and regular time axes.

Grid links change when dimensions, coordinates or the time quantum change.
Aligned layouts with different sample strides can share a grid.

Independent lattice fitting can merge sparse fields whose source codes differ.
Use exact mode when bitwise agreement is required, or a trusted source contract
when its packing parameters and decoder-error bound are valid.

Ordinary holder voting is vulnerable to a Sybil majority. Configure an appropriate
publisher trust anchor when majority voting is insufficient. Signatures establish
who announced data; they do not validate the science behind it.

Network-key rotation is manual: update all remaining participants and restart.
Metadata is visible to every network participant.

## Cache migration and query coverage

The current lattice identity format stores per-slice levels, steps and code
radii. Legacy lattice announcements require reseeding. On startup the node
reindexes cached chunks without deleting their data. Old parity announcements
are withdrawn, so regenerate parity when its identity format changes.

For a regional download, `done` requires every requested time sample in every
required spatial tile. Coverage reports `requested_samples`, `covered_samples`
and `missing_samples`. Successfully transferring a selected but incomplete
cover yields `partial`, not a complete answer. Missing data raise a read error
rather than being silently accepted.
