# Operator guide

## Node roles

| Role | Requirements | Initialization |
|---|---|---|
| Bootstrap and relay | Public IP or DNS name and an open TCP data port | `--bootstrap-node --public-host HOST` |
| Public holder | An externally reachable data port | `--join ztnet://... --listen-public` |
| Node behind NAT | Outbound connections to bootstrap and relay | `--join ztnet://...` |

A NAT node attaches to a relay over a websocket and serves data through it.
Bootstrap nodes provide entry into the DHT; peers then discover each other through
Kademlia and peer exchange. Use two or three public entry points at separate
sites when one bootstrap host is insufficient.

## Start the first node

Install from source as described in the [README](https://github.com/dsuhoi/zarrswarm#install-from-source),
then activate the environment:

```bash
source .venv/bin/activate
zarrswarm init --bootstrap-node --public-host data.example.org --service
systemctl --user enable --now zt-node
```

Replace the hostname with your reachable host. The printed
`ztnet://<id>@data.example.org:7881` invitation is used by other participants.
You can run `zarrswarm node` directly instead of installing a service.
On Linux, `loginctl enable-linger "$USER"` lets a user service continue after logout.

Add `--private` to create an invitation containing a network key.
[Network access](network.md) describes its scope.

Open incoming TCP port 7881. The control API uses port 7882 on loopback and
requires `X-Zt-Client` and a local `Host`. Keep this control port local.

## Join other nodes

```bash
zarrswarm init --join 'ztnet://<id>@data.example.org:7881' --service
zarrswarm init --join 'ztnet://<id>@data.example.org:7881' --listen-public --service
```

The first form uses a relay. With `--listen-public`, the node asks bootstrap to
check its address and port using `/whoami` and `/probe`. If unreachable, it uses
the relay. Add other entry points to `bootstrap = [...]` in `config.toml`.

## Docker

```bash
docker build -t zarrswarm .
mkdir -p "$HOME/zt-home"
docker run --rm --user "$(id -u):$(id -g)" -v "$HOME/zt-home:/zt" zarrswarm init --join 'ztnet://<id>@data.example.org:7881'
docker run -d --name zarrswarm --restart unless-stopped --user "$(id -u):$(id -g)" --network host -v "$HOME/zt-home:/zt" -v /data:/data:ro zarrswarm node
docker exec zarrswarm zarrswarm status
```

On Linux, host networking exposes the data port and keeps the control port on
host loopback. Host CLI and xarray clients can use their normal settings.
If a container's data paths differ from the host's, host clients read chunks
through the node's `/api/read` endpoint.

With bridge networking, publish only `-p 7881:7881` and run control clients
inside the container using `docker exec`. The node files belong to the UID/GID
specified in the example.

## Seed stores and manage disk use

```toml
# ~/.zt/config.toml
upload_mbps = 50

[[seed]]
path = "/data/era5/*.zarr"
```

`upload_mbps` is in MB/s; zero means unlimited. Restart the node after changing
configuration. `zarrswarm seed /path/data.zarr` registers a store immediately
and remembers it across restarts.

The node checks metadata every minute by default and performs a full stat-cached
rescan every tenth check. Appended samples and in-place revisions are then
announced. `cache_max_gb = 200` limits downloaded cached data; it does not limit
the original stores you seed.

Seeding decodes chunks to build identities. Scanning uses one thread per
available core, up to 16, unless `ZT_SCAN_WORKERS` overrides the count.
The node releases unused scan heap memory with `malloc_trim` on glibc.
Unchanged file identities reuse `~/.zt/scan`. `ZT_HASH_CACHE=DIR` shares
hashes between nodes on one host, using file identity and metadata.
File changes that preserve size and modification time are outside this
stat-based cache model.

Corrupt chunks fail receiver validation and are excluded from serving.
Reseed a changed store to refresh it immediately.

## Store repair data

```bash
zarrswarm parity 'zt://<grid>' t2m --k 8 --drop
```

A volunteer can retain a Reed–Solomon parity row instead of a complete dataset.
A row contains one encoded shard per stripe; any `k` compatible data and distinct
parity shards recover that stripe. Actual stored size depends on representation
and compression. See the [protocol](https://github.com/dsuhoi/zarrswarm/blob/main/PROTOCOL.md).

## Monitor nodes

```bash
zarrswarm status
zarrswarm peers 'zt://<grid>'
zarrswarm-tui
journalctl --user -u zt-node -f
```

## Keys and trust

`~/.zt/node.key` is the node's Ed25519 identity. Initialize a distinct identity
on each host. DHT records, manifests and pushdown receipts use its signatures.

Configure `trust = ["<pubkey>"]` to give selected publishers precedence over
ordinary holder votes. A signature authenticates an announcement; it does not
establish the scientific truth of a measurement.

Private networks authenticate each request with a timestamped HMAC. The
network key is not sent in the header, but payloads are unencrypted HTTP.
Synchronize clocks: a difference greater than 120 seconds causes authentication
failures. Use a VPN if you need channel confidentiality.

Manifest versions and previous roots are saved in `~/.zt/heads.json`.
Clients reject observed rollbacks and blacklist holders that sign different
roots at one version. `/api/status` exposes `fraud` and `blacklist`.
The blacklist is local and held in memory until restart.
`/probe` checks the requesting address rather than an arbitrary host.

## Troubleshooting

| Symptom | Check |
|---|---|
| `contacts=0` | Bootstrap reachability, `http://HOST:7881/whoami` and the data-port firewall |
| `ro=True` on a public host | External port reachability; open the port and restart |
| Slow downloads | Holder bandwidth and shared relay capacity in `zarrswarm peers LINK` |
| `cannot reseed ...` | A configured seed path disappeared; other stores continue serving |
| Port 7881 or 7882 is busy | Initialize with `--port N`; use `ZT_CTL=http://127.0.0.1:<N+1>` for that node |

The short `zt` and `zt-tui` command aliases, `zt-node` service name,
`ZT_*` settings and `~/.zt` state directory are supported for existing nodes.
