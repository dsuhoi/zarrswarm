# Configuration reference

`zarrswarm init` creates `~/.zt/config.toml`. Set `--home DIR` or `ZT_HOME`
to use another node directory. Command-line flags override file values.

A legacy `config.json` is read when TOML is absent; initialization migrates it.
Repeated initialization preserves `ctl_port`, `[[seed]]`, `[tuning]` and core
settings, but rewrites other custom keys and comments.

The directory also contains `node.key`, the Ed25519 node identity, and
`state.json`, which records manually registered stores and published names.
Initialize a separate identity on each host.

## Main settings

| Key | Default | Node flag or environment | Meaning |
|---|---|---|---|
| `network` | Unset | None | Invitation retained for reference |
| `network_key` | `""` | `ZT_NETWORK_KEY` | Private-network key; keep it secret |
| `port` | 7881 | `--port` | TCP data port used by peers |
| `host` | `127.0.0.1` | `--host` | Data-port bind address; `0.0.0.0` accepts external connections |
| `ctl_port` | `port + 1` | `--ctl-port` | Local control API for CLI, TUI and xarray |
| `public` | `""` | `--public` | Public data URL; empty means NAT or automatic discovery |
| `bootstrap` | `[]` | `--bootstrap` | Public entry-node URLs |
| `relay` | `""` | `--relay` | Relay used by a node without incoming connections |
| `relay_server` | `false` | `--relay-server` | Relay traffic for NAT nodes |
| `auto` | `false` | `--auto` | Check reachability and choose public address or relay |
| `upload_mbps` | 0 | `--upload-mbps` | Upload limit in MB/s; zero is unlimited |
| `cache_max_gb` | 0 | None | Download-cache limit in GB; zero is unlimited |
| `trust` | `[]` | `--trust` | Publisher public keys with precedence over holder votes |

Cache eviction runs every 30 seconds and reduces excess use to 90% of the limit.
It preserves your seeded files, parity and chunks used within the previous
ten minutes.

## Seed stores at startup

```toml
[[seed]]
path = "/data/era5/*.zarr"

[[seed]]
path = "/scratch/wb2_64x32.zarr"
```

Missing glob matches are logged and skipped. Stores added with `zarrswarm seed`
are also restored from `state.json`. Remove a startup entry when unseeding its
path, or the next startup will seed it again.

## Tuning settings

All keys are optional. The node maps them to environment variables before
importing its runtime modules. Unknown tuning keys fail startup.

| Key | Environment variable | Default | Meaning |
|---|---|---|---|
| `announce_every_s` | `ZT_ANNOUNCE_EVERY` | 600 | DHT reannouncement interval |
| `page_cache_mb` | `ZT_PAGE_CACHE_MB` | 256 | Manifest-page cache |
| `decoded_cache_mb` | `ZT_DECODED_MB` | 256 | Holder's decoded pushdown cache |
| `client_cache_mb` | `ZT_DECODED_CACHE_MB` | 512 | xarray decoded-chunk cache |
| `audit_rate` | `ZT_AUDIT` | 0.05 | Sampled whole-chunk pushdown audit rate |
| `pushdown_frac` | `ZT_PUSHDOWN_FRAC` | 0.25 | Request a slice below this source-chunk fraction |
| `readahead_chunks` | `ZT_READAHEAD` | 4 | Upcoming view chunks prefetched during sequential reads |
| `stall_min_s` | `ZT_STALL_MIN` | 5 | Seconds without data before a batch is considered stalled |
| `xt1_below_bps` | `ZT_XT1_BELOW` | 3e7 | Request xt1 below this link rate in bytes/s |
| `relay_frame_kb` | `ZT_RELAY_FRAME_KB` | 4096 | Relay data-frame size |
| `rescan_every_s` | `ZT_RESCAN_EVERY` | 60 | Metadata check interval; every tenth check is a full stat-cached rescan |
| `follow_every_s` | `ZT_FOLLOW_EVERY` | 300 | Subscription update interval |

`client_cache_mb`, `pushdown_frac` and `readahead_chunks` apply in the Python
client process. Set their environment variables there; the node's configuration
does not configure a separate client.

## Environment variables

The `ZT_*` names are the protocol's existing runtime settings. To inspect
current references, run `rg -o 'ZT_[A-Z0-9_]+' zarrswarm sim bench`.
Here, node means the server process, client means CLI/xarray, and harness means
a script under `sim/`.

| Variable | Process | Default | Meaning |
|---|---|---|---|
| `ZT_HOME` | CLI, TUI | `~/.zt` | Node state directory |
| `ZT_CTL` | Client | `http://127.0.0.1:7882` | Local control API |
| `ZT_CTL_HOST` | Node | `127.0.0.1` | Control bind address; needed by isolated testbed nodes |
| `ZT_NETWORK_KEY` | Node | Unset | Private-network key; secret |
| `ZT_TRUST` | Node | Unset | Comma-separated trusted publisher keys |
| `ZT_ANNOUNCE_EVERY` | Node | 600 | DHT reannouncement interval in seconds |
| `ZT_ANNOUNCE_MBPS` | Node | 0 | Bandwidth hint in MB/s without local shaping; explicit upload limit takes precedence |
| `ZT_RESCAN_EVERY` | Node | 60 | Metadata check interval in seconds |
| `ZT_FOLLOW_EVERY` | Node | 300 | Follow update interval in seconds |
| `ZT_PAGE_CACHE_MB` | Node | 256 | CAPT page-cache size |
| `ZT_DECODED_MB` | Node | 256 | Decoded pushdown cache size |
| `ZT_AUDIT` | Node | 0.05 | Sampled whole-chunk audit rate |
| `ZT_PD_WHOLE_FRAC` | Node | 0.25 | Fetch the whole chunk when grouped pushdown demand exceeds this fraction |
| `ZT_STALL_MIN` | Node | 5 | Seconds without data before declaring a stalled batch |
| `ZT_XT1_BELOW` | Node | 3e7 | xt1 threshold in bytes/s |
| `ZT_RELAY_FRAME_KB` | Node | 4096 | Relay frame size |
| `ZT_COVER_DEADLINE` | Node | 300 | Regional cover-selection deadline in seconds |
| `ZT_JLPS_SLACK` | Node | 0.1 | Predicted makespan slack; choose fewer bytes among covers within it |
| `ZT_CLIENT_MBPS` | Node | 30 | Modeled receiver download and verification capacity in MB/s; zero disables this bound |
| `ZT_CHUNK_OVERHEAD_MS` | Node | 50 | Modeled per-chunk request, disk and verification overhead |
| `ZT_SCAN_WORKERS` | Node | 0 | Scan threads; zero uses available cores, capped at 16 |
| `ZT_HASH_CACHE` | Node | Unset | Shared stat-based scan hash cache, also keyed by layout, metadata and identity version |
| `ZT_VALUE_ID` | Node, client | `lattice` | Floating-point identity: independent lattice fitting or `exact` hashing |
| `ZT_IDENTITY` | Node | Unset | `bytes` separates encoding-specific swarms for the byte-identity control |
| `ZT_DECODED_CACHE_MB` | Client | 512 | xarray decoded-chunk cache size |
| `ZT_PUSHDOWN_FRAC` | Client | 0.25 | Slice-request threshold |
| `ZT_READAHEAD` | Client | 4 | Upcoming virtual chunks prefetched |
| `ZT_EMU_LATENCY_MS` | Emulated node | 0 | Application-added response delay in milliseconds |
| `ZT_EMU_LOSS` | Emulated node | 0 | Probability of intentionally hanging a data-port request |
| `ZT_EMU_STRATEGY` | Emulated node | `maxflow` | Peer strategy: `maxflow`, `rarest`, `random`, `single` |
| `ZT_EMU_ROUTER` | Harness | Unset | Internal marker for the network-namespace router |
| `ZT_KEEP_LOGS` | Harness | Unset | Retain stopped swarm directories and node logs |
| `ZT_SIM_ROOT` | Harness | `~/.cache/zt_sim` | Heterogeneous testbed work directory |
| `ZT_SIM_DIR` | Harness | `~/.cache/zt_sim` | Simulator work directory |

Cache sizes with an `_MB` suffix are in MB; intervals above are in seconds.
`ZT_EMU_*` options are testbed controls and are not read from `config.toml`.
See the [experiment guide](experiments.md).
