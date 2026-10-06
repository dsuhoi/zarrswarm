# Reproduce the experiments

Drivers in `bench/` and `sim/` write JSON records. Run them from the repository root with the checkout's
Python environment. Always set a new output path: many defaults point to existing research records.
Large provider datasets live outside the checkout and are fetched separately.

The current package uses `zarrswarm`. Frozen archives retain the original paths and hashes of the code that
was actually measured. Running today's driver does not recreate that earlier source version.

## Results used by the manuscript

The [result inventory](https://github.com/dsuhoi/zarrswarm/blob/main/bench/revalidation/README.md) distinguishes
completed runs, diagnostic records and measured source snapshots.

| Question | Result files under `bench/revalidation/` |
|---|---|
| Do independent provider values pool? | `provider_identity_controls_v5.json`, `provider_merge_v5.json`, `decoder_survey_v5.json`, `goes_decoders_v5.json` |
| Where does fitting accept a changed source count? | `identity_boundary_v9.json`, `gfs_packing_v9.json`; see [source contracts](source-contracts.md) |
| Does the source contract survive transfer and cache restart? | `packing_network_v9_final.json`, `integration_checks_v9.json` |
| Does joint layout/holder selection improve query time? | `process_receiver_fixed_v5.json`, `paired_stats_receiver_fixed_v5.json` |
| Does the comparison hold with kernel TCP and closed holders? | `external_nat_bound_relay_v5.json`, `external_nat_bound_relay_stats_v5.json`, `external_nat_bound_relay_checks_v5.json` |
| How does Internet access compare with a public bucket and mirrors? | `multisite_v5_final.json` |
| What does a small exhaustive planner oracle show? | `planner_cover_oracle_v5.json`, `assignment_controls_v5.json` |
| What happens on disjoint seasonal fields? | `provider_seasonal_v5.json`, `provider_seasonal_stats_v5.json` |
| Can a cache follow a changing public feed? | `live_mrms_dry_v5.json`, `live_mrms_wet_fixed_v5.json`, `live_mrms_checks_v5.json` |
| What are the CPU and metadata costs? | `edge_v5.json`, `metadata_v5.json`, `identifier_size_v5.json` |

Network timings precede the source-contract implementation. Their workloads preserve identical decoded
values across all encodings, so exact-value identity can also pool those copies. Layout-selection benefits
and tolerance to independent decoders are separate findings. The source-contract controls establish their
own narrower mechanism and local-transfer results; they are not a new WAN timing experiment.

## Start with local checks

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python -m zarrswarm.plan
.venv/bin/python -m zarrswarm.jlps
.venv/bin/python -m zarrswarm.parity
```

The tests use generated data and loopback connections. They require no station infrastructure or large
provider download. Their success checks functionality, not forecast quality or Internet throughput.

Research drivers need dependencies beyond the minimal package. For example, provider/decoder drivers use
Requests, NetCDF4, ecCodes, cfgrib or gribberish; satellite traces use Skyfield; the OpenNeuro example uses
NiBabel. Install the dependencies for the chosen experiment into the same environment. `uv sync --inexact`
retains separately installed research packages while syncing the locked application dependencies.

## Provider and decoder controls

| Driver | Input or purpose | Retained original output |
|---|---|---|
| `bench/replica_survey.py` | Nine public ERA5 copies | `bench/replica_survey.json` |
| `bench/lattice_probe.py` | ARCO/NCAR, four variables at three instants | `bench/lattice_probe.json` |
| `bench/real_provider_merge.py WORKDIR` | Stage and merge provider fields locally | `bench/real_provider_merge.json` |
| `bench/lattice_controls.py WORKDIR` | Positive, changed-value and multi-slice controls | `bench/lattice_controls.json` |
| `bench/provider_identity_controls.py` | Current estimator on the original provider fields | `bench/revalidation/provider_identity_controls_v5.json` |
| `bench/decoder_survey.py GRIB_DIR` | MRMS, GFS and ERA5 decoding | `bench/decoder_survey.json` |
| `bench/goes_decoders.py DIR` | GOES-16 ABI L1b decoder/precision controls | `bench/goes_decoders.json` |
| `bench/fmri_demo.py WORKDIR` | OpenNeuro ds000102 integer-data example | `bench/fmri_demo.json` |
| `bench/identity_boundary.py` | Sparse source-count changes and supplied-quantizer controls | Versioned `identity_boundary_*.json` |
| `bench/packing_network.py` | Signed contracts through independent local node processes | Versioned `packing_network_*.json` |

Consult each driver's `--help` before a new run. Provider controls require the retained raw fields or a
new download with recorded hashes. Decoder precision and quantizer provenance affect identity results;
do not silently convert all inputs to a common dtype and then claim native-decoder agreement.

## Prepare heterogeneous replicas

```bash
.venv/bin/python bench/make_variants.py ~/zarrswarm-data/variants-month --start 2020-01-01 --days 31
```

This creates five storage variants of the same ARCO ERA5 2 m temperature values: hourly maps, six-hourly
maps, point-oriented time chunks and tiled layouts, with different codecs and Zarr versions. This fixture
measures layout and scheduling behavior. It does not introduce independently measured provider values.

Use the retained `bench/sat_traces_meteor.json` for the placement behind the original results. The trace
generator uses current orbital inputs; rerunning it is a new trace, not an exact reconstruction of the old one:

```bash
.venv/bin/python bench/sat_traces.py --days 31 --stations 40 --sats METEOR-M2 --out bench/sat-traces-new.json
.venv/bin/python bench/avail_mc.py --traces bench/sat-traces-new.json --out bench/availability-new.json
```

The availability calculation is Monte Carlo under declared independent failures or longitude-band outages.
It is separate from running a network of real ground stations.

## Three execution environments

### Processes with application shaping

`sim/procswarm.py` starts one real `python -m zarrswarm.cli node` process per peer, with independent sockets
on loopback. A node token bucket sets upload rate; `ZT_EMU_LATENCY_MS` injects delay in the application.
This exercises the implementation but does not reproduce kernel packet queues and losses.

### Kernel network emulation

`sim/netemu.py` uses user/network namespaces, veth pairs, a bridge and `tc netem`/`tbf`. It applies delay,
loss and bandwidth shaping on both veth ends, so TCP queues and retransmissions come from the kernel.
The holder advertises its rate but does not apply a second application-level shaper.

The host needs unprivileged user namespaces and the `veth`, `sch_netem` and `sch_tbf` modules. An administrator
can load those modules once. Then check the environment:

```bash
.venv/bin/python sim/netemu.py selftest
```

This self-check transfers generated data between two nodes at 1 MB/s and 50 ms delay. Do not treat it as a
completed benchmark matrix.

### Internet sites

`sim/multisite.py` reads a TOML configuration such as `sim/multisite.toml`. Replace the example SSH hosts,
interpreters, source paths, data paths and query ranges with your own. The controller starts a bootstrap/relay,
remote holders and a fresh client for each query. Reverse SSH tunnels make the relay reachable from remote
sites; node sessions last for the run. A new private network key is generated per run.

The historical configuration names the actual sites used by that experiment. It is not a ready-to-run public
service. Check ownership, allocated storage and ongoing work before starting another run on a shared host.

| Phase | Comparison |
|---|---|
| `cloud` | xarray reads directly from the public ARCO bucket |
| `http` | A catalogue client divides native chunk files across HTTP mirrors; eight requests per mirror |
| `mirror` | One holder with the public layout |
| `swarm` | All holders, with JLPS and minimum-byte selection |
| `bytes` | The best completed byte-identical swarm, selected with prior knowledge |

```bash
.venv/bin/python sim/multisite.py sim/multisite.toml --phases cloud,http,mirror,swarm,bytes --reps 3 --out sim/multisite-new.json
.venv/bin/python bench/summarize_multisite.py sim/multisite-new.json
```

The byte baseline is an oracle over byte swarms, not a blind discovery policy. Historical first-contact
probing ablations remain separate from the completed measured matrix.

## Heterogeneous-query driver

```text
python sim/e_hetero.py VARIANTS_DIR [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--procs` | Off | Separate processes with application shaping |
| `--emu` | Off | Kernel namespaces and link shaping; with neither flag, nodes share one process |
| `--placement` | `random` | `sat` assigns trace-covered hours to the first `--stations` holders |
| `--peers` | 48 | Total nodes, including bootstrap/relays |
| `--boot`, `--nat` | 3, 0.3 | Bootstrap/relay count and fraction of NAT holders |
| `--rep-start`, `--reps` | 0, 3 | Repetition range `range(rep_start, reps)`; `--reps` is the end index |
| `--modes` | `values,bytes` | Value identity and byte identity |
| `--covers` | `jlps` | Add `bytes` to compare minimum-byte layout selection |
| `--queries` | `map_day_1h,series_point_1h,period_6h` | A day of maps, a point series, and the full period at six-hour steps |
| `--rate-mbps` | 4.0 | Median holder upload rate, in MB/s |
| `--traces`, `--stations` | `bench/sat_traces.json`, 24 | Trace input and number of station holders |
| `--out` | `sim/results_ehet.json` | JSON output; explicitly choose a new path |

The repetition index fixes the random placement. Use the same index and fixture when comparing execution
environments. The work directory is `ZT_SIM_ROOT`, defaulting to `~/.cache/zt_sim`; `ZT_KEEP_LOGS=1` retains
node logs. Each completed row is written as the run progresses.

```bash
variant_dir=~/zarrswarm-data/variants-month
.venv/bin/python sim/e_hetero.py "$variant_dir" --procs --placement sat --traces bench/sat_traces_meteor.json --peers 36 --stations 22 --rate-mbps 1 --covers jlps,bytes --rep-start 0 --reps 10 --out sim/heterogeneous-new.json
.venv/bin/python bench/summarize_ehet.py sim/heterogeneous-new.json --csv bench/heterogeneous-new.csv --stats bench/heterogeneous-new-stats.json
```

For a holder-count sweep, run the same fixture at 12, 24, 36 and 48 peers with 7, 14, 22 and 29 stations,
respectively. For the original kernel comparison, repeat the same placements with `--emu` and use
`bench/compare_emulators.py PROCESS_JSON KERNEL_JSON --reps N --out NEW_JSON`. Later receiver/relay fixes
have their own frozen matrices; the older kernel run is historical evidence, not the current comparison.

## Summaries and acceptance checks

`bench/summarize_ehet.py` reports medians, ranges, transferred bytes and coverage for complete exact answers.
Its `--stats` output uses paired time ratios, a 10,000-resample bootstrap with seed 0, and records the paired
repetition indices. `bench/summarize_multisite.py` reports source-reference errors as well as time and volume.
`bench/summarize_sweep.py` writes the holder-count CSV to `bench/speed_vs_peers.csv`.

Accept a network timing row only with `state=done`, full unrounded requested-sample coverage and a downloaded
payload check equal to the reference. Run that check after timing. Reading missing truth through xarray can
conceal an incomplete transfer and is not a valid completion check. Direct WAN controls require zero maximum
absolute error for these lossless workload fixtures.

Before the first complete byte answer, the oracle uses a 1800-second deadline. Once it has a complete answer,
later candidates get at least 120 seconds or three times the best completed time. A quick partial answer
must not set the deadline for a slower complete one.

The final process replay contains ten placements with 90 complete exact answers. The completed kernel/NAT
comparison contains 27 complete exact answers across three placements. WAN records contain 36 node answers
and 18 cloud/HTTP controls. Their source versions differ and are identified by their SHA manifests; do not
combine their rows under a claim that one final implementation was measured everywhere.

A new run should record input hashes, source hashes, configuration, random seeds, completion checks and host
conditions. Keep failed or interrupted rows distinguishable from accepted results. Remove task-owned clients
and temporary caches after verifying results, without removing shared datasets or other jobs.
