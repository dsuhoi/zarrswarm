# Measured results and source snapshots

This directory retains scientific controls, completed network runs and the executable source versions used
for those runs. The current manuscript is [the IEEE Access edition](../../paper/ieee_access/README.md), titled
**Verified Peer-to-Peer Sharing of Gridded Sensor Data Across Encodings**.

Use the [experiment guide](../../docs/experiments.md) for commands and execution environments, and the
[source-contract guide](../../docs/source-contracts.md) for the current packing API. Result values and frozen
archives retain their original hashes. Archive module paths describe their measured version; the current
implementation lives in `zarrswarm/`.

A version suffix identifies a recorded stage, not a promise that every file with that suffix uses the same
source. Each matrix and its SHA manifest establish the version actually measured. Historical or incomplete
records are retained for diagnosis and excluded from the completed comparisons described below.

## Source-code identity controls

| Evidence | Scope and result |
|---|---|
| `identity_boundary_v9.json` | Fitting accepts 13 of 20 changed sparse fields. Supplied source quantizers accept no changes, match all 20 valid decodings, and pass a separate 36-positive/72-negative matrix. |
| `gfs_packing_v9.json` | All 64 eligible native-decoder tile pairs match; all 64 true source-step changes are rejected. |
| `packing_network_v9_final.json` | Six independent loopback processes transfer signed per-time contracts across dtypes/layouts. Another layout supplies all 65,536 codes after publisher loss; cache restart preserves the contract. |
| `integration_protocol_v9.json`, `integration_checks_v9.json` | Declared controls, input/source hashes and completed integration checks. |
| `source_integration_v9.tgz`, `source_sha256_integration_v9.json` | Frozen 40-file implementation/test snapshot, with the recorded 108-test run. |

Source parameters are supplied by an operator and authenticated by a trusted publisher. No automatic recovery
of GRIB packing provenance from arbitrary Zarr, new WAN speedup, or independent station deployment is claimed.
`identity_boundary_v7.json`, `identity_boundary_v8.json`, `gfs_packing_v8.json` and `packing_checks_v8.json`
retain the earlier boundary and numeric-API stages separately.

## Provider and decoder evidence

`lattice_controls_v5.json` contains 1,536 single-slice positive cases, five negative controls and 1,088
multi-slice positive/opposing-shift pairs. `provider_identity_controls_v5.json` applies the working estimator
to four original survey variables at three January instants: 768 tile pairs, changed fitted-count controls,
and common zero-origin/ARCO-phase quantizers. The common-quantizer baseline receives shared parameters;
the fitting estimator infers its own.

`provider_merge_v5.json` records five local provider/trust cases with unchanged ARCO/NCAR decoded values
staged in hourly Zarr. This is not a live NetCDF-serving deployment. `decoder_survey_v5.json` and
`goes_decoders_v5.json` preserve independent-decoder and precision controls.

`provider_seasonal_protocol_v5.json` declares four seasonal 2024 dates before acquisition.
`provider_seasonal_v5.json` and `provider_seasonal_stats_v5.json` retain 16 fields and 1,024 tile pairs:
896 matches and no accepted changes by one fitted count. The 128 separated temperature pairs infer different
steps. A common-NCAR-quantizer diagnosis matches all 256 temperature pairs but accepts seven changed tiles;
`provider_seasonal_diagnostic_v5.py` and its JSON preserve this post-hoc analysis. It is not a prespecified
holdout baseline. `original_bit_audit_v5.json` checks the original float32 fields independently.

`gfs_pipelines_v7.json` uses eight pinned fields and an explicit float64-to-float32 conversion, producing
65,536 agreeing values. `gfs_native_precision_v7.json` reuses the same raw payloads without that conversion:
79.04–96.22% numeric differences, 63 of 64 fitting matches and no accepted true-GRIB-step changes. The unmatched
pressure pair is in `gfs_native_split_v7.json`; the estimator was not retuned. Raw/derived inputs are in
`gfs_pipelines_data_v7.tgz`, with entry hashes in `gfs_pipelines_checks_v7.json`. Native and common-precision
source archives and manifests remain distinct.

## Planner controls

`planner_cover_oracle_v5.json` enumerates complete covers and integral holder assignments on 240 small fixed
graphs at nine parameter settings. Seeds 0–119 were used during development; seeds 120–239 check the corrected
solver without later tuning. The default reaches 112 model optima on the held-out graphs; its maximum ratio
to the oracle is 1.2405. These are model costs, not observed network times.

`assignment_controls_v5.json` checks 600 fixed-cover instances. Its maximum schedule/optimum ratio is 1.25;
one instance exceeds 1.2 after order normalization. It does not establish global optimality of joint layout
and holder selection. `planner_example_v7.json` is a code-checked illustration of the cost model.

## Completed process replay

`process_receiver_fixed_v5.json` contains 90 selected complete exact answers across placements 0–9.
`process_receiver_fixed_primary_v5.json` and `process_receiver_fixed_tail_v5.json` retain the 63- and 27-row
batches. Configuration and all 16 core-module hashes agree. `process_receiver_fixed_checks_v5.json` verifies
sample counts, completion, downloaded-payload reference checks and the 22-file frozen source archive.

`paired_stats_receiver_fixed_v5.json` records paired ratios and 10,000-resample bootstrap intervals with
seed 0. Ratios against byte identity are 1.95, 2.35 and 1.29 for maps, point series and six-hour sampling;
ratios against minimum-byte selection are 1.87, 0.99 and 1.30. Both point-series intervals include one.

`process_batch_protocol_v5.json`, `process_batch_split_v5.json` and `process_batch_orchestration_v5.tgz`
record the predetermined batch split and overlapping execution. Background load was not controlled; these
are ten placements on one shared host. The postflight record confirms cleanup of task-owned processes and
scratch. `process_fibonacci_v5.json` is the earlier matrix and is not substituted for this final replay.

## Kernel TCP, closed holders and external transport

`external_nat_bound_relay_v5.json` contains 27 complete exact answers across three kernel-emulated placements.
Every answer carries relay traffic; the placements have 7, 5 and 13 closed holders. Native minimum-byte and
aria2 use matching chunk keys and payload volumes. `external_nat_bound_relay_checks_v5.json` verifies catalogue
hashes, frozen sources, preflight byte identities, port restrictions and matched inputs.

`external_nat_bound_relay_stats_v5.json` reports times, ranges, volumes and paired ratios.
`nat-bound-catalogues-v5.tgz` retains the three catalogues/preflights and a separate smoke test.
`external_nat_bound_relay_source_v5.tgz` and its SHA manifest freeze the measured implementation.
`receiver_nat_postflight_v5.json` records task-owned process/cache cleanup.

The older `kernel_v5.json` has two fully recorded placements and one partial byte point answer, with
`kernel_stats_v5.json` summarizing complete answers. The unfinished third placement remains diagnostic;
its rows do not enter the current kernel comparison. Later work corrected a loopback-only relay address,
so the old timings are historical, not measurements of the corrected path.

`external_aria2_cloud_v5.json` retains a separate 27-query comparison on full public copies.
`external_aria2_stats_v5.json` summarizes it; `external_aria2_point_replay_v5.json` holds three separate
diagnostic answers. The primary and replay archives preserve different drivers. A later fixed-latency
series, `external_latency_fixed_v5.json`, contains 24 complete exact reads on one placement; its eight native
point reads take 8.09–8.29 seconds. Repeated reads on one placement are not independent deployments.

## Internet access comparison

`multisite_v5_final.json` contains 36 node answers and 18 public-cloud/HTTP controls. Only explicitly identified
cloud/HTTP rows were reused from earlier partial acquisitions. `source_wan_v5.tgz`,
`source_sha256_v5_measured.json` and `deployed_source_snapshots_v5.json` identify the measured source on all
three sites. Process, kernel and WAN matrices use different frozen revisions.

All five network workload encodings preserve identical decoded values. Exact-value identity can therefore
pool them too. The workload establishes benefits from heterogeneous layouts and holder selection; the
independent-provider controls separately test tolerant value comparison.

## Live-feed and cache correctness

`live_mrms_dry_v5.json` and `live_mrms_wet_fixed_v5.json` record two completed public-feed runs, each with three
future two-minute fields. All six updates are complete and numerically exact on received payloads. The wet
crop was selected from a precursor before collecting future updates. Its three updates change 162,139,
124,402 and 9,986 cells; cache lag is 103.58–117.65 seconds with scan/follow intervals of 60 seconds.
Default follow is 300 seconds.

The scope is three node instances in one process on one shared host over loopback. It does not measure
station throughput or independent-provider decoding. `live_mrms_checks_v5.json` verifies crop/input hashes,
native time steps, changes and completion. Successful and prior diagnostic sources are separate archives.
`source_valid_prefix_cache_v5.tgz` and its SHA manifest include the later finite-padding guard; the earlier
live-feed timings were not rerun after that guard.

## CPU, metadata and exact transport

`edge_v5.json` and `identifier_size_v5.json` measure decode/identity CPU cost and identifier length on one
pinned Xeon Platinum 8358 CPU, with eight chunks per layout and three repetitions after warmup.
`metadata_v5.json` measures registration and synthetic exact-ID manifests. These metadata sizes do not
characterize long temporal L3 identifiers.

`source_byte_exact_codecs_v5.tgz` and `source_sha256_v5_byte_exact_codecs.json` preserve the exact-roundtrip
stage. Parity rejects numeric-only round trips that change signed zero or NaN payloads; temporal transport
normalizes big-endian input to its little-endian wire dtype. Later source-contract and cache stages retain
their own tests and hashes.

## Interpreting the records

A completed timing answer requires full requested-sample coverage and downloaded payload equality to the
reference. Reference checks run after timing and do not fetch missing samples through xarray. Fresh clients
avoid query-cache reuse; probed rates and catalogue discovery can still differ. A quick partial byte answer
does not define the deadline for a complete byte baseline.

`final_results_v11.json` freezes the manuscript's scientific records by SHA-256. It also describes local
submission artifacts that are outside this public source tree. The current package migration updates module
paths and documentation; it is not a rerun of those frozen network measurements. Large provider datasets and
raw build logs are excluded from the repository.
