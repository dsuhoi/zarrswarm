# Revalidation: L3 and signed source quantizers (2026-10-05)

## Current v11: title revision

The title is **Verified Peer-to-Peer Sharing of Gridded Sensor Data Across Encodings**.
`paper_checks_v11.json` records the 12-core / 16-total PDF, all-page visual QA,
unchanged 25 other manuscript files and unchanged 40-file tested implementation.
`final_results_v11.json` and `author_bundle_checks_v11.json` record the updated author bundle.
No new tests or measurements were run. `paper_checks_v5.json` points to v11;
the frozen v10 bundle and versioned evidence remain unchanged.

## Previous v10: final manuscript and frozen results

`paper_checks_v10.json` records the final 12-core / 16-total PDF, 129-word two-paragraph conclusion,
all-page visual QA, unchanged measured table numbers and unchanged 40-file tested source.
`final_results_v10.json` freezes the result files and measured archives by SHA-256.
`paper_checks_v5.json` is the current compatibility alias; `paper_checks_v5_before_final_v10.json`
preserves v9. `manuscript_before_final_v10.tgz` preserves the preceding paper.
The implementation, original controls and 108-test result remain v9; no new measurements were made here.
[Final report and author review bundle](../../paper/FINALIZATION.md).
The sections below describe earlier revisions.

## Current v9: source codes across the protocol

- `integration_protocol_v9.json`: controls and seven pre-change hashes, recorded before implementation.
- `source_before_integration_v9.tgz`: pre-change files; `source_integration_v9.tgz` contains the current
  40-file snapshot and both protocols. `source_sha256_integration_v9.json` records 108 passing tests.
- `identity_boundary_v9.json`: unchanged 13/20 legacy false accepts; supplied source parameters give
  zero false accepts, 20 valid matches, and a separate 36-positive / 72-negative matrix.
- `gfs_packing_v9.json`: 64 native pairs match and all 64 true-quantum changes are rejected.
- `packing_network_v9_final.json`: six independent local processes; signed per-time contracts pool
  native dtypes/layouts, another layout serves all 65,536 source codes after publisher loss, and cache restart works.
- `pytest_integration_v9.txt`: 108 passed, 44 library warnings, 146.55 seconds.
- `integration_checks_v9.json`: source/archive hashes, controls and retained failed attempts.
- `paper_checks_v9.json`: current 12-core / 16-total PDF, stock ACM style, embedded fonts and visual QA;
  `paper_checks_v5.json` points to this revision. `paper_checks_v5_before_integration_v9.json` preserves v8.

The operator supplies the source parameters; no automatic GRIB-provenance recovery from arbitrary Zarr is claimed.
See [implementation and limits](../../docs/SOURCE_CONTRACT_INTEGRATION_V9.md).
The remaining sections record earlier measured versions; their timings were not rerun for v9.

This directory records mechanism controls, source snapshots and network replays separately. The historical experiment filename `sim/results_ehet_v5.json` does not denote estimator lattice-v5.

Mechanism and resource evidence:

- `lattice_controls_v5.json`: 1536 single-slice positives, five negative controls, and 1088 multi-slice positive/opposing-shift pairs.
- `provider_identity_controls_v5.json`: working L3 estimator on all four original survey variables and three January instants (768 tile pairs), single-count changes, and common zero-origin / ARCO-phase quantizers. Common phase and step succeed too; this baseline receives shared parameters that L3 estimates independently.
- `planner_cover_oracle_v5.json`: 240 three-peer, two-layout fixed graphs and nine parameter settings, enumerating all minimal complete covers and integral assignments. Seeds 0–119 exposed the catalogue-order bug; seeds 120–239 check the corrected solver without subsequent changes. Default on held-out graphs: 112 model optima, maximum ratio 1.2405. These are cost-model results, not network timings.
- `provider_merge_v5.json`: five local provider/trust cases. Unchanged ARCO/NCAR decoded values are staged in hourly Zarr; this is not a live NetCDF-serving deployment.
- `decoder_survey_v5.json`, `goes_decoders_v5.json`: independent decoder, precision and single-count controls.
- `assignment_controls_v5.json`: existing exhaustive test on 600 fixed-cover instances, maximum schedule/optimum 1.25; one instance exceeds 1.2 after order normalization. This is not global optimality of JLPS.
- `edge_v5.json`, `identifier_size_v5.json`: actual-time-axis CPU cost and identifier lengths, one pinned Xeon Platinum 8358 CPU, eight chunks per layout, three repetitions after warmup.
- `metadata_v5.json`, `capt_v5.txt`: registration and synthetic exact-ID manifest checks. Those sizes do not characterize long temporal L3 identifiers.
- `pytest_v5.txt`: 80 passing regression tests in 308.25 seconds, 44 Zarr API/runpy warnings.
- `paper_checks_v5.json`: compatibility alias of the latest PDF QA; versioned `paper_checks_v*.json` and `paper_checks_v5_before_*.json` preserve earlier checks.
- `storage_v5.json`: scoped task-owned disk/process audit, including cloud postflight after restored access. Fibonacci owns 190.01 MiB including the new provider fields/dependencies; cloud owns 185.45 MiB, the VPS 1.83 MiB. Existing hardlink blocks are separate; no task-owned benchmark jobs remain active.

Source versions:

- `source_sha256_v5_measured.json`: earlier L3 snapshot used by the WAN experiment; deployed unchanged source hashes were verified on all three sites.
- `source_sha256_v5_before_planner_fix.json`: bounded catalogue refresh and complete-answer oracle deadline, before the cached-byte bound/candidate corrections.
- `source_sha256_v5_planner_measured.json`: frozen source of the earlier Fibonacci and kernel replays, with corrected JLPS bounds, remote prices and the minimum-byte candidate.
- `source_sha256_v5_final.json`: snapshot before the receiver/relay fixes below. After freezing the earlier network source, the above-400-signature fallback was corrected to report relay/receiver/local costs and assignment rounding was made independent of catalogue order. Regression and fixed-graph controls check these changes; earlier network timings retain their measured versions.
- `source_sha256_v5_before_order_fix.json`, `assignment_controls_v5_before_order_fix.json`, `pytest_v5_before_order_fix.txt`: prior current-code checks preserved before the new order fix.
- `source_wan_v5.tgz`, `source_network_replay_v5.tgz`: frozen executable source archives; all 16 core-file hashes were verified against the corresponding snapshots. They preserve the measured versions after later fixes.
- `deployed_source_snapshots_v5.json`: read-only hashes of deployed source directories. Every network matrix also records its core-module hashes at startup.

Historical `process_fibonacci_v5.json` contains 90 unique complete exact answers. `kernel_v5.json` contains the first two fully recorded placements: 17 complete exact answers and one partial byte point answer (740 of 744 samples). `kernel_stats_v5.json` records the full-answer summaries; `kernel_v5_observed_24.json` preserves the earlier download. After restoring cloud access, `kernel_v5_observed_25.json` confirms the first two placements are unchanged and the third remains incomplete (25 of 27 rows). The queued process replay did not run, and no task-owned node or driver remains active. No results from the unfinished third placement enter the paper. The old cloud-process matrix is archived as `process_cloud_v5_before_oracle_fix.json`. Accepted timing answers require full requested sample coverage and downloaded payloads equal to the reference. Local `/api/read` checks run after timing and never fetch missing reference data through xarray. Matched placements use independent fresh clients; catalogues and probed rates can differ. All five network workload encodings preserve identical decoded values, so exact-value identity can also pool them; the tolerant-comparison mechanism is tested separately by the independent-provider controls.

`*_before_oracle_fix.json`, `*_before_planner_fix.json`, intermediate parts, `_partial.json` and v4 files preserve historical or interrupted evidence and do not enter the final replay comparison. The old kernel source was stopped before its last expensive query; its 26 diagnostic rows are retained. WAN data are in `multisite_v5_final.json`: 36 node answers and 18 independent cloud/HTTP controls, with source filenames recorded. Only explicitly extracted cloud/HTTP controls from earlier partial runs are reused.

See [the correctness report](../../docs/CORRECTNESS_REVALIDATION.md) for commands, scientific limits and final run counts. Dataset files remain outside the checkout; remote windows use existing files and hard links. Original remote source checkouts and other jobs are unchanged.

## Historical external transport comparison on full public copies, 05.10.2026

`external_aria2_cloud_v5.json`: 27 full/exact queries on Cloud.ru; actual aria2 against native JLPS and minimum-byte on identical full catalogues. `external_aria2_stats_v5.json`: medians, ranges, paired ratios. A 217.32-second point answer is retained. `external_aria2_point_replay_v5.json`: three separate diagnostic answers, never pooled with the main matrix. `external_aria2_catalogues_v5.tgz`: three main catalogues plus replay catalogue. The primary and replay source archives/SHA files preserve their distinct drivers; all core modules are unchanged. Successful smoke is `external_aria2_smoke_v5.json`; lifecycle/disk audit is `external_aria2_postflight_v5.json`. Protocol and interpretation: `docs/EXTERNAL_COMPARISON.md`.

## Receiver latency and partial-copy NAT comparison

`external_latency_before_v5.json` retains ten full answers from an interrupted diagnostic series, including a 207.609-second replay. The next query exited with SIGSEGV; its stack is `latency-before-crash-stack.txt` and its cause is not separately established. `external_latency_fixed_v5.json` contains 24 full/exact answers over four cycles of two workloads on one placement. Eight native point reads take 8.09–8.29 seconds. These are repeated reads, not independent deployments. `external_latency_stats_v5.json` and `latency_provenance_checks_v5.json` verify the statistics and matching normalized catalogue/placement/rate/encoding data. Catalogues are in `latency-catalogues-v5.tgz`.

The diagnostic and fixed source archives preserve their own drivers and receiver versions. `source_sha256_v5_receiver_bound_relay.json` records the receiver/relay source, including the shared completion-wait and bound-address relay fixes. `latency_regression_before_v5.txt` records two expected failures on the old path; `pytest_receiver_bound_relay_v5.txt` records 82 passing tests on that source.

`external_nat_bound_relay_source_v5.tgz` freezes the kernel partial-copy comparison with actual aria2, blocked private inbound ports, controller-only administrative access, relay preflight and per-query relay counters. Its three-client smoke is `external_nat_bound_relay_smoke_v5.json`. `process_receiver_fixed_source_v5.tgz` freezes the new ten-placement Fibonacci ablations; it precedes the bound-address fix, which leaves the loopback-bound process testbed's relay route unchanged. All archives have SHA-256 manifests. Protocol, interpretation and final result inventory: [LATENCY_AND_NAT_REVALIDATION.md](../../docs/LATENCY_AND_NAT_REVALIDATION.md).

`external_nat_bound_relay_v5.json` contains the completed 27-query kernel comparison. All answers are full/exact and all carry relay traffic. Closed-holder counts are 7, 5 and 13. `external_nat_bound_relay_stats_v5.json` contains medians, ranges, payload volumes and paired ratios; `external_nat_bound_relay_checks_v5.json` verifies each catalogue hash, source hashes, preflight CID/size, receiver port restrictions and matching native-min/aria2 keys/bytes. `nat-bound-catalogues-v5.tgz` contains the three main catalogues and preflights plus the separate smoke. `receiver_nat_postflight_v5.json` records no task-owned active processes or receiver/node/partial scratch. The prior 502 preflight failure is `nat_bound_relay_before_preflight_failure_v5.log`; it is not included in the matrix. The earlier kernel source's hardcoded loopback prevented private-holder relay reads; its old timings remain historical and no longer enter the paper's kernel table.

## Completed receiver-fixed process replay

`process_receiver_fixed_v5.json` replaces the paper's earlier process matrix: 90 unique selected full/exact answers across placements 0–9. `process_receiver_fixed_primary_v5.json` and `process_receiver_fixed_tail_v5.json` preserve raw batches of 63 and 27 rows; the cutoff copy and two run logs remain separate. All common parameters and 16 module hashes agree. `process_receiver_fixed_checks_v5.json` verifies every configuration, requested sample count, full coverage, payload-reference check and the 22-file frozen archive. `paired_stats_receiver_fixed_v5.json` contains 10000-resample paired bootstrap intervals with seed 0; the existing summarizer generates Figure 5's CSV. Ratios against bytes are 1.95 / 2.35 / 1.29; against min-bytes 1.87 / 0.99 / 1.30. Both point intervals include one.

`process_batch_protocol_v5.json`, `process_batch_split_v5.json` and `process_batch_orchestration_v5.tgz` record the predetermined 0–6 / 7–9 split, overlapping batches and the answer-count-only cutoff. Background load was not controlled; these are placements on one shared host. `process_receiver_postflight_v5.json` verifies all 22 remote source files per batch, no owned active processes or temporary homes/partial fixtures, and 6971392 allocated bytes in the two new roots. Dataset and shared hash cache were retained. The old `process_fibonacci_v5.json` remains historical.


## Byte-exact codecs and disjoint seasonal provider test

`source_sha256_v5_byte_exact_codecs.json` and `source_byte_exact_codecs_v5.tgz` freeze the 32 source files of that stage. Parity conversion now rejects numeric-only round trips that alter signed zero or NaN payloads; XT1 normalizes big-endian input to its little-endian wire dtype. `exact_roundtrip_before_v5.txt` records two expected failures; `transport_byte_order_before_v5.txt` records four expected failures. Their after files pass. `pytest_byte_exact_codecs_v5.txt`: 92 passed, 45 library warnings, 129.31 seconds.

`provider_seasonal_protocol_v5.json` declares four seasonal 2024 dates before fetching; `provider_seasonal_source_v5.tgz` freezes the measured implementation before the later transport byte-order fix. Fitting and L3 identity are unchanged. `provider_seasonal_v5.json`, `provider_seasonal_stats_v5.json` and the log retain all 16 fields / 1024 pairs: 896 matches, zero accepted single-count changes. Shared ARCO phase/step gives the same counts. The 128 separated temperature pairs have different fitted steps. `provider_seasonal_diagnostic_v5.py` and JSON reproduce the post-hoc temperature diagnosis; a common NCAR quantizer matches all 256 temperature pairs but accepts seven changed tiles. It is not a prespecified holdout baseline or estimator tuning. `original_bit_audit_v5.json` checks actual float32 bits on all 12 original cached survey fields; original reported equality fractions are unchanged.

`provider_seasonal_postflight_v5.json` verifies no task-owned active processes, all 32 source hashes and cached field hashes (in the stats), and a new 127.84 MiB remote root. Cached fields remain for reproducibility. Existing data/services/jobs are unchanged. Detailed commands and interpretation: [CODECS_AND_PROVIDER_HOLDOUT.md](../../docs/CODECS_AND_PROVIDER_HOLDOUT.md).


## Live MRMS ingestion and current cache correctness

`live_mrms_dry_v5.json` and `live_mrms_wet_fixed_v5.json` contain two completed public-feed runs, each with three future two-minute fields. All six updates are complete and numerically exact on received payloads. The wet crop was chosen from a precursor before collecting future updates: `wet_selection_v5.json`, `mrms_wet_selection_v5.py`. Its three updates change 162139 / 124402 / 9986 cells. Cache lags span 103.58–117.65 s with scan/follow both 60 s; default follow is 300 s. `live_mrms_checks_v5.json` verifies source/crop hashes, native time steps, completion, changes and postflight. Scope: three instances in one process on one shared host over loopback, not station throughput or independent-provider decoding.

`live_mrms_wet_before_v5.json`, `live_mrms_wet_v5.log`, `wet_offline_diagnostic_v5.json` and `wet_receiver_diagnostic_v5.json` preserve the failed wet attempt: rewritten source chunks reused stale decoded entries, leaving three spatial tiles outdated. The common decoded cache now keys by CID plus decoder metadata. Failed rows are not pooled with successful measurements. Dry and successful-wet logs are `live_mrms_v5.log` and `live_mrms_wet_fixed_v5.log`.

Frozen measured versions: `source_live_ingest_v5.tgz` (34-file dry), `source_live_ingest_wet_v5.tgz` (35-file wet failure), `source_decoded_content_cache_v5.tgz` (35-file successful wet). Their corresponding SHA manifests remain separate. `source_valid_prefix_cache_v5.tgz` and `source_sha256_v5_valid_prefix_cache.json` freeze the current 35-file source after the final finite-padding guard; all hashes match current files and archive entries. MRMS timings retain their measured pre-guard source, with NaN unknown-time padding. `source_live_ingest_smoke_initial_v5.tgz` preserves the interrupted initial smoke.

Before/after logs cover first-poll detection, growth/revision/local authority, decoded transport and hyperslab freshness, decoder-metadata separation and finite integer padding. `pytest_valid_prefix_cache_v5.txt`: 98 passed, 44 library warnings, 128.04 seconds. Previous 92/95/97-test results remain historical. `paper_checks_v5.json` records the final PDF, 12 core / 14 total pages, all 14 visual checks, embedded fonts and current source hashes; `paper_checks_v5_before_live_ingest.json` preserves the preceding PDF checks. The new MRMS root retains 156.18 MiB of unique file blocks with no task-owned active processes; this is separate from older roots. [Protocol, fixes and interpretation](../../docs/LIVE_INGEST_REVALIDATION.md).


## Argument revision v6

`latex_argument_v6.txt` records the successful final build. `paper_checks_v5.json` now records the revised PDF (12 core / 14 total pages), all 14 visual checks, 24 manuscript source/data hashes, unchanged 35-file tested source and Table 2(b) complement percentages derived from `decoder_survey_v5.json`. `paper_checks_v5_before_argument_v6.json` preserves the preceding PDF checks. Live ingestion is now Results subsection 3.2; CPU throughput is explicitly decode plus lattice identity. Measured experiment artifacts and the 98-test run retain their original versions. [Editorial changes](../../paper/REVISION_NOTES.md).


## Research review v7 (current paper)

`identity_boundary_v7.json`: 20 fixed cases with known source quantum 1; 13 changed sparse cases match lattice identity, zero exact matches, and all four dense cases reject the change. Earlier provider controls change the **fitted** field step; their raw labels/results are preserved, not interpreted as a universal source-count guarantee.

`gfs_pipelines_v7.json`: eight pinned GFS fields, 65,536 values, common float32 comparison. NOAA ecCodes outputs are converted from float64; the service returns float32. Every value agrees. The protocol phrase coordinate selection only omits that explicit target-precision conversion, which is clarified in `gfs_pipelines_checks_v7.json` and `gfs_native_precision_checks_v7.json`. `gfs_native_precision_v7.json` reuses the exact same raw payloads without conversion: 79.04–96.22% numeric differences, 63 of 64 pairs match, zero true-GRIB-quantum changes accepted. The remaining pressure pair and its fitted steps are in `gfs_native_split_v7.json`; the estimator was not retuned. `gfs_pipelines_v7_before_units_guard.json` preserves the first unit-notation failure and is not pooled.

`gfs_pipelines_data_v7.tgz` retains 26 raw/derived files, with entry hashes in `gfs_pipelines_checks_v7.json`. Six-file common-precision and native-analysis source archives/manifests are separate (`review_controls_source_v7.*`, `review_native_controls_source_v7.*`). `planner_example_v7.json` is a code-checked cost-model illustration, not network timing. `pytest_parity_review_v7.txt`: five existing integration tests passed in 43.16 seconds. The complete 98-test evidence remains `pytest_valid_prefix_cache_v5.txt`; all 35 current tested-source hashes are unchanged.

Current paper QA: `paper_checks_v7.json` and legacy `paper_checks_v5.json`; previous v6 preserved as `paper_checks_v5_before_research_review_v7.json`. The paper has 12 core / 15 total pages, five tables, six figures, and Appendix A contains detailed fitting/planning. `latex_review_v7.txt` records the build. [Scientific findings and manuscript changes](../../paper/RESEARCH_REVIEW_V7.md).

Acquisition driver follow-up: only the `transforms` protocol label in `bench/gfs_pipelines.py` now explicitly states the float64-to-float32 conversion. Numerical code/results are unchanged; the measured pre-label-correction source remains in `review_controls_source_v7.tgz`. Old/current driver hashes and the distinction are recorded in `gfs_pipelines_checks_v7.json`.


## Source-packing API v8

- `packing_protocol_v8.json`: fixed before implementation/runs; common source step/origin and 0.45-step budget, separate synthetic transfer matrix and the same retained GFS sample.
- `identity_boundary_v8.json`: old value-only baseline still accepts 13/20 changes; strict metadata-based API accepts none, matches all 20 valid decodings, and passes 36 positives / 72 negatives in the transfer matrix.
- `gfs_packing_v8.json`: 64 eligible and matching native tile pairs, 0 accepted true-quantum changes. The previous value-only 63/64 statistics and payload hashes are unchanged.
- `pytest_packing_v8.txt`: 100 passed, 44 library warnings, 131.22 s; command `rtk proxy .venv/bin/python -m pytest -q tests`. The initial collection failure is retained in `pytest_packing_v8_collection_error.txt` and excluded from executed-test results.
- `source_before_packing_v8.tgz` / `source_before_packing_sha256_v8.json`: four pre-change sources. Temporary raw copies were moved outside test discovery, to `/tmp/zt_source_before_packing_v8`.
- `source_packing_v8.tgz` / `source_sha256_packing_v8.json`: current frozen source; `packing_checks_v8.json` verifies protocol, inputs, baseline continuity and results.
- `paper_checks_v8.json`: current PDF/layout/source QA; `paper_checks_v5.json` remains the compatibility alias for latest QA. `paper_checks_v7.json` is the preserved previous snapshot.

This is a numeric API with caller-supplied source metadata; automatic Zarr ingest does not propagate or authenticate it. No new remote allocation, network speedup or independent station deployment is claimed. [Mechanism and limitations](../../docs/PACKING_IDENTITY_V8.md).
