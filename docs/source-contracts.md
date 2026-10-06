# Verify source codes

A source contract supplies the quantization parameters needed to compare original
integer codes across floating-point decoding pipelines. ZarrSwarm carries it
through scanning, signed manifests, transfer validation, local caching and repair.

Without source parameters, independently fitting a lattice is ambiguous on
sparse values. If the source step is 1 but the observed values are only 0 and 16,
a fit may infer step 16. Changing one 0 to 1 can preserve the fitted relative
codes. The research controls contain 13 falsely accepted source-code changes in
20 such cases; these are verification failures, not successful matches.

## Contract assumptions

The publisher supplies `step`, `origin` and `max_error`, with
`0 < max_error < step / 2`. Under a valid bound on the actual decoder error,
the source code is uniquely recoverable. Receivers also reject values whose
residual exceeds the declared error budget.

Passing that residual check does not establish that the declared step or error
bound is correct. The publisher is the trust anchor for those parameters.
The contract authenticates packing metadata, not the scientific truth of the
field.

## Register a source and its mirrors

Create `packing.json` as a mapping from variable names to
`[step, origin, max_error]`. Values use the array's units. For time-varying
packing, use a catalogue indexed by integer UTC seconds:

```json
{
  "t2m": [1.0, 0.0, 0.4]
}
```

This is an illustrative contract. Use parameters from your actual source
format; do not infer them from this example.

```json
{
  "t2m": {
    "time": {
      "1577836800": [1.0, 0.0, 0.4],
      "1577840400": [0.5, 0.0, 0.2]
    }
  }
}
```

Inspect a store, then seed it through the publisher's running node:

```bash
zarrswarm scan DATA.zarr --packing packing.json
zarrswarm seed DATA.zarr --packing packing.json --packing-out signed.json
```

A mirror seeds its own compatible store with `--packing signed.json`.
Mirrors and receivers configure the original publisher's public key through
`zarrswarm node --trust KEY` or `trust` in the node configuration.
A mirror forwards the original signature instead of signing a replacement
contract under its own key.

Parameters are bound to the field name, grid and units. An unsigned, modified
or absent contract does not enter the selected source-code family. Float32 and
float64 encodings can belong to the same source-code family.

## Validation and caching

Receivers verify both the downloaded values and values after local re-encoding.
A representation that changes source codes is rejected before caching.
Scan-cache keys include metadata; changing a contract invalidates old identities
even if chunk bytes stay the same.

Cached copies retain the signed contract and remain ordinary holders.
Changing the trusted-key policy invalidates the merged view. For growing chunks,
the node checks the existing prefix's source codes. Physical padding at time and
spatial edges is excluded from observations.

Reed–Solomon repair can encode absolute source codes. Reconstructed and saved
values pass the same checks as transferred values.

## Recorded controls

| Check | Independent fitting | Source contract |
|---|---:|---:|
| Falsely accepted source-code changes, 20 cases | 13 | 0 |
| Valid copies within 0.4 source steps, 20 cases | Not tested here | 20 |
| Valid copies in a separate 36-case matrix | Not tested here | 36 |
| Accepted changes of one source count, 72 cases | Not tested here | 0 |
| Matched native GFS pairs, 64 tiles | 63 | 64 |
| Accepted changes of one true GRIB step, 64 tiles | 0 | 0 |

The numerical records are
[identity_boundary_v9.json](https://github.com/dsuhoi/zarrswarm/blob/main/bench/revalidation/identity_boundary_v9.json)
and [gfs_packing_v9.json](https://github.com/dsuhoi/zarrswarm/blob/main/bench/revalidation/gfs_packing_v9.json).

The [process experiment](https://github.com/dsuhoi/zarrswarm/blob/main/bench/revalidation/packing_network_v9_final.json)
used six independent processes on one machine and eight saved GFS fields.
All 64 compatible pairs pooled, while all 64 one-count changes stayed separate.
After the publisher stopped, an alternative layout supplied all 65,536 source
codes correctly. Restarting the receiver with remote holders stopped recovered
the same answer from its signed cache.

[The regression tests](https://github.com/dsuhoi/zarrswarm/blob/main/tests/test_packing.py)
also cover repair, altered signatures and budgets, missing contracts,
re-encoding corruption, prefix growth and scan-cache invalidation.
These controls establish local protocol behavior; they do not measure WAN
speedups, ARM throughput or independent station deployment.

## Remaining limits

Plain Zarr without explicit packing metadata still uses independent fitting.
Mirrors must forward the complete signed time catalogue; altering it creates a
different family. A stat-based scan cache cannot detect edits that preserve
file size and modification time.

Coverage is selected by time chunk: a contested spatial tile can exclude that
layout's entire time chunk. The implementation does not combine arbitrary
incomplete spatial covers. Source-contract verification has not remeasured
the earlier Internet speed results.
