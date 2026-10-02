"""Motivation study: public copies of the same ERA5 field (0.25 deg, 721x1440) on GCS.

For every copy: dims order, chunk shape, codec, time step, format. For one common instant: are the decoded values of
2m_temperature identical (value identity, our vcid) and are the stored chunk bytes identical (byte identity: what
IPFS/BitTorrent/HTTP mirrors can deduplicate)? Output: JSON + a table.

python bench/replica_survey.py [--time 2010-03-03T06] [--out bench/replica_survey.json]
"""
import argparse
import hashlib
import json
import urllib.request

import numpy as np
import xarray as xr
import zarr

GCS = "https://storage.googleapis.com"
COPIES = [f"{GCS}/gcp-public-data-arco-era5/ar/{p}" for p in (
    "full_37-1h-0p25deg-chunk-1.zarr-v3", "1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
    "1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2", "1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2",
    "1959-2022-6h-1440x721.zarr", "1959-2023_01_10-full_37-1h-1440x721.zarr")] + \
    [f"{GCS}/weatherbench2/datasets/era5/{p}" for p in (
        "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", "1959-2023_01_10-wb13-6h-1440x721.zarr",
        "1959-2022-6h-1440x721.zarr", "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")]
VAR = "2m_temperature"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="2010-03-03T06")
    ap.add_argument("--out", default="bench/replica_survey.json")
    a = ap.parse_args()
    t = np.datetime64(a.time)
    rows = []
    for url in COPIES:
        name = url.split(GCS + "/")[1]
        try:
            ds = xr.open_zarr(url, consolidated=True, chunks=None)
            arr = zarr.open_array(f"{url}/{VAR}", mode="r")
        except Exception as e:
            rows.append({"copy": name, "error": repr(e)[:160]})
            print(name, "ERROR", repr(e)[:120], flush=True)
            continue
        v = ds[VAR]
        times = ds.time.values
        dt_h = float((times[1] - times[0]) / np.timedelta64(1, "h"))
        r = {"copy": name, "format": arr.metadata.zarr_format, "dims": list(v.dims), "shape": list(v.shape),
             "chunks": list(arr.chunks), "codec": str(arr.compressors if hasattr(arr, "compressors") else "")[:80],
             "dtype": str(arr.dtype), "dt_h": dt_h, "t0": str(times[0])[:13], "t1": str(times[-1])[:13],
             "lat0": float(ds.latitude[0]), "nvars": len(ds.data_vars)}
        i = int(np.searchsorted(times, t))
        if i < len(times) and times[i] == t:
            sel = [slice(None)] * v.ndim
            sel[v.dims.index("time")] = i
            vals = np.asarray(arr[tuple(sel)])
            if v.dims.index("latitude") > v.dims.index("longitude"):
                vals = vals.T  # canonical (lat, lon) for value comparison
            if r["lat0"] < 0:
                vals = vals[::-1]
            r["value_sha"] = hashlib.sha256(np.ascontiguousarray(vals, dtype="<f4").tobytes()).hexdigest()[:16]
            ct = arr.chunks[v.dims.index("time")]
            idx = [0] * v.ndim
            idx[v.dims.index("time")] = i // ct
            raw = urllib.request.urlopen(f"{url}/{VAR}/{arr.metadata.encode_chunk_key(tuple(idx))}", timeout=120).read()
            r["chunk_bytes"], r["chunk_sha"] = len(raw), hashlib.sha256(raw).hexdigest()[:16]
        rows.append(r)
        print(json.dumps(r), flush=True)
    ok = [r for r in rows if "value_sha" in r]
    vals = {r["value_sha"] for r in ok}
    chunks = {r["chunk_sha"] for r in ok}
    layouts = {(tuple(r["dims"]), tuple(r["chunks"]), r["codec"]) for r in ok}
    summary = {"time": a.time, "copies_with_instant": len(ok), "distinct_values": len(vals),
               "distinct_chunk_bytes": len(chunks), "distinct_layouts": len(layouts),
               "distinct_time_steps_h": sorted({r["dt_h"] for r in ok})}
    print(json.dumps(summary))
    json.dump({"rows": rows, "summary": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
