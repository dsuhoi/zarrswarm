"""Four real-world storage variants of the SAME values (ERA5 2m_temperature, 0.25 deg, from ARCO-ERA5):

  V1 arco-1h   hourly,  chunks (1, 721, 1440),  Blosc-lz4, zarr v2  (the public ARCO/WB2 layout)
  V2 wb2-6h    6-hourly, chunks (1, 721, 1440), Blosc-lz4, zarr v2  (WeatherBench2 6 h copy: identical bytes/step)
  V3 ts-1h     hourly,  chunks (168, 91, 180),  zstd,      zarr v3  (rechunked for time series, e.g. Rechunker)
  V4 tile-6h   6-hourly, chunks (4, 361, 720),  gzip,      zarr v3  (a regional user's re-encoding)
  V5 tiles-1h  hourly,  chunks (1, 361, 720),  zstd,      zarr v3  (a ground station's own ingest software)

python bench/make_variants.py OUTDIR [--start 2020-01-01 --days 31]
"""
import argparse
import time
from pathlib import Path

import numcodecs
import xarray as xr
from zarr.codecs import GzipCodec, ZstdCodec

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VAR = "2m_temperature"
VARIANTS = {
    "V1-arco-1h": dict(step=1, chunks=(1, 721, 1440), fmt=2, codec="lz4"),
    "V2-wb2-6h": dict(step=6, chunks=(1, 721, 1440), fmt=2, codec="lz4"),
    "V3-ts-1h": dict(step=1, chunks=(168, 91, 180), fmt=3, codec="zstd"),
    "V4-tile-6h": dict(step=6, chunks=(4, 361, 720), fmt=3, codec="gzip"),
    "V5-tiles-1h": dict(step=1, chunks=(1, 361, 720), fmt=3, codec="zstd"),
}


def clean(ds):
    for v in ds.variables.values():
        for k in ("chunks", "preferred_chunks", "compressor", "compressors", "filters", "serializer"):
            v.encoding.pop(k, None)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--days", type=int, default=31)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    end = (xr.date_range(a.start, periods=2, freq=f"{a.days}D")[1] - xr.coding.times.pd.Timedelta("1h")).isoformat()
    src = clean(xr.open_zarr(ARCO, consolidated=True, chunks=None)[[VAR]].sel(time=slice(a.start, end)).load())
    print(f"fetched {src.sizes} in {time.time() - t:.0f}s", flush=True)
    for name, v in VARIANTS.items():
        p = out / f"{name}.zarr"
        if p.exists():
            continue
        ds = src.isel(time=slice(None, None, v["step"]))
        if v["fmt"] == 2:
            comp = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
            enc = {VAR: {"chunks": v["chunks"], "compressor": comp}}
        else:
            enc = {VAR: {"chunks": v["chunks"], "compressors": [ZstdCodec(level=3) if v["codec"] == "zstd"
                                                                 else GzipCodec(level=5)]}}
        clean(ds).to_zarr(p, encoding=enc, consolidated=False, zarr_format=v["fmt"])
        print(f"{name}: {dict(ds.sizes)} -> {p}", flush=True)


if __name__ == "__main__":
    main()
