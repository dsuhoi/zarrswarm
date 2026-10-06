"""Verify the swarm union of the real ERA5 (WeatherBench2) replicas against the oracle slice.
python bench/verify_wb2.py LINK --ctl URL --oracle ~/.cache/zs_real/wb2/wb2_oracle.zarr"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zarrswarm as zs

COVER = {"2m_temperature": ("2000-01-01", "2004-12-31T18"), "10m_u_component_of_wind": ("2000-01-01", "2002-12-31T18"),
         "geopotential": ("2000-01-01", "2002-12-31T18"), "mean_sea_level_pressure": ("2002-01-01", "2004-12-31T18")}
ap = argparse.ArgumentParser(); ap.add_argument("link"); ap.add_argument("--ctl"); ap.add_argument("--oracle")
a = ap.parse_args()
t = time.time()
ds = zs.open_dataset(a.link, ctl=a.ctl)
ora = xr.open_zarr(Path(a.oracle).expanduser(), consolidated=False)
out = {"open_s": round(time.time() - t, 2), "vars": sorted(ds.data_vars)}
for v, (lo, hi) in COVER.items():
    t = time.time()
    got = ds[v].load()
    ref = ora[v].sel(time=slice(lo, hi)).transpose(*got.dims)
    inside = got.sel(time=slice(lo, hi))
    outside = got.where((got.time < np.datetime64(lo)) | (got.time > np.datetime64(hi)), drop=True)
    out[v] = {"shape": list(got.shape), "load_s": round(time.time() - t, 2),
              "equal_in_cover": bool(np.array_equal(inside.values, ref.values)),
              "nan_outside": bool(np.isnan(outside.values).all()) if outside.size else True}
print(json.dumps(out, indent=1))
