"""Real ERA5 (WeatherBench2, 6-hourly, 64x32) -> oracle slice + two heterogeneous replicas.

R1: zarr v3, 2000-2002, t2m + u10 + geopotential@500/850, source chunks (100, 64, 32)
R2: zarr v2, 2002-2004, t2m + msl, chunks (120, 32, 16)
python bench/make_wb2_replicas.py OUTDIR
"""
import sys
from pathlib import Path

import xarray as xr

URL = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
VARS = ["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure", "geopotential"]


def main():
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    src = xr.open_zarr(URL, consolidated=True)[VARS].sel(time=slice("2000-01-01", "2004-12-31"))
    src = src.sel(level=[500, 850])
    oracle = out / "wb2_oracle.zarr"
    if not oracle.exists():
        ds = src.load()
        for v in ds.variables.values():
            v.encoding.clear()
        ds.to_zarr(oracle, zarr_format=3, consolidated=False)
    ds = xr.open_zarr(oracle, consolidated=False)
    r1 = ds[["2m_temperature", "10m_u_component_of_wind", "geopotential"]].sel(time=slice("2000-01-01", "2002-12-31")).load()
    enc1 = {v: {"chunks": (100, 64, 32) if r1[v].ndim == 3 else (100, 2, 64, 32)} for v in r1.data_vars}
    for v in r1.variables.values():
        v.encoding.clear()
    r1.to_zarr(out / "wb2_R1_v3.zarr", zarr_format=3, encoding=enc1, consolidated=False, mode="w")
    r2 = ds[["2m_temperature", "mean_sea_level_pressure"]].sel(time=slice("2002-01-01", "2004-12-31")).load()
    for v in r2.variables.values():
        v.encoding.clear()
    r2.to_zarr(out / "wb2_R2_v2.zarr", zarr_format=2, encoding={v: {"chunks": (120, 32, 16)} for v in r2.data_vars},
               consolidated=False, mode="w")
    print({"oracle": dict(ds.sizes), "R1": dict(r1.sizes), "R2": dict(r2.sizes)})


if __name__ == "__main__":
    main()
