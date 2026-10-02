"""Deterministic ERA5-like replica generator (1 deg global, hourly). Values depend only on absolute time,
so independently generated replicas of overlapping periods are value-identical (like real mirrors).

python bench/make_era5like.py OUT.zarr --start 2020-01-01 --hours 1440 --vars t2m,u10 --chunks 24,181,360
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

LAT = np.linspace(90, -90, 181)
LON = np.arange(0, 360, 1.0)


def field(var: str, times: pd.DatetimeIndex) -> np.ndarray:
    h = times.values.astype("datetime64[h]").astype("int64").astype("float64")[:, None, None]
    lat = np.deg2rad(LAT)[None, :, None]
    lon = np.deg2rad(LON)[None, None, :]
    doy = 2 * np.pi * h / (24 * 365.25)
    diurnal = 2 * np.pi * h / 24 + lon
    if var == "t2m":
        f = 288 - 40 * np.sin(lat) ** 2 - 12 * np.sin(lat) * np.cos(doy) + 4 * np.cos(lat) * np.cos(diurnal)
    elif var == "u10":
        f = 8 * np.cos(3 * lat) * np.sin(lon * 2 + h / 97) + 3 * np.sin(doy)
    elif var == "v10":
        f = 6 * np.sin(2 * lat) * np.cos(lon * 3 - h / 71)
    else:
        f = np.sin(lat * 5 + h / 50) * np.cos(lon * 4)
    return f.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--hours", type=int, default=1440)
    ap.add_argument("--vars", default="t2m")
    ap.add_argument("--chunks", default="24,181,360")
    ap.add_argument("--block", type=int, default=240)
    a = ap.parse_args()
    times = pd.date_range(a.start, periods=a.hours, freq="h")
    ch = tuple(int(x) for x in a.chunks.split(","))
    names = a.vars.split(",")
    enc = {v: {"chunks": ch} for v in names}
    for i in range(0, a.hours, a.block):  # append in blocks to keep memory small
        t = times[i:i + a.block]
        ds = xr.Dataset({v: (("time", "lat", "lon"), field(v, t)) for v in names},
                        coords={"time": t, "lat": LAT, "lon": LON})
        if i == 0:
            ds.to_zarr(a.out, mode="w", encoding=enc | {"time": {"units": f"hours since {a.start}", "dtype": "int64"}},
                       consolidated=False)
        else:
            ds.to_zarr(a.out, append_dim="time", consolidated=False)
        print(f"{a.out}: {i + len(t)}/{a.hours}", flush=True)


if __name__ == "__main__":
    main()
