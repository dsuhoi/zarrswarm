"""Same GFS run through NOAA GRIB and Unidata's operational NetCDF service.

Small, predeclared regional sample; original payloads and coordinates are retained.
Run with the existing research environment: python bench/gfs_pipelines.py WORK --out RESULT
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import eccodes
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zarr_torrent import codec

FIELDS = (("TMP", "2 m above ground", "Temperature_height_above_ground", 2, "K"),
          ("UGRD", "10 m above ground", "u-component_of_wind_height_above_ground", 10, "m/s"),
          ("VGRD", "10 m above ground", "v-component_of_wind_height_above_ground", 10, "m/s"),
          ("PRMSL", "mean sea level", "Pressure_reduced_to_MSL_msl", None, "Pa"))
RUN = datetime(2026, 10, 5, tzinfo=timezone.utc)
NOAA = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261005/00/atmos/gfs.t00z.pgrb2.0p25."
UCAR = ("https://tds.scigw.unidata.ucar.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p25deg/"
        "GFS_Global_0p25deg_20261005_0000.grib2")


def fetch(url, path, byte_range=None):
    headers = {} if byte_range is None else {"Range": f"bytes={byte_range[0]}-{byte_range[1]-1}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
        if byte_range:
            assert response.status == 206, "Refuse a whole global GRIB download"
            assert response.headers["Content-Range"].startswith(f"bytes {byte_range[0]}-{byte_range[1]-1}/")
        body = response.read(8 * 1024 * 1024 + 1)
        assert len(body) <= 8 * 1024 * 1024
        if byte_range:
            assert len(body) == byte_range[1] - byte_range[0]
        path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "estimator": codec.ESTIMATOR,
              "protocol": {"run": RUN.isoformat(), "leads_hours": [0, 6], "fields": FIELDS,
                           "region": {"north": 50, "south": 34.25, "west": 0, "east": 31.75},
                           "tiles": [32, 32], "transforms": "NOAA ecCodes float64 output converted to float32; NetCDF native float32; coordinate selection; no scale/offset arithmetic.",
                           "negative_control": "First cell of each tile + one quantum from original GRIB E,D.",
                           "unit_alias": "ecCodes m s**-1 and Unidata m/s denote the same unit; values unchanged.",
                           "max_download_bytes_per_payload": 8 * 1024 * 1024},
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in ("bench/gfs_pipelines.py", "zarr_torrent/codec.py")},
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "xarray", "eccodes", "h5netcdf")},
              "fields": [], "work": str(args.work.resolve())}
    args.out.write_text(json.dumps(result, indent=2) + "\n")  # before fetching any field
    for lead in result["protocol"]["leads_hours"]:
        source = NOAA + f"f{lead:03d}"
        index_path = args.work / f"f{lead:03d}.idx"
        index_hash = fetch(source + ".idx", index_path)
        lines = index_path.read_text().splitlines()
        for short, level, name, height, unit in FIELDS:
            tag = f"{short}_f{lead:03d}"
            hits = [i for i, line in enumerate(lines) if f":{short}:{level}:" in line]
            assert len(hits) == 1
            i = hits[0]
            start, end = int(lines[i].split(":")[1]), int(lines[i + 1].split(":")[1])
            gp, nc = args.work / f"{tag}.grib2", args.work / f"{tag}.nc"
            gh = fetch(source, gp, (start, end))
            valid = (RUN + timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M:%SZ")
            params = dict(result["protocol"]["region"], var=name, time=valid, accept="netcdf4", addLatLon="true")
            if height is not None:
                params["vertCoord"] = height
            url = UCAR + "?" + urllib.parse.urlencode(params)
            nh = fetch(url, nc)
            with gp.open("rb") as fh:
                h = eccodes.codes_grib_new_from_file(fh)
                try:
                    assert eccodes.codes_get(h, "dataDate") == 20261005
                    assert eccodes.codes_get(h, "dataTime") == 0 and eccodes.codes_get(h, "endStep") == lead
                    assert eccodes.codes_get(h, "units") == ("m s**-1" if unit == "m/s" else unit)
                    shape = (eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni"))
                    values = eccodes.codes_get_values(h).reshape(shape).astype("f4")
                    lats = eccodes.codes_get_array(h, "latitudes").reshape(shape)[:, 0]
                    lons = eccodes.codes_get_array(h, "longitudes").reshape(shape)[0]
                    E, D = (eccodes.codes_get(h, k) for k in ("binaryScaleFactor", "decimalScaleFactor"))
                    step = float(2.0 ** E * 10.0 ** -D)
                finally:
                    eccodes.codes_release(h)
            with xr.open_dataset(nc, engine="h5netcdf") as ds:
                field = ds[name].squeeze(drop=True)
                assert field.dims == ("latitude", "longitude") and field.dtype == np.dtype("f4")
                assert field.attrs["units"] == unit
                assert any(np.datetime64(valid[:-1]) == np.asarray(ds[c].values).reshape(-1)[0]
                           for c in ds.coords if c.startswith("time"))
                ys = [np.flatnonzero(lats == v).item() for v in field.latitude.values]
                xs = [np.flatnonzero(lons == v).item() for v in field.longitude.values]
                a, b = values[np.ix_(ys, xs)], field.values
                assert a.shape == b.shape == (64, 128) and np.isfinite(a).all() and np.isfinite(b).all()
                np.savez(args.work / f"{tag}.npz", noaa=a, unidata=b, latitude=field.latitude.values,
                         longitude=field.longitude.values)
            matches = changes = 0
            for y in range(0, 64, 32):
                for x in range(0, 128, 32):
                    ta, tb = a[y:y + 32, x:x + 32], b[y:y + 32, x:x + 32]
                    changed = ta.copy()
                    changed[0, 0] += step
                    assert changed[0, 0] != ta[0, 0]
                    va = codec.vcid_of(ta)
                    matches += codec.same_vcid(va, codec.vcid_of(tb))
                    changes += codec.same_vcid(va, codec.vcid_of(changed))
            row = {"short_name": short, "lead_hours": lead, "units": unit, "source_quantum": step,
                   "shape": list(a.shape), "tiles": 8, "matching_tiles": matches,
                   "accepted_source_count_changes": changes,
                   "bit_equal_cells": int(np.count_nonzero(a.view("u4") == b.view("u4"))),
                   "cells": a.size, "max_abs_error": float(np.max(np.abs(a.astype("f8") - b))),
                   "source_urls": {"NOAA": source, "Unidata": url},
                   "payload_sha256": {"GRIB": gh, "NetCDF": nh, "index": index_hash}}
            result["fields"].append(row)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(tag, row["bit_equal_cells"], matches, changes, flush=True)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
