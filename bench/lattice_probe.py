"""Probe: quantization-lattice identity of independently decoded copies.

Data are born quantized (ADC counts, GRIB packing R + k*2^E, NetCDF scale/offset). Two decoders of the same codes
produce floats that differ by less than half a step. Each copy estimates its lattice from its OWN values: the
coarsest step s = 2^e whose codes k = round((v - phi)/s) keep every distinct value distinct (injective), phi the
circular mean phase. Identity = the integer codes.

Checks: ERA5 from Google ARCO (Zarr) vs NCAR ds633.0 on AWS (NetCDF) - independent copies of the same GRIB codes -
per full field and per chunk-sized tile (tile lattice estimated from the tile alone, and from the whole field);
negative control: two 1.5 deg regriddings of ERA5 (different data, differences 1e-4 K) must NOT merge.

python bench/lattice_probe.py [--out bench/lattice_probe.json]
"""
import argparse
import json

import fsspec
import numpy as np
import xarray as xr

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
NCAR = "https://nsf-ncar-era5.s3.amazonaws.com/e5.oper.an.sfc/202001/e5.oper.an.sfc.{code}.ll025sc.2020010100_2020013123.nc"
PAIRS = {"2m_temperature": "128_167_2t", "mean_sea_level_pressure": "128_151_msl",
         "10m_u_component_of_wind": "128_165_10u", "total_column_water_vapour": "128_137_tcwv"}
WB2 = "https://storage.googleapis.com/weatherbench2/datasets/era5/"


def lattice(v):
    """(s, phi): coarsest s = 2^e whose codes are injective on the distinct values; None for < 2 values."""
    u = np.unique(v[np.isfinite(v)].astype("f8"))
    if u.size < 2:
        return None
    e_hi = int(np.floor(np.log2(np.min(np.diff(u))))) + 1  # a step above the smallest gap cannot be injective
    for e in range(e_hi, e_hi - 40, -1):
        s = 2.0 ** e
        phi = (np.angle(np.exp(2j * np.pi * u / s).mean()) / (2 * np.pi) % 1.0) * s
        if np.unique(np.round((u - phi) / s)).size == u.size:
            return s, phi
    return None


def codes(v, lat):
    s, phi = lat
    return np.round((np.asarray(v, "f8") - phi) / s).astype("i8")


def compare(a, b, tile):
    la, lb = lattice(a), lattice(b)
    r = {"exact_equal": round(float((a == b).mean()), 4), "unique": [int(np.unique(a).size), int(np.unique(b).size)],
         "log2_step": [la and int(np.log2(la[0])), lb and int(np.log2(lb[0]))]}
    same = bool(la and lb and la[0] == lb[0])
    r["codes_equal"] = round(float((codes(a, la) == codes(b, lb)).mean()), 6) if same else None
    th, tw = tile
    own = fld = n = 0
    for i in range(0, a.shape[0], th):
        for j in range(0, a.shape[1], tw):
            ta, tb = a[i:i + th, j:j + tw], b[i:i + th, j:j + tw]
            n += 1
            xa, xb = lattice(ta), lattice(tb)  # tile lattice from the tile alone
            own += bool(xa and xb and xa[0] == xb[0] and np.array_equal(codes(ta, xa), codes(tb, xb)))
            fld += bool(same and np.array_equal(codes(ta, la), codes(tb, lb)))  # lattice of the whole field
    r["tiles_identical_own_lattice"] = f"{own}/{n}"
    r["tiles_identical_field_lattice"] = f"{fld}/{n}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/lattice_probe.json")
    a = ap.parse_args()
    arco = xr.open_zarr(ARCO, consolidated=True, chunks=None)
    res = {"arco_vs_ncar": {}, "negative_wb2_1p5deg": {}, "names": {}}
    for var, code in PAIRS.items():
        d = xr.open_dataset(fsspec.open(NCAR.format(code=code), block_size=8 << 20).open(), engine="h5netcdf", chunks=None)
        nv = list(d.data_vars)[0]
        res["names"][var] = {"ncar_name": nv, "ncar_attrs": {k: str(x) for k, x in d[nv].attrs.items()},
                             "arco_attrs": {k: str(x) for k, x in arco[var].attrs.items()},
                             "coord_dtypes": [str(d.latitude.dtype), str(arco.latitude.dtype)],
                             "coords_equal": bool(np.array_equal(d.latitude.values.astype("f8"), arco.latitude.values.astype("f8"))
                                                  and np.array_equal(d.longitude.values.astype("f8"), arco.longitude.values.astype("f8")))}
        for ts in ("2020-01-03T18", "2020-01-15T06", "2020-01-28T00"):
            t = np.datetime64(ts)
            r = compare(arco[var].sel(time=t).values, d[nv].sel(time=t).values, (91, 180))
            res["arco_vs_ncar"][f"{var}@{ts}"] = r
            print(var, ts, json.dumps(r), flush=True)
    h = xr.open_zarr(WB2 + "1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr", consolidated=True, chunks=None)
    s6 = xr.open_zarr(WB2 + "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr", consolidated=True, chunks=None)
    for ts in ("2020-03-03T06", "2010-07-01T00"):
        r = compare(h["2m_temperature"].sel(time=ts).values, s6["2m_temperature"].sel(time=ts).values, (60, 61))
        res["negative_wb2_1p5deg"][ts] = r
        print("NEG wb2 1.5deg", ts, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
