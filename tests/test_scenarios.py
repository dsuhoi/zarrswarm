"""Scenario matrix over one localhost swarm (bootstrap+relay, 2 public seeders, 1 NAT'd seeder, NAT'd client).

Oracle for every scenario: xarray.combine_first over the source replicas opened directly, reindexed to the
swarm's union time axis. The swarm view must be value-identical (NaN == NaN) for every variable.
"""
import asyncio
import os
import shutil
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import zarrswarm as zt
from zarrswarm.node import Node
from zarrswarm.scan import scan
from zarrswarm.store import http

LAT, LON, LEV = np.linspace(-10, 10, 8), np.linspace(0, 30, 12), np.array([1000, 850, 500], dtype="float64")


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def swarm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("swarm")
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(120)
    bp = port()
    boot = f"http://127.0.0.1:{bp}"
    nodes = {"boot": Node(tmp / "h_boot", port=bp, ctl_port=port(), relay_server=True)}
    run(nodes["boot"].start())
    for name, kw in [("a", {}), ("b", {}), ("c", {"relay": boot}), ("client", {"relay": boot})]:
        nodes[name] = Node(tmp / f"h_{name}", port=port(), ctl_port=port(), bootstrap=[boot], **kw)
        run(nodes[name].start())
    yield {"nodes": nodes, "run": run, "tmp": tmp, "boot": boot}
    for n in nodes.values():
        run(n.stop())
    loop.call_soon_threadsafe(loop.stop)


def ctl(sw, name):
    return f"http://127.0.0.1:{sw['nodes'][name].ctl_port}"


def field(name, times, shape_extra):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    base = rng.normal(size=shape_extra).astype("float32")
    s = (times.values.astype("datetime64[s]").astype("int64") / 3600.0)
    return (np.sin(s / 30.0)[(...,) + (None,) * len(shape_extra)] * 5 + base[None]).astype("float32")


def make(path, times, vars_, fmt=3, chunks=None, shards=None, encoding=None, level_vars=(), static=False):
    coords = {"lat": LAT, "lon": LON}
    data = {}
    if times is not None:
        coords["time"] = times
    for v in vars_:
        if static:
            data[v] = (("lat", "lon"), field(v, pd.DatetimeIndex(["2000-01-01"]), (8, 12))[0])
        elif v in level_vars:
            coords["level"] = LEV
            data[v] = (("time", "level", "lat", "lon"), field(v, times, (3, 8, 12)))
        else:
            data[v] = (("time", "lat", "lon"), field(v, times, (8, 12)))
    ds = xr.Dataset(data, coords=coords)
    enc = {}
    for v in vars_:
        e = dict((encoding or {}).get(v, {}))
        nd = ds[v].ndim
        if chunks:
            e["chunks"] = tuple(chunks[:nd]) if v not in level_vars else (chunks[0], 3) + tuple(chunks[1:3])
        if shards and fmt == 3:
            e["shards"] = tuple(shards[:nd])
        enc[v] = e
    ds.to_zarr(path, zarr_format=fmt, encoding=enc, consolidated=False)
    return str(path)


def seed(sw, node, path):
    return http(ctl(sw, node), "POST", "/api/seed", {"path": path})["link"]


def oracle(paths, tname="time"):
    srcs = [xr.open_zarr(p, consolidated=False) for p in paths]
    out = srcs[0]
    for s in srcs[1:]:
        out = out.combine_first(s)
    return out


def check_equal(ds, exp, variables):
    for v in variables:
        got = ds[v]
        e = exp[v]
        if "time" in got.dims:
            e = e.reindex(time=got.time.values)
        np.testing.assert_array_equal(got.values, e.transpose(*got.dims).values, err_msg=v)


def open_client(sw, link, **kw):
    return zt.open_dataset(link, ctl=ctl(sw, "client"), **kw)


# ----------------------------------------------------------------------------------------------- scenarios

def test_zarr_v2_union_different_ranges(swarm):
    t = swarm["tmp"] / "s_v2"
    pa = make(t / "a.zarr", pd.date_range("2021-01-01", periods=72, freq="h"), ["t2m"], fmt=2, chunks=(24, 4, 6))
    pb = make(t / "b.zarr", pd.date_range("2021-01-02", periods=96, freq="h"), ["t2m", "q"], fmt=2, chunks=(24, 4, 6))
    la, lb = seed(swarm, "a", pa), seed(swarm, "c", pb)
    assert la == lb
    ds = open_client(swarm, la)
    assert ds.sizes["time"] == 24 * 5
    check_equal(ds, oracle([pa, pb]), ["t2m", "q"])


def test_heterogeneous_layouts_single_index_and_rechunk(swarm):
    """Replicas chunked differently are one grid; native, direct and assembled views all agree."""
    t = swarm["tmp"] / "s_layout"
    times = pd.date_range("2021-03-01", periods=24 * 6, freq="h")
    pa = make(t / "a.zarr", times[:24 * 4], ["t2m_l"], chunks=(24, 4, 6))
    pb = make(t / "b.zarr", times[24 * 2:], ["t2m_l"], chunks=(48, 8, 12))
    la, lb = seed(swarm, "a", pa), seed(swarm, "b", pb)
    assert la == lb, "different chunk layouts must share one grid (single index)"
    exp = oracle([pa, pb])
    native = open_client(swarm, la)
    check_equal(native, exp, ["t2m_l"])
    # time-series friendly view: whole time axis, one pixel per chunk -> assembled from both layouts
    ts = open_client(swarm, la, chunking={"time": -1, "lat": 1, "lon": 1})
    check_equal(ts, exp, ["t2m_l"])
    st = zt.store_of(ts)
    assert st.stats["assembled"] > 0 and st.stats["direct"] == 0
    # view equal to layout B -> direct pass-through wherever B exists, assembled elsewhere
    vb = open_client(swarm, la, chunking={"time": 48, "lat": 8, "lon": 12})
    check_equal(vb, exp, ["t2m_l"])


def test_sharded_v3(swarm):
    t = swarm["tmp"] / "s_shard"
    times = pd.date_range("2021-05-01", periods=96, freq="h")
    pa = make(t / "a.zarr", times[:48], ["t2m_s"], chunks=(12, 4, 6), shards=(24, 8, 12))
    pb = make(t / "b.zarr", times[24:], ["t2m_s"], chunks=(12, 4, 6), shards=(24, 8, 12))
    link = seed(swarm, "a", pa)
    assert seed(swarm, "b", pb) == link
    ds = open_client(swarm, link)
    check_equal(ds, oracle([pa, pb]), ["t2m_s"])


def test_static_no_time_and_scalar(swarm):
    t = swarm["tmp"] / "s_static"
    p = make(t / "a.zarr", None, ["orog", "lsm"], static=True, chunks=(4, 6))
    z = zarr.open_group(p, mode="a")
    arr = z.create_array("scale", shape=(), dtype="float64", fill_value=0.0)
    arr[...] = 3.5
    arr.attrs["_ARRAY_DIMENSIONS"] = []
    link = seed(swarm, "b", p)
    ds = open_client(swarm, link)
    check_equal(ds, oracle([p]), ["orog", "lsm"])
    assert float(ds["scale"]) == 3.5


def test_cf_packed_int_with_fill_and_missing_chunk(swarm):
    t = swarm["tmp"] / "s_int"
    times = pd.date_range("2021-07-01", periods=48, freq="h")
    enc = {"t2m_i": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 0.0, "_FillValue": -32767}}
    pa = make(t / "a.zarr", times, ["t2m_i"], chunks=(12, 4, 6), encoding=enc)
    # delete one chunk file: must come back as fill (NaN after decoding), identical to the source
    victim = next(f for f in Path(pa, "t2m_i").rglob("*") if f.is_file() and f.name != "zarr.json")
    victim.unlink()
    link = seed(swarm, "c", pa)
    ds = open_client(swarm, link)
    exp = oracle([pa])
    check_equal(ds, exp, ["t2m_i"])


def test_daily_float_units_with_phase(swarm):
    t = swarm["tmp"] / "s_daily"
    times = pd.date_range("2019-12-30T12:00", periods=40, freq="D")
    enc = {"time": {"units": "days since 1950-01-01", "dtype": "float64"}}
    pa = make(t / "a.zarr", times[:25], ["tp"], chunks=(5, 8, 12))
    ds_b = xr.Dataset({"tp": (("time", "lat", "lon"), field("tp", times[20:], (8, 12)))},
                      coords={"time": times[20:], "lat": LAT, "lon": LON})
    pb = str(t / "b.zarr")
    ds_b.to_zarr(pb, encoding={"tp": {"chunks": (5, 8, 12)}} | enc, consolidated=False)
    la, lb = seed(swarm, "a", pa), seed(swarm, "b", pb)
    assert la == lb, "CF units/dtype of time must not split the grid"
    ds = open_client(swarm, la)
    assert (ds.time.values == times.values).all()
    check_equal(ds, oracle([pa, pb]), ["tp"])


def test_multilevel_and_surface_vars_dask_and_dataarray_ops(swarm):
    t = swarm["tmp"] / "s_lev"
    times = pd.date_range("2021-09-01", periods=48, freq="h")
    pa = make(t / "a.zarr", times, ["z", "t2m_m"], chunks=(12, 4, 6), level_vars=("z",))
    link = seed(swarm, "a", pa)
    ds = open_client(swarm, link, chunks={})  # dask-backed
    exp = oracle([pa])
    check_equal(ds, exp, ["z", "t2m_m"])
    win = slice(times[3], times[29])  # label-based: merged sub-grids share a longer union time axis
    da = ds.z.sel(level=850, lat=slice(-5, 5), time=win)
    ref = exp.z.sel(level=850, lat=slice(-5, 5), time=win)
    np.testing.assert_allclose(da.mean("time").compute().values, ref.mean("time").values, rtol=1e-6)
    zt.prefetch(ds.t2m_m.sel(time=slice(times[0], times[11])))  # DataArray prefetch


def test_search_index(swarm):
    t = swarm["tmp"] / "s_search"
    ds = xr.Dataset({"sst": (("time", "lat", "lon"), field("sst", pd.date_range("2022-01-01", periods=24, freq="h"), (8, 12)),
                             {"standard_name": "sea_surface_temperature"})},
                    coords={"time": pd.date_range("2022-01-01", periods=24, freq="h"), "lat": LAT, "lon": LON})
    p = str(t / "a.zarr")
    ds.to_zarr(p, consolidated=False)
    link = seed(swarm, "b", p)
    for tag in ("sst", "sea_surface_temperature"):
        hits = http(ctl(swarm, "client"), "GET", f"/api/search?tag={tag}")
        assert any(f"zt://{h['grid']}" == link for h in hits), (tag, hits)


def test_irregular_time_is_rejected(tmp_path):
    times = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04"])
    p = make(tmp_path / "irr.zarr", times, ["t2m"], chunks=(1, 8, 12))
    with pytest.raises(ValueError, match="regular"):
        scan(p)


def test_auto_addressing(swarm):
    """A node with --auto learns its address from the bootstrap (STUN-like) and verifies reachability."""
    run = swarm["run"]
    n = Node(swarm["tmp"] / "h_auto", host="127.0.0.1", port=port(), ctl_port=port(), bootstrap=[swarm["boot"]],
             auto=True)
    run(n.start())
    try:
        assert n.addr == f"http://127.0.0.1:{n.port}" and not n.ro
    finally:
        run(n.stop())


@pytest.mark.parametrize("n_v2,n_v3", [(2, 1), (1, 2)])
def test_mixed_zarr_formats_in_one_view(swarm, n_v2, n_v3):
    """v2 and v3 replicas of one grid: the minority format is re-served through assembled metadata."""
    lat = np.linspace(0, 4, 5) + n_v2  # own grid per parametrization
    t = swarm["tmp"] / f"s_mixed_{n_v2}{n_v3}"
    times = pd.date_range("2022-06-01", periods=48, freq="h")
    paths = []
    for i in range(n_v2 + n_v3):
        fmt = 2 if i < n_v2 else 3
        name = f"m{i}"
        ds = xr.Dataset({name: (("time", "lat", "lon"), field(name, times, (5, 12)))},
                        coords={"time": times, "lat": lat, "lon": LON})
        p = str(t / f"{name}.zarr")
        ds.to_zarr(p, zarr_format=fmt, encoding={name: {"chunks": (12, 5, 6)}}, consolidated=False)
        paths.append(p)
    links = {seed(swarm, ["a", "b", "c"][i % 3], p) for i, p in enumerate(paths)}
    assert len(links) == 1
    ds = open_client(swarm, links.pop())
    check_equal(ds, oracle(paths), [f"m{i}" for i in range(len(paths))])


def test_region_download_jlps_and_bytes_cover(swarm):
    """Server-side cover selection over mixed layouts: both covers download the region completely."""
    t = swarm["tmp"] / "s_region"
    times = pd.date_range("2023-01-01", periods=24 * 14, freq="h")
    pa = make(t / "a.zarr", times, ["r2m"], chunks=(24, 8, 12))
    pb = make(t / "b.zarr", times, ["r2m"], chunks=(168, 8, 12))
    link = seed(swarm, "a", pa)
    assert seed(swarm, "b", pb) == link
    grid = link.removeprefix("zt://")
    for cover in ("bytes", "jlps"):
        jid = http(ctl(swarm, "client"), "POST", "/api/download",
                   {"grid": grid, "cover": cover, "region": {"var": "r2m", "t0": "2023-01-03", "t1": "2023-01-05"}})["job"]
        job = zt.store.wait_job(ctl(swarm, "client"), jid)
        assert job["state"] == "done" and job["missing"] == 0 and job["done"] > 0, job
    ds = open_client(swarm, link)
    check_equal(ds.sel(time=slice("2023-01-03", "2023-01-05")), oracle([pa]).sel(time=slice("2023-01-03", "2023-01-05")),
                ["r2m"])


def test_spatial_region_download_only_touches_overlapping_tiles(swarm):
    t = swarm["tmp"] / "s_bbox"
    times = pd.date_range("2023-06-01", periods=48, freq="h")
    p = make(t / "a.zarr", times, ["bb"], chunks=(24, 4, 6))       # lat 8 -> 2 tiles, lon 12 -> 2 tiles
    link = seed(swarm, "a", p)
    grid = link.removeprefix("zt://")
    jid = http(ctl(swarm, "client"), "POST", "/api/download",
               {"grid": grid, "region": {"var": "bb", "isel": {"lat": [0, 3], "lon": [7, 12]}}})["job"]
    job = zt.store.wait_job(ctl(swarm, "client"), jid)
    assert job["state"] == "done" and job["total"] == 2, job   # 2 time chunks x 1 lat tile x 1 lon tile
    ds = open_client(swarm, link)
    sub = ds.bb.isel(lat=slice(0, 3), lon=slice(7, 12)).sel(time=slice(times[0], times[-1]))
    ref = oracle([p]).bb.isel(lat=slice(0, 3), lon=slice(7, 12))
    np.testing.assert_array_equal(sub.values, ref.transpose(*sub.dims).values)


def test_cli_get_to_netcdf_and_zarr_with_sel(swarm, tmp_path):
    from zarrswarm.cli import main as zt_main
    t = swarm["tmp"] / "s_cli_out"
    times = pd.date_range("2023-09-01", periods=48, freq="h")
    p = make(t / "a.zarr", times, ["nc1"], chunks=(24, 4, 6))
    link = seed(swarm, "a", p)
    exp = oracle([p]).nc1.sel(lat=slice(-3, 10))
    for out in (str(tmp_path / "o.nc"), str(tmp_path / "o.zarr")):
        zt_main(["--ctl", ctl(swarm, "client"), "get", link, "--vars", "nc1", "--sel", "lat=-3:10",
                 "--time", "2023-09-01:2023-09-02", "--out", out])
        got = xr.open_dataset(out) if out.endswith(".nc") else xr.open_zarr(out, consolidated=False)
        np.testing.assert_array_equal(got.nc1.values, exp.transpose(*got.nc1.dims).values)


def test_sequential_scan_triggers_readahead(swarm):
    t = swarm["tmp"] / "s_ra"
    times = pd.date_range("2023-11-01", periods=24 * 10, freq="h")
    p = make(t / "a.zarr", times, ["ra"], chunks=(24, 8, 12))
    link = seed(swarm, "b", p)
    ds = open_client(swarm, link)
    for d in range(3):  # a training loop reading day after day
        ds.ra.sel(time=slice(times[24 * d], times[24 * d + 23])).values
    st = zt.store_of(ds)
    assert st.stats["readahead"] >= 1
    time.sleep(1.5)  # background prefetch lands in the local cache
    g = link.removeprefix("zt://")
    have = http(ctl(swarm, "client"), "GET", "/api/status")["grids"][g]["chunks"]
    assert have >= 3 + 1
