"""End-to-end on localhost: 5 nodes (bootstrap+relay, 2 public seeders, 1 NAT'd seeder behind the
relay, 1 NAT'd client). Three replicas with different time ranges, variable sets and codecs."""
import asyncio
import socket
import threading

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import zarrswarm as zs
from zarrswarm.node import Node
from zarrswarm.store import http

LAT, LON = np.linspace(-10, 10, 8), np.linspace(0, 30, 12)


def field(var, times):
    s = (times.values.astype("datetime64[s]").astype("int64") / 3600.0)[:, None, None]
    k = {"t2m": 1.0, "u10": 2.0, "v10": 3.0}[var]
    return (np.sin(s / 50.0 * k + LAT[None, :, None] / 7) * np.cos(LON[None, None, :] / 9) * 10 + 280 * (var == "t2m")).astype("float32")


def make(path, vars_, t0, t1, blosc=False):
    times = pd.date_range(t0, t1, freq="h")
    ds = xr.Dataset({v: (("time", "lat", "lon"), field(v, times)) for v in vars_},
                    coords={"time": times, "lat": LAT, "lon": LON})
    enc = {v: {"chunks": (24, 4, 6)} for v in vars_}
    if blosc:
        for v in vars_:
            enc[v]["compressors"] = [zarr.codecs.BloscCodec(cname="lz4", clevel=3)]
    ds.to_zarr(path, zarr_format=3, encoding=enc, consolidated=False)
    return str(path)


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def swarm(tmp_path):
    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, daemon=True)
    th.start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(120)
    bp = port()
    boot_url = f"http://127.0.0.1:{bp}"
    nodes = {"boot": Node(tmp_path / "h_boot", port=bp, ctl_port=port(), relay_server=True)}
    run(nodes["boot"].start())
    for name, kw in [("a", {}), ("b", {}), ("c", {"relay": boot_url}), ("client", {"relay": boot_url})]:
        n = Node(tmp_path / f"h_{name}", port=port(), ctl_port=port(), bootstrap=[boot_url], **kw)
        run(n.start())
        nodes[name] = n
    yield nodes, run, tmp_path
    for n in nodes.values():
        run(n.stop())
    loop.call_soon_threadsafe(loop.stop)


def ctl(n):
    return f"http://127.0.0.1:{n.ctl_port}"


def test_swarm_union_transcode_relay_progressive(swarm):
    nodes, run, tmp = swarm
    pa = make(tmp / "a.zarr", ["t2m", "u10"], "2020-01-01", "2020-01-10 23:00")
    pb = make(tmp / "b.zarr", ["t2m", "v10"], "2020-01-06", "2020-01-20 11:00", blosc=True)  # partial tail chunk
    pc = make(tmp / "c.zarr", ["t2m"], "2020-01-18", "2020-01-25 23:00")
    links = {http(ctl(nodes[n]), "POST", "/api/seed", {"path": p})["link"] for n, p in (("a", pa), ("b", pb), ("c", pc))}
    assert len(links) == 1, links  # same grid family despite different extent/vars/codecs/time units
    link = links.pop()

    ds = zs.open_dataset(link, ctl=ctl(nodes["client"]))
    times = pd.date_range("2020-01-01", "2020-01-25 23:00", freq="h")
    assert ds.sizes["time"] == len(times)
    assert (ds.time.values == times.values).all()

    t2m = ds.t2m.values
    np.testing.assert_array_equal(t2m, field("t2m", times))
    u10 = ds.u10.values
    m = times <= "2020-01-10 23:00"
    np.testing.assert_array_equal(u10[m], field("u10", times[m]))
    assert np.isnan(u10[~m]).all()
    v10 = ds.v10.values
    m = (times >= "2020-01-06") & (times <= "2020-01-20 11:00")
    np.testing.assert_array_equal(v10[m], field("v10", times[m]))
    assert np.isnan(v10[~m]).all()

    st = http(ctl(nodes["client"]), "GET", "/api/status")
    # the NAT'd client downloaded from several peers including the NAT'd seeder via relay
    peers = {p for p, b in st["bw"].items()}
    assert nodes["c"].ident.id in peers and len(peers) >= 2

    # sliced prefetch + reopened dataset served from cache
    sub = ds.t2m.sel(time=slice("2020-01-03", "2020-01-04"))
    zs.prefetch(sub)
    np.testing.assert_array_equal(sub.values, field("t2m", pd.date_range("2020-01-03", "2020-01-04 23:00", freq="h")))

    # progressive error-bounded mean on a fresh client that has nothing cached
    fresh = Node(tmp / "h_fresh", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{nodes['boot'].port}"])
    run(fresh.start())
    try:
        ds2 = zs.open_dataset(link, ctl=ctl(fresh))
        r = zs.progressive_mean(ds2, "u10", rel_err=0.5, min_chunks=4)
        true = float(np.nanmean(field("u10", pd.date_range("2020-01-01", "2020-01-10 23:00", freq="h"))))
        assert r["mean"] is not None and abs(r["mean"] - true) <= max(3 * r["ci95"], 1e-6), f"{r} {true}"
        r2 = zs.progressive_mean_vas(ds2, "u10", rel_err=0.5)
        assert abs(r2["mean"] - true) <= max(3 * r2["ci95"], 1e-6), f"{r2} {true}"
    finally:
        run(fresh.stop())

    # signed mutable name in the DHT
    named = http(ctl(nodes["a"]), "POST", "/api/name", {"name": "era5-test", "target": link})["link"]
    assert http(ctl(nodes["client"]), "GET", "/api/resolve?link=" + named)["grid"] == link.removeprefix("zs://")
