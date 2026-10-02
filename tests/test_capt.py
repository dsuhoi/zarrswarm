"""CAPT manifests: used instead of the full manifest, and a grown replica costs only a few new pages."""
import asyncio
import socket
import threading

import numpy as np
import pandas as pd
import xarray as xr

from zarr_torrent.node import Node
from zarr_torrent.store import http


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_capt_incremental_manifest_sync(tmp_path):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(120)
    bp = port()
    seeder = Node(tmp_path / "s", port=bp, ctl_port=port())
    client = Node(tmp_path / "c", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    run(seeder.start())
    run(client.start())
    sctl, cctl = f"http://127.0.0.1:{seeder.ctl_port}", f"http://127.0.0.1:{client.ctl_port}"
    p = str(tmp_path / "grow.zarr")
    times = pd.date_range("2020-01-01", periods=24 * 400, freq="h")

    def ds(n):
        return xr.Dataset({v: (("time", "y", "x"), np.ones((n, 2, 2), "float32") * i) for i, v in enumerate("abcd")},
                          coords={"time": times[:n], "y": [0.0, 1.0], "x": [0.0, 1.0]})
    ds(24 * 399).to_zarr(p, encoding={v: {"chunks": (24, 2, 2)} for v in "abcd"}, consolidated=False)
    link = http(sctl, "POST", "/api/seed", {"path": p})["link"]
    grid = link.removeprefix("zt://")
    v1 = http(cctl, "GET", f"/api/view/{grid}?refresh=1")
    full = http(cctl, "GET", "/api/status")["page_fetched"]
    assert v1["nchunks"] >= 4 * 399 and full > 0
    assert http(sctl, "GET", "/api/status")["served"].get("legacy_manifest", 0) == 0  # CAPT path was used
    ds(24 * 400).isel(time=slice(24 * 399, None)).to_zarr(p, append_dim="time")  # one more day
    http(sctl, "POST", "/api/seed", {"path": p})
    v2 = http(cctl, "GET", f"/api/view/{grid}?refresh=1")
    delta = http(cctl, "GET", "/api/status")["page_fetched"] - full
    assert v2["nchunks"] == v1["nchunks"] + 4
    assert delta < 0.4 * full, (delta, full)  # tiny tree here; at 1.5e5 chunks the capt self-check shows ~1.4%
    # a fresh client asking for one week never downloads the whole manifest (CAPT range view)
    fresh = Node(tmp_path / "f", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    run(fresh.start())
    fctl = f"http://127.0.0.1:{fresh.ctl_port}"
    jid = http(fctl, "POST", "/api/download", {"grid": grid, "region": {"var": "c", "t0": "2020-06-01",
                                                                       "t1": "2020-06-07"}})["job"]
    from zarr_torrent.store import wait_job
    import zarr_torrent as zt
    job = wait_job(fctl, jid)
    assert job["state"] == "done" and job["done"] == 7 and job["cover"].get("view") == "capt-range", job
    assert http(fctl, "GET", "/api/status")["page_fetched"] < 0.3 * full
    got = zt.open_dataset(link, ctl=fctl).c.sel(time=slice("2020-06-01", "2020-06-07")).values
    assert got.shape == (168, 2, 2) and (got == 2).all()
    for n in (fresh, client, seeder):
        run(n.stop())
