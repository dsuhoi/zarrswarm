"""config.toml: `zt init` writes a commented, parseable file; re-init keeps [[seed]]/[tuning]; a node seeds the
config's globs at start."""
import asyncio

import numpy as np
import pandas as pd
import xarray as xr

from zarr_torrent import cli
from zarr_torrent.node import Node


def test_init_writes_toml_and_keeps_user_sections(tmp_path):
    home = tmp_path / "h"
    cli.main(["init", "--home", str(home), "--bootstrap-node", "--public-host", "zt.example.org", "--port", "7990"])
    cfg = cli._config(home)
    assert cfg["network"].endswith("@zt.example.org:7990") and cfg["port"] == 7990 and cfg["relay_server"] is True
    assert cfg["public"] == "http://zt.example.org:7990" and cfg["bootstrap"] == [] and cfg["upload_mbps"] == 0
    text = (home / "config.toml").read_text()
    (home / "config.toml").write_text(text.replace("# ctl_port = 7991", "ctl_port = 7999").replace("[tuning]\n", '[[seed]]\npath = "/d/*.zarr"\n\n[tuning]\n'
                                                   "audit_rate = 0.1\n"))
    cli.main(["init", "--home", str(home), "--join", cfg["network"], "--listen-public"])  # switch role
    cfg = cli._config(home)
    assert cfg["ctl_port"] == 7999 and cfg["seed"] == [{"path": "/d/*.zarr"}] and cfg["tuning"] == {"audit_rate": 0.1}
    assert cfg["bootstrap"] == ["http://zt.example.org:7990"] and cfg["auto"] is True and cfg["host"] == "0.0.0.0"
    assert set(cli.TUNING) >= set(cfg["tuning"])


def test_node_seeds_config_globs_at_start(tmp_path):
    ds = xr.Dataset({"t": (("time", "x"), np.arange(48.0).reshape(24, 2))},
                    coords={"time": pd.date_range("2020-01-01", periods=24, freq="h"), "x": [0.0, 1.0]})
    for n in ("a", "b"):
        ds.isel(time=slice(0, 12) if n == "a" else slice(12, 24)).to_zarr(tmp_path / f"{n}.zarr", consolidated=False)

    async def run():
        node = Node(tmp_path / "home", port=0, ctl_port=0, seeds=[str(tmp_path / "*.zarr"), str(tmp_path / "gone")])
        node.port, node.ctl_port = _free(), _free()
        await node.start()
        try:
            return sorted({p for p, _ in node.seeds})
        finally:
            await node.stop()
    assert asyncio.run(run()) == [str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")]  # missing path: logged, skipped


def _free():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_client_reads_through_api_when_node_paths_are_invisible(tmp_path, monkeypatch):
    """Node in a container: its file paths do not exist for the client, which must fall back to /api/read."""
    import threading
    from pathlib import Path

    import zarr_torrent as zt
    from zarr_torrent import store
    from zarr_torrent.store import http
    vals = np.arange(240 * 12, dtype="f4").reshape(240, 3, 4)
    xr.Dataset({"t2m": (("time", "y", "x"), vals)},
               coords={"time": pd.date_range("2020-01-01", periods=240, freq="h"), "y": np.arange(3.0),
                       "x": np.arange(4.0)}).to_zarr(tmp_path / "d.zarr", encoding={"t2m": {"chunks": (24, 3, 4)}},
                                                     consolidated=False)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    node = Node(tmp_path / "home", port=_free(), ctl_port=_free())
    asyncio.run_coroutine_threadsafe(node.start(), loop).result(30)
    ctl = f"http://127.0.0.1:{node.ctl_port}"
    link = http(ctl, "POST", "/api/seed", {"path": str(tmp_path / "d.zarr")})["link"]

    class Elsewhere(type(Path())):  # the client's filesystem does not have the node's files
        def read_bytes(self):
            raise FileNotFoundError(self)
    monkeypatch.setattr(store, "Path", Elsewhere)
    try:
        np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl).t2m.values, vals)
        r = store.progressive_mean_vas(zt.open_dataset(link, ctl=ctl), "t2m", rel_err=1e-9)
        assert abs(r["mean"] - vals.mean()) < 1e-6 * abs(vals.mean()), r
        try:  # /api/read serves only indexed chunks, never an arbitrary path
            http(ctl, "POST", "/api/read", {"grid": link.removeprefix("zt://").split("+")[0], "key": "../../etc"})
            raise AssertionError("arbitrary key served")
        except RuntimeError as e:
            assert "404" in str(e)
    finally:
        asyncio.run_coroutine_threadsafe(node.stop(), loop).result(30)
