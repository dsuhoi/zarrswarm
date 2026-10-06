"""Headless TUI test, torrent-client flow: seed through the dialog on one node; on another search the metadata
index, open the add dialog from a hit (metadata loaded, variables listed), download a slice with export, inspect
content tree / piece map / peers, edit settings, remove."""
import asyncio
import socket
import threading

import numpy as np
import pandas as pd
import xarray as xr
from textual.widgets import DataTable, Input, SelectionList, Static, TabbedContent, TextArea, Tree

from zarrswarm.cli import _config
from zarrswarm.node import Node
from zarrswarm.store import http
from zarrswarm.tui import AddDialog, SearchDialog, SeedDialog, SettingsScreen, ZtTui


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def until(pilot, cond, n=100):
    for _ in range(n):
        await pilot.pause(0.1)
        if cond():
            return True
    return False


def test_tui_torrent_flow(tmp_path, monkeypatch):
    from zarrswarm import store
    monkeypatch.setattr(store, "READAHEAD", 0)  # export scans time: no prefetch past the slice, exact piece map
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(60)
    bp = port()
    seeder = Node(tmp_path / "h_s", port=bp, ctl_port=port(), relay_server=True)
    run(seeder.start())
    client = Node(tmp_path / "h_c", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"],
                  relay=f"http://127.0.0.1:{bp}")
    run(client.start())
    times = pd.date_range("2020-01-01", periods=48, freq="h")
    vals = np.random.rand(48, 4, 5).astype("float32")
    p = str(tmp_path / "d.zarr")
    xr.Dataset({"t2m": (("time", "lat", "lon"), vals, {"long_name": "2 metre temperature", "units": "K"}),
                "u10": (("time", "lat", "lon"), vals + 1)},
               coords={"time": times, "lat": np.arange(4.0), "lon": np.arange(5.0)}).to_zarr(
        p, encoding={"t2m": {"chunks": (12, 4, 5)}, "u10": {"chunks": (12, 4, 5)}}, consolidated=False)
    out = str(tmp_path / "out.nc")

    async def drive():
        app = ZtTui(f"http://127.0.0.1:{seeder.ctl_port}", home=tmp_path / "h_s")
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.press("s")
            dlg = app.screen
            assert isinstance(dlg, SeedDialog)
            dlg.query_one("#path", Input).value = p
            await pilot.click("#scan")
            assert await until(pilot, lambda: "t2m" in str(dlg.query_one("#preview", Static).render()))
            await pilot.click("#ok")
            assert await until(pilot, lambda: app.query_one("#list", DataTable).row_count == 1)
            assert "раздача" in str(app.query_one("#list", DataTable).get_row_at(0)[3])
            grid = app.rows[0]["grid"]
        app = ZtTui(f"http://127.0.0.1:{client.ctl_port}", home=tmp_path / "h_c")
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.press("slash")
            sd = app.screen
            assert isinstance(sd, SearchDialog)
            sd.query_one("#tag", Input).value = "2 metre temperature"
            await pilot.press("enter")
            assert await until(pilot, lambda: sd.query_one("#found", DataTable).row_count >= 1)
            assert str(sd.query_one("#found", DataTable).get_row_at(0)[0]) == f"zt://{grid}"
            await pilot.press("enter")  # hit -> add dialog with the link, metadata loaded automatically
            assert await until(pilot, lambda: isinstance(app.screen, AddDialog))
            ad = app.screen
            sl = ad.query_one("#vars", SelectionList)
            assert await until(pilot, lambda: sl.option_count == 2)
            assert "время 2020-01-01 00:00 … 2020-01-02 23:00" in str(ad.query_one("#info", Static).render())
            sl.deselect("u10")
            ad.query_one("#time", Input).value = "2020-01-01:2020-01-01"
            ad.query_one("#out", Input).value = out
            await pilot.click("#ok")
            assert await until(pilot, lambda: any(r["jobs"] and r["jobs"][-1]["state"] == "done" for r in app.rows))
            assert await until(pilot, lambda: __import__("os").path.exists(out), 100)
            row = next(r for r in app.rows if r["grid"] == grid)
            assert row["kind"] == "done" and row["chunks"] >= 2
            # details: content tree, piece map, peers
            app.query_one("#list", DataTable).focus()
            app.selected = grid
            tabs = app.query_one("#details", TabbedContent)
            tabs.active = "d_content"
            tree = app.query_one("#content", Tree)
            labels = lambda: " ".join(str(n.label) for n in tree.root.children for n in [n, *n.children])
            assert await until(pilot, lambda: "t2m" in labels() and "u10" in labels())
            assert "2 metre temperature" in labels() and "lat: 4" in labels()
            tabs.active = "d_pieces"
            pieces = app.query_one("#pieces", Static)
            assert await until(pilot, lambda: "t2m" in str(pieces.render()))
            lines = str(pieces.render()).splitlines()
            strip = lines[next(i for i, l in enumerate(lines) if l.startswith("t2m")) + 1]
            half = len(strip) // 2  # day 1 downloaded here, day 2 only at the seeder (one holder: rare)
            assert set(strip[:half - 1]) == {"█"} and set(strip[half + 1:]) == {"░"}, strip
            tabs.active = "d_peers"
            peers = app.query_one("#peers", DataTable)
            assert await until(pilot, lambda: peers.row_count >= 2)
            # settings
            await pilot.press("o")
            st = app.screen
            assert isinstance(st, SettingsScreen)
            st.query_one("#upload_mbps", Input).value = "12.5"
            st.query_one("#cache_max_gb", Input).value = "50"
            st.query_one("#seeds", TextArea).text = "/data/*.zarr"
            st.query_one("#tuning", TextArea).text = "audit_rate = 0.2"
            st.query_one("#save").press()
            assert await until(pilot, lambda: not isinstance(app.screen, SettingsScreen))
        cfg = _config(tmp_path / "h_c")
        assert cfg["upload_mbps"] == 12.5 and cfg["cache_max_gb"] == 50 and cfg["seed"] == [{"path": "/data/*.zarr"}]
        assert cfg["tuning"] == {"audit_rate": 0.2}
        # remove on the seeder: unseed after confirmation
        app = ZtTui(f"http://127.0.0.1:{seeder.ctl_port}", home=tmp_path / "h_s")
        async with app.run_test(size=(160, 50)) as pilot:
            assert await until(pilot, lambda: app.query_one("#list", DataTable).row_count == 1)
            app.query_one("#list", DataTable).focus()
            app.selected = grid
            await pilot.press("delete")
            await pilot.click("#yes")
            assert await until(pilot, lambda: not http(app.ctl, "GET", "/api/status")["seeds"])

    asyncio.run(drive())
    got = xr.open_dataset(out, engine="h5netcdf")
    np.testing.assert_array_equal(got.t2m.values, vals[:24])
    assert "u10" not in got
    for n in (client, seeder):
        run(n.stop())
