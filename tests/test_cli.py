"""The CLI as a user runs it: real `zt` processes, config files, a closed two-member network, every subcommand."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ZT = [sys.executable, "-m", "zarrswarm.cli"]
_ports = iter(range(10000, 30000, 2))


def port():
    # Both CLI ports must be free; stay below Linux's outgoing ephemeral range.
    for p in _ports:
        with socket.socket() as data, socket.socket() as control:
            try:
                data.bind(("127.0.0.1", p))
                control.bind(("127.0.0.1", p + 1))
                return p
            except OSError:
                continue
    raise RuntimeError("no free test port pair")


def zt(home, ctl_port, *args, check=True):
    env = dict(os.environ, ZT_HOME=str(home), ZT_CTL=f"http://127.0.0.1:{ctl_port}")
    r = subprocess.run(ZT + list(args), env=env, capture_output=True, text=True, timeout=120)
    if check and r.returncode:
        raise AssertionError(f"zt {' '.join(args)} -> {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def wait_up(ctl_port, home, want=""):
    for _ in range(150):  # up (and, with `want`, done seeding: [[seed]] globs are seeded after start)
        try:
            st = zt(home, ctl_port, "status")
            if want in st:
                return st
        except AssertionError:
            pass
        time.sleep(0.2)
    raise AssertionError("node did not start")


def test_cli_end_to_end(tmp_path):
    bp, cp = port(), port()
    a_home, b_home = tmp_path / "a", tmp_path / "b"
    zt(a_home, bp + 1, "init", "--bootstrap-node", "--public-host", "127.0.0.1", "--port", str(bp), "--private")
    import tomllib
    invite = tomllib.loads((a_home / "config.toml").read_text())["network"]
    assert "?k=" in invite
    zt(b_home, cp + 1, "init", "--join", invite, "--port", str(cp))
    vals = np.arange(24 * 10 * 12, dtype="f4").reshape(240, 3, 4)
    ds = xr.Dataset({"t2m": (("time", "lat", "lon"), vals, {"standard_name": "air_temperature", "units": "K"})},
                    coords={"time": pd.date_range("2020-01-01", periods=240, freq="h"),
                            "lat": [50.0, 55.0, 60.0], "lon": [0.0, 10.0, 20.0, 30.0]})
    (tmp_path / "data").mkdir()
    ds.to_zarr(tmp_path / "data" / "era.zarr", encoding={"t2m": {"chunks": (24, 3, 4)}}, consolidated=False)
    cfg = (a_home / "config.toml").read_text().replace('# path = "/data/era5/*.zarr"',
                                                       f'[[seed]]\npath = "{tmp_path / "data"}/*.zarr"')
    (a_home / "config.toml").write_text(cfg)
    env = lambda h, p: dict(os.environ, ZT_HOME=str(h), ZT_CTL=f"http://127.0.0.1:{p}")
    procs = [subprocess.Popen(ZT + ["node"], env=env(h, p + 1), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for h, p in ((a_home, bp), (b_home, cp))]
    try:
        st_a = wait_up(bp + 1, a_home, "zt://")
        wait_up(cp + 1, b_home)
        assert "pubkey" in st_a
        link = next(w for w in st_a.split() if w.startswith("zt://"))  # seeded from [[seed]] glob at start
        for _ in range(50):
            hits = zt(b_home, cp + 1, "search", "air_temperature")
            if link in hits:
                break
            time.sleep(0.2)
        assert link in hits and "2020-01-10 23:00" in hits
        assert "chunks" in zt(b_home, cp + 1, "peers", link)
        out = tmp_path / "eu.nc"
        zt(b_home, cp + 1, "get", link, "--vars", "t2m", "--time", "2020-01-02:2020-01-03",
           "--sel", "lat=55:60,lon=10:20", "--out", str(out))
        got = xr.open_dataset(out, engine="h5netcdf").t2m.values
        np.testing.assert_array_equal(got, vals[24:72, 1:3, 1:3])
        m = json.loads(zt(b_home, cp + 1, "mean", link, "t2m", "--rel-err", "1e-9").splitlines()[-1])
        assert abs(m["mean"] - vals.mean()) < 1e-6 * vals.mean()
        name = zt(a_home, bp + 1, "name", "era", link).strip()
        assert name.startswith("zt://era@")
        for _ in range(50):
            r = subprocess.run(ZT + ["get", name, "--vars", "t2m", "--time", "2020-01-05:2020-01-05",
                                     "--out", str(tmp_path / "n.zarr")], env=env(b_home, cp + 1),
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                break
            time.sleep(0.2)
        assert r.returncode == 0, r.stderr
        np.testing.assert_array_equal(xr.open_zarr(tmp_path / "n.zarr").t2m.values, vals[96:120])
        par = json.loads(zt(b_home, cp + 1, "parity", link, "t2m", "--k", "4").splitlines()[-1])
        assert par["stripes"] >= 1
        zt(a_home, bp + 1, "unseed", str(tmp_path / "data" / "era.zarr"))
        assert "era.zarr" not in zt(a_home, bp + 1, "status")
    finally:
        for p in procs:
            p.terminate()
            p.wait(10)
