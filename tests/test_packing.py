"""Source-contract regression checks: signatures, real transfer, padding, cache and repair."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import zarr_torrent as zt
from zarr_torrent import codec
from zarr_torrent.common import check_signed, cid_of
from zarr_torrent.node import merge_view
from zarr_torrent.scan import scan
from zarr_torrent.store import View, http, wait_job
from test_security import ctl, net  # reuse the existing five-node local network


def write(path, values, times, chunks):
    xr.Dataset({"v": (("time", "y", "x"), values, {"units": "K"})},
               coords={"time": times, "y": np.arange(values.shape[1], dtype="f8"),
                       "x": np.arange(values.shape[2], dtype="f8")}).to_zarr(
        path, encoding={"v": {"chunks": chunks}}, consolidated=False)
    return str(path)


def samples(n=14):
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    codes = np.arange(15).reshape(3, 5) * 16
    triples = [(1.0 if i % 2 else 2.0, 268.125 + i / 4, .1) for i in range(n)]
    values = np.stack([origin + step * codes for step, origin, _ in triples])
    spec = {"v": {"time": {str(int(t.timestamp())): list(p) for t, p in zip(times, triples)}}}
    return times, values, spec


def test_slice_contract_and_scan_cache_invalidation(tmp_path, monkeypatch):
    a = np.array([[0., 16, np.nan], [100., 132, np.inf]])
    params = {"slices": [[1, 0, .1], [2, 100, .1]]}
    assert codec.vcid_of(a, 0, packing=params) == codec.vcid_of(a.astype("f4"), -2, packing=params)
    b = a.copy(); b[1, 0] += 2
    assert codec.vcid_of(a, 0, packing=params) != codec.vcid_of(b, 0, packing=params)
    with pytest.raises(ValueError, match="per slice"):
        codec.vcid_of(a, 0, packing={"slices": params["slices"][:1]})
    with pytest.raises(ValueError, match="packing"):
        codec.vcid_of(a, 0, packing={"slices": [[True, 0, .1], [2, 100, .1]]})
    times, values, _ = samples(2)
    path = write(tmp_path / "cache.zarr", values - values[:, :1, :1], times, (1, 2, 3))
    monkeypatch.setenv("ZT_HASH_CACHE", str(tmp_path / "shared"))
    first = scan(path, tmp_path / "scan", packing={"v": [1, 0, .1]})
    second = scan(path, tmp_path / "scan", packing={"v": [2, 0, .1]})
    m1, m2 = next(iter(first["subgrids"].values())), next(iter(second["subgrids"].values()))
    keys = [k for k in m1["chunks"] if k.startswith("v@")]
    assert keys and all(m1["chunks"][k][0] == m2["chunks"][k][0] for k in keys)
    assert all(m1["chunks"][k][1] != m2["chunks"][k][1] for k in keys)
    with pytest.raises(ValueError, match="budget"):
        scan(path, tmp_path / "scan", packing={"v": [1, .4, .1]})


def test_cli_scan_reports_source_contract_subgrids(tmp_path, capsys):
    from zarr_torrent.cli import main
    times, values, spec = samples(2)
    path = write(tmp_path / "scan.zarr", values, times, (1, 3, 5))
    params = tmp_path / "packing.json"; params.write_text(json.dumps(spec))
    main(["scan", path, "--packing", str(params)])
    result = json.loads(capsys.readouterr().out)
    assert result["link"] == "zt://" + "+".join(sorted(result["grids"]))
    grid = next(iter(result["grids"].values()))
    assert grid["arrays"]["v"]["source_packing"] and grid["chunks"] == 4


def test_native_dtypes_signed_mirror_transfer_and_restart(net):
    nodes, run, tmp = net
    source, mirror, client = nodes["h1"], nodes["h2"], nodes["client"]
    times, values, spec = samples()
    p1 = write(tmp / "source.zarr", values, times, (4, 2, 3))
    p2 = write(tmp / "mirror.zarr", (values + .025).astype("f4"), times, (4, 2, 3))
    result = http(ctl(source), "POST", "/api/seed", {"path": p1, "packing": spec})
    contract = result["packing"]["v"]
    assert check_signed(contract)
    mirror.trusted.add(source.ident.id); client.trusted.add(source.ident.id)
    http(ctl(mirror), "POST", "/api/seed", {"path": p2, "packing": {"v": contract}})
    grid = result["link"].removeprefix("zt://")
    view = run(client.view(grid, refresh=True))
    keys = [k for k in view["best"] if k.startswith("v@")]
    assert len(keys) == 16 and all(len(view["best"][k]["src"]) == 2 for k in keys)
    # Original publisher is no longer available; its detached signature is carried by the mirror.
    run(source.stop())
    got = zt.open_dataset(result["link"], ctl=ctl(client)).v.values
    assert np.all(np.abs(got - values) < .1)
    run(client.stop()); run(mirror.stop())
    run(mirror.start()); run(client.start())
    assert mirror.local[grid]["arrays"]["v"]["packing"] == contract
    assert client.local[grid]["arrays"]["v"]["packing"] == contract
    assert not client.local[grid].get("_seeded_keys")  # downloaded cache is not a value authority
    got2 = zt.open_dataset(result["link"], ctl=ctl(client)).v.values
    np.testing.assert_array_equal(got2, got)
    run(source.start())  # fixture owns lifecycle


def test_contract_authentication_and_family_downgrade(net):
    nodes, run, tmp = net
    source, mirror, evil, client = (nodes[k] for k in ("h1", "h2", "evil", "client"))
    times, values, spec = samples(2)
    p = write(tmp / "source.zarr", values, times, (1, 3, 5))
    res = run(source.add_seed(p, packing=spec))
    grid = next(iter(res["subgrids"]))
    honest = source.local[grid]
    contract = honest["arrays"]["v"]["packing"]
    for change in ("signature", "units", "variable", "grid_id", "budget"):
        bad = copy.deepcopy(contract)
        if change == "signature": bad["sig"] = "0" * 128
        elif change == "budget": next(iter(bad["parameters"]["time"].values()))[2] = .45
        else: bad[change] = "wrong"
        with pytest.raises(ValueError):
            run(mirror.add_seed(p, packing={"v": bad}))
    invalid = copy.deepcopy(honest)
    invalid["arrays"]["v"].pop("packing")  # claim the same vfid while omitting its contract
    weakened = copy.deepcopy(honest)
    weak = copy.deepcopy(contract); weak.pop("sig"); weak.pop("pk")
    next(iter(weak["parameters"]["time"].values()))[2] = .45
    weakened["arrays"]["v"]["packing"] = evil.ident.signed(weak)
    default = next(iter(scan(p)["subgrids"].values()))
    mans = {source.ident.id: honest, "omit": invalid, evil.ident.id: weakened,
            "sybil1": default, "sybil2": default, "sybil3": default}
    view = merge_view(grid, mans, client.ident.id, {source.ident.id})
    assert view["arrays"]["v"]["packing"] == contract
    assert all([p for p, _, _ in b["src"]] == [source.ident.id]
               for k, b in view["best"].items() if k.startswith("v@"))
    untrusted = merge_view(grid, {"mirror": honest}, client.ident.id)
    assert "v" not in untrusted["arrays"]


def test_changed_source_code_and_bad_transcode_rejected(net, monkeypatch):
    nodes, run, tmp = net
    source, evil, client = (nodes[k] for k in ("h1", "evil", "client"))
    times, values, spec = samples(2)
    source_path = write(tmp / "good.zarr", values, times, (1, 3, 5))
    res = run(source.add_seed(source_path, packing=spec)); grid = next(iter(res["subgrids"]))
    honest = source.local[grid]
    bad = values.copy(); bad[0, 0, 0] += 2
    evil_path = write(tmp / "changed.zarr", bad, times, (1, 3, 5))
    run(evil.add_seed(evil_path, packing={"v": honest["arrays"]["v"]["packing"]}))
    changed = copy.deepcopy(evil.local[grid])
    key = sorted(k for k in honest["chunks"] if k.startswith("v@"))[0]
    changed["chunks"][key][1] = honest["chunks"][key][1]
    view = merge_view(grid, {source.ident.id: honest, evil.ident.id: changed}, client.ident.id, {source.ident.id})
    assert client._verify_store(view, key, evil.ident.id, Path(changed["files"][key]).read_bytes()) is None
    a = view["arrays"]["v"]; li = a["layouts"][key.split("@")[1].split("/")[0]]
    raw = Path(honest["files"][key]).read_bytes()
    good = codec.decode(li["docs"], raw)
    before = codec.encode
    def corrupt(docs, vals):
        changed = vals.copy(); changed.flat[0] += 2
        return before(docs, changed)
    monkeypatch.setattr(codec, "encode", corrupt)
    assert client._verify_store(view, key, source.ident.id, codec.xt1_encode(good)) is None


def test_growing_source_chunk_checks_known_prefix(net):
    nodes, run, tmp = net
    times, values, spec = samples(3)
    paths = {n: write(tmp / f"{n}.zarr", data, tm, (4, 2, 3)) for n, data, tm in (
        ("h1", values[:2], times[:2]), ("h2", (values + .025).astype("f4"), times),
        ("evil", values.copy(), times))}
    initial = next(iter(scan(paths["h1"], packing=spec)["subgrids"].values()))
    contract = nodes["boot"].ident.signed(initial["arrays"]["v"]["packing"])
    for n in paths:
        run(nodes[n].add_seed(paths[n], packing={"v": contract}))
    grid = next(g for g, m in nodes["h1"].local.items() if "v" in m["arrays"])
    client = nodes["client"]
    key = sorted(k for k in nodes["h1"].local[grid]["chunks"] if k.startswith("v@"))[0]
    def check(new):
        mans = {nodes[n].ident.id: nodes[n].local[grid] for n in ("h1", new)}
        view = merge_view(grid, mans, client.ident.id, {nodes["boot"].ident.id})
        view["addrs"] = {nodes[n].ident.id: nodes[n].addr for n in ("h1", new)}
        assert view["best"][key]["nv"] == 3 and view["best"][key]["rival"]["nv"] == 2
        raw = Path(nodes[new].local[grid]["files"][key]).read_bytes()
        stored = client._verify_store(view, key, nodes[new].ident.id, raw)
        assert stored is not None
        return run(client._rival_ok(view, grid, key, stored[0]))
    assert check("h2")  # different decoder, same original counts; storage padding is irrelevant
    arr = zarr.open_array(str(Path(paths["evil"]) / "v"), mode="r+")
    arr[0, 0, 0] += 2
    run(nodes["evil"].add_seed(paths["evil"]))
    assert not check("evil")


def test_contested_layout_falls_back_without_inventing_hourly_samples(net):
    nodes, run, tmp = net
    times, values, spec = samples(2)
    old = list(spec["v"]["time"].values())
    times = pd.date_range(times[0], periods=2, freq="6h")
    spec["v"]["time"] = {str(int(t.timestamp())): p for t, p in zip(times, old)}
    good = write(tmp / "good.zarr", values, times, (1, 3, 5))
    initial = next(iter(scan(good, packing=spec)["subgrids"].values()))
    contract = nodes["boot"].ident.signed(initial["arrays"]["v"]["packing"])
    changed = values.copy(); changed[:, 0, 0] += [2, 1]
    bad = write(tmp / "bad.zarr", changed, times, (1, 3, 5))
    alternate = write(tmp / "alt.zarr", (values + .025).astype("f4"), times, (2, 2, 3))
    for name, path in (("h1", good), ("evil", bad), ("h2", alternate)):
        run(nodes[name].add_seed(path, packing={"v": contract}))
    client = nodes["client"]; client.trusted.add(nodes["boot"].ident.id)
    grid = next(g for g, m in nodes["h1"].local.items() if "v" in m["arrays"])
    view = run(client.view(grid, refresh=True))
    same = next(l for l in view["arrays"]["v"]["layouts"] if l.startswith("1x3x5"))
    assert view["arrays"]["v"]["layouts"][same]["cov"] == []
    assert all(b.get("contested") for k, b in view["best"].items() if k.startswith(f"v@{same}/"))
    assert View(view).S == 6
    got = zt.open_dataset(f"zt://{grid}", ctl=ctl(client)).v.values
    assert got.shape == values.shape and np.max(np.abs(got - values)) < .1


def test_source_count_repair_across_layouts(net):
    nodes, run, tmp = net
    source, mirror, vol, client = (nodes[k] for k in ("h1", "h2", "evil", "client"))
    times, values, spec = samples(8)
    p1 = write(tmp / "complete.zarr", values, times, (1, 3, 5))
    res = http(ctl(source), "POST", "/api/seed", {"path": p1, "packing": spec})
    contract = res["packing"]["v"]; grid = res["link"].removeprefix("zt://")
    for n in (mirror, vol, client): n.trusted.add(source.ident.id)
    # Mirror has a different time/spatial layout, a different native dtype, and lacks the first two times.
    p2 = write(tmp / "partial.zarr", (values + .025).astype("f4"), times, (2, 2, 3))
    arr = zarr.open_array(str(Path(p2) / "v"), mode="r")
    for y, x in np.ndindex(2, 2):
        (Path(p2) / "v" / arr.metadata.encode_chunk_key((0, y, x))).unlink()
    http(ctl(mirror), "POST", "/api/seed", {"path": p2, "packing": {"v": contract}})
    view = run(vol.view(grid, refresh=True))
    lay = next(l for l in view["arrays"]["v"]["layouts"] if l.startswith("1x3x5"))
    made = http(ctl(vol), "POST", "/api/parity", {"grid": grid, "var": "v", "layout": lay,
                "kind": "v", "k": 4, "d": 1, "row": 0, "drop": True})
    assert made["stripes"] == made["code_stripes"] == 2
    # Two losses in the first stripe need a second independent RS row.
    nodes["boot"].trusted.add(source.ident.id)
    made2 = http(ctl(nodes["boot"]), "POST", "/api/parity", {"grid": grid, "var": "v", "layout": lay,
                 "kind": "v", "k": 4, "d": 1, "row": 1, "drop": True})
    assert made2["stripes"] == made2["code_stripes"] == 2
    run(source.stop())
    jid = http(ctl(client), "POST", "/api/download", {"grid": grid, "region": {"var": "v"}})["job"]
    job = wait_job(ctl(client), jid)
    assert job["state"] == "done" and job.get("restored") == 2, job
    got = zt.open_dataset(res["link"], ctl=ctl(client)).v.values
    assert np.all(np.abs(got - values) < .1)
    run(source.start())
