"""Regression cases for per-slice identity and request completeness."""
import json
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from zarr_torrent import codec, jlps, parity


def test_jlps_cached_bytes_do_not_prune_faster_cover():
    arrays = {"v": {"taxis": 0, "dims": ["time", "lat", "lon"], "layouts": {
        name: {"chunks": [n, 1000, 1000], "phase": 0,
               "docs": {".zarray": {"shape": [2, 1000, 1000]}}}
        for name, n in (("A", 2), ("B", 1))}}}
    best = {key: {"src": [(peer, "cid", size)], "nv": nv}
            for key, peer, size, nv in (("v@A/0.0.0", "fast", 1_000_000, 2),
                                       ("v@B/0.0.0", "local", 4_000_000, 1),
                                       ("v@B/1.0.0", "slow", 100_000, 1))}
    keys, _, elapsed, _ = jlps.jlps("v", arrays, best, 0, 2,
                                   {"fast": 60e6, "slow": 30e6, "local": 1e12})
    assert set(keys) == {"v@B/0.0.0", "v@B/1.0.0"}
    assert elapsed < .11


def test_jlps_includes_minimum_byte_candidate_without_price_iterations():
    # Even when candidate generation stops immediately, retain a usable reference cover.
    arrays = {"v": {"taxis": 0, "layouts": {
        "A": {"chunks": [1], "phase": 0}, "B": {"chunks": [2], "phase": 0}}}}
    best = {"v@A/0": {"src": [("slow", "a", 1e6)]},
            "v@A/1": {"src": [("slow", "b", 1e6)]},
            "v@B/0": {"src": [("fast", "c", 3e6)]}}
    keys, _, elapsed, _ = jlps.jlps("v", arrays, best, 0, 2,
                                   {"slow": 1e6, "fast": 30e6}, iters=0)
    assert set(keys) == set(jlps.bytes_greedy_cover("v", arrays, best, 0, 2))
    assert elapsed > 0


def test_assignment_fallback_reports_relay_receiver_and_local_costs():
    from zarr_torrent.plan import plan

    chunks = {"one": (10, ("p",))}
    _, elapsed = plan(chunks, {"p": 10}, max_groups=0,
                      via={"p": "r"}, via_bw={"r": 1}, client_bw=2)
    assert elapsed == 10  # relay, rather than the unconstrained one-second peer transfer
    assert plan(chunks, {"p": 10}, max_groups=0, client_bw=2)[1] == 5
    assert plan(chunks, {"p": 10}, max_groups=0, client_bw=2, local={"p"})[1] == 1


def test_byte_oracle_does_not_censor_complete_candidate_by_partial_time():
    from sim.e_hetero import candidate_deadline

    assert candidate_deadline(None) == 1800
    assert candidate_deadline({"coverage": .958, "state": "partial", "seconds": 10}) == 1800
    assert candidate_deadline({"coverage": 1, "state": "partial", "seconds": 10}) == 1800
    assert candidate_deadline({"coverage": 1, "state": "done", "seconds": 10}) == 120
    assert candidate_deadline({"coverage": 1, "state": "done", "seconds": 60}) == 180


def test_terminal_job_state_waits_for_coverage():
    from zarr_torrent.node import Node

    async def check():
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        key = "v@A/0"
        view = {"arrays": {"v": {"taxis": 0, "dims": ["time"], "layouts": {
            "A": {"chunks": [1], "phase": 0, "docs": {".zarray": {"shape": [1]}}}}}},
            "best": {key: {"src": [("p", "cid", 1)], "nv": 1}}}

        async def fetch(*args, **kwargs):
            return {key: "downloaded"}

        async def slow_view(*args):
            entered.set()
            await release.wait()
            return view

        async def announce(*args):
            finished.set()

        job = {"id": "j", "grid": "g", "cover": {"lattice": [1, 0], "request": {
            "var": "v", "g_lo": 0, "g_hi": 1, "isel": {}}}}
        node = SimpleNamespace(local={}, views={}, _job_keys={"j": ([key], None)},
                               fetch=fetch, view=slow_view, announce=announce, _flush=lambda: None)
        Node._launch(node, job)
        await entered.wait()
        assert job["state"] == "running" and job["t1"] is None
        release.set()
        await finished.wait()
        assert job["state"] == "done" and job["t1"] is not None
        assert job["coverage"] == {"requested_samples": 1, "covered_samples": 1, "missing_samples": 0}

    asyncio.run(check())


@pytest.mark.parametrize("late_manifest", [False, True])
def test_incomplete_cover_refreshes_catalogue_once(late_manifest):
    from zarr_torrent.node import Node

    async def check():
        layout = {"chunks": [1], "phase": 0, "docs": {".zarray": {"shape": [2]}}}
        arrays = {"v": {"taxis": 0, "dims": ["time"], "layouts": {"A": layout}}}
        initial = {"grid_id": "g", "arrays": arrays, "best": {
            "v@A/0": {"src": [("p", "cid", 1)], "nv": 1}}}
        fresh = dict(initial, best=dict(initial["best"]))
        if late_manifest:
            fresh["best"]["v@A/1"] = {"src": [("p", "cid2", 1)], "nv": 1}
        refreshes = []

        async def view(grid, refresh=False):
            refreshes.append((grid, refresh))
            return fresh

        async def cover_core(v, *args):
            return list(v["best"]), {"cover": "bytes"}

        node = SimpleNamespace(view=view, _cover_core=cover_core)
        node._cover = lambda *args, **kwargs: Node._cover(node, *args, **kwargs)
        keys, info = await node._cover(initial, "v", 0, 2, None, "bytes")
        assert refreshes == [("g", True)]
        assert info["_view"] is fresh and info["catalogue_refresh"] == 1
        coverage = jlps.region_coverage("v", arrays, fresh["best"], keys, 0, 2)
        assert coverage["covered_samples"] == (2 if late_manifest else 1)
        assert coverage["missing_samples"] == (0 if late_manifest else 1)
        refreshes.clear()
        if late_manifest:
            await node._cover(fresh, "v", 0, 2, None, "bytes")
            assert not refreshes  # complete plans do not incur another catalogue round

    asyncio.run(check())


def test_opposing_slice_changes_cannot_cancel():
    a = np.tile(np.arange(16, dtype="f4"), (2, 1))
    for axis in (0, 1, -1):
        x = a if axis == 0 else a.T
        y = (a + np.array([[1], [-1]], dtype="f4")) if axis == 0 else (a + [[1], [-1]]).T
        assert not codec.same_vcid(codec.vcid_of(x, axis), codec.vcid_of(y, axis))
    b = a.copy()
    b[0] *= 0.5
    b[1] *= 1.5
    assert not codec.same_vcid(codec.vcid_of(a, 0), codec.vcid_of(b, 0))
    assert not codec.same_vcid(codec.vcid_of(a, 0), codec.vcid_of(a, 1))


def test_nonfinite_classes_constants_and_malformed_ids():
    a = np.array([[0, 1, np.nan], [4, 4, np.inf]], dtype="f4")
    va = codec.vcid_of(a, 0)
    assert codec.same_vcid(va, codec.vcid_of(a.astype("f8"), 0))
    for replacement in (np.inf, -np.inf, 0):
        b = a.copy()
        b[0, 2] = replacement
        assert not codec.same_vcid(va, codec.vcid_of(b, 0))
    b = a.copy()
    b[1, 2] = -np.inf
    assert not codec.same_vcid(va, codec.vcid_of(b, 0))
    for bad in ("L:old:0:1", "L3:broken:[]", "L3:" + "a" * 32 + ":[[NaN,1]]",
                "L3:" + "a" * 32 + ":[[0,-1]]", "L3:" + "a" * 32 + ":{}"):
        assert not codec.same_vcid(bad, bad)
    codes, params = codec.lattice_codes(np.tile(np.arange(16, dtype="f4"), (2, 1)))
    with pytest.raises(ValueError, match="per slice"):
        codec.from_lattice_codes(codes, params[:-1], "f4")


@pytest.mark.parametrize("bits", [[0x80000000, 0x3f800000, 0x40000000],
                                  [0x00000000, 0x3f800000, 0x7fc00001]])
def test_exact_lattice_codes_preserve_zero_sign_and_nan_payload(bits, monkeypatch):
    monkeypatch.setattr(codec, "VALUE_ID", "lattice")
    values = np.array(bits, dtype="u4").view("f4")
    codes, params = codec.lattice_codes(values, exact=False)
    rebuilt = codec.from_lattice_codes(codes, params, values.dtype)
    assert np.array_equal(values, rebuilt, equal_nan=True)
    assert values.tobytes() != rebuilt.tobytes()
    assert codec.lattice_codes(values) is None  # use raw-value parity for these bits


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", ">f8", "<i4", ">i4", "<u8", ">u8"])
def test_transport_roundtrip_preserves_bits_and_byte_order(dtype):
    values = np.arange(12).reshape(3, 4).astype(dtype)
    if values.dtype.kind == "f":
        values[0, 0], values[0, 1], values[0, 2] = -0.0, np.inf, -np.inf
        unsigned = np.dtype(dtype.replace("f", "u"))
        values.view(unsigned)[0, 3] = 0x7fc00001 if values.dtype.itemsize == 4 else 0x7ff8000000000001
    values = values[:, ::-1]  # transport accepts noncontiguous decoded arrays too
    canonical = values.astype(values.dtype.newbyteorder("<"))
    rebuilt = codec.xt1_decode(codec.xt1_encode(values))
    assert rebuilt.dtype == canonical.dtype
    assert rebuilt.tobytes() == canonical.tobytes()


def test_decoded_cache_separates_decoder_versions_for_identical_bytes():
    from zarr_torrent.common import cid_of
    from zarr_torrent.pushdown import DecodedLRU
    raw = np.array([1.0], dtype="<f4").tobytes()
    meta = {"zarr_format": 2, "shape": [1], "chunks": [1], "dtype": "<f4",
            "fill_value": 0, "compressor": None, "filters": None, "order": "C"}
    cache = DecodedLRU()
    floats = cache.get(cid_of(raw), {".zarray": meta}, lambda: raw)
    integers = cache.get(cid_of(raw), {".zarray": dict(meta, dtype="<i4")}, lambda: raw)
    np.testing.assert_array_equal(floats, [1.0])
    assert integers.dtype == np.dtype("<i4")
    np.testing.assert_array_equal(integers, np.frombuffer(raw, dtype="<i4"))


def test_growing_integer_chunk_preserves_valid_prefix_not_physical_padding(tmp_path):
    from zarr_torrent.common import cid_of
    from zarr_torrent.node import Node
    node = Node(tmp_path / "node")
    old = np.zeros((4, 4), dtype="<i4")
    old[:2] = np.arange(8).reshape(2, 4)
    new = old.copy()
    new[2] = 100
    docs = {".zarray": {"zarr_format": 2, "shape": [4, 4], "chunks": [4, 4], "dtype": "<i4",
                         "fill_value": 0, "compressor": None, "filters": None, "order": "C"}}
    def put(values):
        raw = codec.encode(docs, values)
        cid = cid_of(raw)
        path = node._cas(cid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return cid, path
    old_cid, old_path = put(old)
    new_cid, _ = put(new)
    key = "v@4x4/0.0"
    arrays = {"v": {"taxis": 0, "layouts": {"4x4": {"docs": docs}}}}
    node.local["grid"] = {"files": {key: str(old_path)}}
    view = {"arrays": arrays, "pinfo": {node.ident.id: arrays}, "best": {key: {"rival": {
        "vcid": codec.vcid_of(old, 0), "nv": 2, "src": [(node.ident.id, old_cid, 0)]}}}}
    assert asyncio.run(node._rival_ok(view, "grid", key, new_cid)), "padding is outside the declared prefix"
    new[0, 0] += 1
    wrong_cid, _ = put(new)
    assert not asyncio.run(node._rival_ok(view, "grid", key, wrong_cid)), "a known integer value changed"


def test_prefix_tolerance_is_per_slice_and_preserves_known_nonfinite_cells():
    a = np.array([[0., 1., 2.], [0., .01, .02]])
    b = a.copy()
    b[1] += .1
    assert not codec.agrees_with_prefix(b, a, 0)
    assert codec.agrees_with_prefix(a.copy(), a, 0)
    a[1, 1] = np.inf
    b = a.copy()
    b[1, 1] = np.nan
    assert not codec.agrees_with_prefix(b, a, 0)


def test_coverage_requires_all_spatial_tiles_and_valid_samples():
    li = {"chunks": [2, 2], "phase": 0, "docs": {".zarray": {"shape": [4, 4]}}}
    arrays = {"v": {"taxis": 0, "dims": ["time", "x"], "layouts": {"A": li}}}
    best = {f"v@A/{t}.{x}": {"src": [("p", "cid", 1)], "nv": 2}
            for t in range(2) for x in range(2)}
    cov = lambda keys: jlps.region_coverage("v", arrays, best, keys, -1, 5)
    assert cov(list(best)) == {"requested_samples": 6, "covered_samples": 4, "missing_samples": 2}
    missing_tile = [k for k in best if k != "v@A/1.1"]
    assert cov(missing_tile)["covered_samples"] == 2
    best["v@A/1.1"]["nv"] = 1
    assert cov(list(best))["covered_samples"] == 3
    assert jlps.region_coverage("v", arrays, best, ["v@A/0.0", "v@A/1.0"], 0, 4,
                               {"x": [0, 1]})["covered_samples"] == 4


def test_step_comparison_is_symmetric():
    digest = "a" * 32
    a = f"L3:{digest}:" + json.dumps([[.25, 1., 0]])
    b = f"L3:{digest}:" + json.dumps([[0., 1. + 1e-6, 0]])
    assert codec.same_vcid(a, b) == codec.same_vcid(b, a)


def test_parity_rejects_inconsistent_stripe_headers():
    data = [b"abc", b"def", b"ghi"]
    members = [(str(i), str(i), 1, b) for i, b in enumerate(data)]
    a = parity.encode(members, row=0)
    members[0] = ("other", "other", 1, data[0])
    b = parity.encode(members, row=1)
    with pytest.raises(ValueError, match="stripe"):
        parity.restore_many([a, b], {2: data[2]}, [0, 1])


def test_step_error_cannot_accumulate_over_a_large_code_range():
    a = np.arange(-300000, 300001, dtype="f8")
    b = a * (1 + 1e-6)
    assert not codec.same_vcid(codec.vcid_of(a), codec.vcid_of(b))


def test_trusted_packing_preserves_sparse_source_codes(monkeypatch):
    monkeypatch.setattr(codec, "VALUE_ID", "lattice")
    inferred_accepts = 0
    for stride in (1, 2, 4, 8, 16):
        for size in (2, 4, 16, 256):
            a = np.arange(size, dtype="f4") * stride
            changed = a.copy()
            changed[0] += 1
            inferred_accepts += codec.same_vcid(codec.vcid_of(a), codec.vcid_of(changed))
            va = codec.vcid_of(a, packing=(1, 0, .45))
            assert not codec.same_vcid(va, codec.vcid_of(changed, packing=(1, 0, .45)))
            decoded = a.astype("f8") + .4 * np.cos(a.astype("f8"))
            assert codec.same_vcid(va, codec.vcid_of(decoded, packing=(1, 0, .45)))
    assert inferred_accepts == 13  # the new input closes the sparse-code ambiguity; the old estimator is retained


def test_trusted_packing_fails_closed_and_preserves_masks(monkeypatch):
    monkeypatch.setattr(codec, "VALUE_ID", "lattice")
    a = np.array([[0, 16, np.nan], [4, np.inf, -np.inf]], dtype="f4")
    va = codec.vcid_of(a, 0, packing=(1, 0, .45))
    assert codec.same_vcid(va, codec.vcid_of(a.astype(">f8"), 0, packing=(1, 0, .1)))
    assert va != codec.vcid_of(a, 1, packing=(1, 0, .45))
    assert va != codec.vcid_of(a, 0, packing=(1, 1, .45))
    b = a.copy(); b[1, 1] = -np.inf
    assert va != codec.vcid_of(b, 0, packing=(1, 0, .45))
    for bad in ((0, 0, .1), (1, 0, 0), (1, 0, -.1), (1, 0, .5), (1, np.inf, .1),
                (1, 0, np.nan), (1, 0), (1, 0, .1, 2)):
        with pytest.raises(ValueError, match="packing"):
            codec.vcid_of(a, packing=bad)
    with pytest.raises(ValueError, match="budget"):
        codec.vcid_of(np.array([.46, 16]), packing=(1, 0, .45))
    with pytest.raises(ValueError, match="precision"):
        codec.vcid_of(np.array([2 ** 52], dtype="f8"), packing=(1, 0, .45))
    with pytest.raises(ValueError, match="precision"):
        codec.vcid_of(np.array([1e16]), packing=(1, 1e16, .45))
    with pytest.raises(ValueError, match="time axis"):
        codec.vcid_of(a, 2, packing=(1, 0, .45))
    monkeypatch.setattr(codec, "VALUE_ID", "exact")
    with pytest.raises(ValueError, match="lattice mode"):
        codec.vcid_of(a, packing=(1, 0, .45))


def test_benchmark_checks_only_downloaded_payloads_and_excludes_edge_padding(tmp_path, monkeypatch):
    import pandas as pd
    import xarray as xr
    from sim import e_hetero
    from zarr_torrent.scan import scan
    path = tmp_path / "truth.zarr"
    values = np.arange(36, dtype="f4").reshape(4, 3, 3)
    xr.Dataset({e_hetero.VAR: (("time", "y", "x"), values)},
               coords={"time": pd.date_range("2020-01-01", periods=4, freq="h"),
                       "y": np.arange(3), "x": np.arange(3)}).to_zarr(
        path, encoding={e_hetero.VAR: {"chunks": (3, 2, 2)}}, consolidated=False, zarr_format=2)
    sg = next(iter(scan(path)["subgrids"].values()))
    view = {k: sg[k] for k in ("grid", "arrays")}  # public /api/view deliberately omits per-chunk voting data
    def read_local(ctl, method, route, body, raw):
        assert method == "POST" and route == "/api/read" and raw
        return Path(sg["files"][body["key"]]).read_bytes()
    from pathlib import Path
    monkeypatch.setattr(e_hetero, "http", read_local)
    e_hetero.verify_downloaded_chunks("local", sg["grid_id"], view, list(sg["chunks"]), path)
