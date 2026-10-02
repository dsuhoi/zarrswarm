"""Scaling of manifest encode/decode + merge_view with many chunks and peers (no network)."""
import json, os, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zarr_torrent.common import cjson
from zarr_torrent.node import merge_view

def manifest(node, n_time, nvars, lay="24x181x360+0", start=0):
    arrays = {f"v{i}": {"vfid": f"vf{i}", "dims": ["time", "lat", "lon"], "taxis": 0, "fmt": 3, "attrs": {},
                        "layouts": {lay: {"fid": f"f{i}", "docs": {"zarr.json": {"shape": [1, 181, 360]}},
                                          "chunks": [24, 181, 360], "phase": 0}}} for i in range(nvars)}
    chunks = {f"v{i}@{lay}/{c}.0.0": [os.urandom(16).hex(), os.urandom(16).hex(), 250000, 24]
              for i in range(nvars) for c in range(start, start + n_time)}
    return {"node": node, "grid": {"time": {"dt": 3600, "rphase": 0}}, "gdocs": {}, "arrays": arrays, "chunks": chunks}

for n_time, nvars, peers in ((14600, 5, 4), (14600, 10, 8)):
    mans = {f"p{j}": manifest(f"p{j}", n_time, nvars, start=j * 1000) for j in range(peers)}
    t = time.perf_counter(); body = cjson(mans["p0"]); comp = zlib.compress(body, 3); te = time.perf_counter() - t
    t = time.perf_counter(); json.loads(zlib.decompress(comp)); td = time.perf_counter() - t
    t = time.perf_counter(); v = merge_view("g", mans, "p0"); tm = time.perf_counter() - t
    print(f"chunks/peer={n_time*nvars:>7} peers={peers}: manifest {len(body)/1e6:.1f} MB raw {len(comp)/1e6:.1f} MB zlib, "
          f"encode {te:.2f}s decode {td:.2f}s, merge_view {tm:.2f}s, best={len(v['best'])}")
