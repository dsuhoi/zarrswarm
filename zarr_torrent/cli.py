"""zt - command line client.

  zt init --bootstrap-node --public-host HOST   # first node of a network: prints ztnet://ID@HOST:PORT
  zt init --join ztnet://ID@HOST:PORT           # any other node: address (public/relay) is found automatically
  zt node                                       # run with ~/.zt/config.toml (flags override)
  zt scan   PATH                                # offline: grid id, variables, layouts, chunks
  zt seed   PATH                                # seed via the local node -> zt://<grid>
  zt search TAG                                 # metadata index: variables / standard_name / long_name
  zt get    LINK [--vars a,b] [--time A:B] [--out out.zarr] [--chunking time=8760,lat=1] [--progressive]
  zt mean   LINK VAR [--time A:B] [--rel-err 0.01]
  zt name   NAME LINK                           # signed mutable name -> zt://NAME@<pubkey>
  zt follow LINK --vars t2m --last 7d           # subscription: keep the newest 7 days downloaded
  zt status | peers LINK | unseed PATH
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .store import CTL, http, open_dataset, open_views, progressive_mean, progressive_mean_vas, step_seconds, wait_job

DEFAULT_PORT = 7881


def _mb(b):
    return f"{b / 1e6:,.1f} MB"


def _home(a) -> Path:
    return Path(getattr(a, "home", None) or os.environ.get("ZT_HOME", "~/.zt")).expanduser()


TUNING = {  # config [tuning] key -> env variable read by node/store at import (docs/config.md)
    "announce_every_s": "ZT_ANNOUNCE_EVERY", "page_cache_mb": "ZT_PAGE_CACHE_MB", "decoded_cache_mb": "ZT_DECODED_MB",
    "client_cache_mb": "ZT_DECODED_CACHE_MB", "audit_rate": "ZT_AUDIT", "pushdown_frac": "ZT_PUSHDOWN_FRAC",
    "readahead_chunks": "ZT_READAHEAD", "stall_min_s": "ZT_STALL_MIN", "xt1_below_bps": "ZT_XT1_BELOW",
    "relay_frame_kb": "ZT_RELAY_FRAME_KB", "rescan_every_s": "ZT_RESCAN_EVERY", "follow_every_s": "ZT_FOLLOW_EVERY"}


def _config(home: Path) -> dict:
    """config.toml (written by `zt init`, commented) or legacy config.json."""
    import tomllib
    t, j = home / "config.toml", home / "config.json"
    if t.exists():
        return tomllib.loads(t.read_text())
    return json.loads(j.read_text()) if j.exists() else {}


def _toml_val(v) -> str:
    return json.dumps(v) if v is not None else '""'  # JSON strings/lists/numbers/bools are valid TOML values


def _write_config(home: Path, cfg: dict):
    """Commented config.toml; unknown keys already in the file are lost, so edit the file, not re-init."""
    v = lambda k, d=None: _toml_val(cfg.get(k, d))
    seeds = "".join(f'\n[[seed]]\npath = {_toml_val(x["path"])}\n' for x in cfg.get("seed", []))
    tuning = "".join(f"{k} = {_toml_val(x)}\n" for k, x in cfg.get("tuning", {}).items())
    path = home / "config.toml"
    path.touch(mode=0o600)
    path.chmod(0o600)  # may hold the network key
    path.write_text(f"""# zarr-torrent node configuration (docs/config.md). Command-line flags override these values.

# network this node belongs to (printed by `zt init`, give it to other hosts)
network = {v("network", "")}
# closed network: shared key every request must carry ("" = open network). SECRET: it is also in the invite.
network_key = {v("network_key", "")}

# data port (peers connect here) and bind address: "0.0.0.0" = reachable from outside, "127.0.0.1" = outbound only
port = {v("port")}
host = {v("host", "127.0.0.1")}
# control API (CLI/TUI/xarray talk to it), always bound to 127.0.0.1; default port + 1
{f"ctl_port = {cfg['ctl_port']}" if cfg.get("ctl_port") else f"# ctl_port = {cfg.get('port', DEFAULT_PORT) + 1}"}

# externally reachable URL of the data port ("" = unknown / behind NAT)
public = {v("public")}
# bootstrap nodes (any public node of the network)
bootstrap = {v("bootstrap", [])}
# relay to attach to when not publicly reachable ("" = none)
relay = {v("relay")}
# relay traffic for NAT'd peers (public nodes)
relay_server = {v("relay_server", False)}
# detect public reachability through the bootstrap node at start
auto = {v("auto", False)}

# upload cap in MB/s (0 = unlimited): shared by all peers downloading from this node
upload_mbps = {v("upload_mbps", 0)}

# downloaded data kept in the node cache, GB (0 = unlimited): least recently used chunks are evicted first;
# your own seeded files and volunteer parity are never touched
cache_max_gb = {v("cache_max_gb", 0)}

# publisher public keys whose values win over the holder majority
trust = {v("trust", [])}

# datasets to seed at every start (paths or globs; `zt seed` additions are remembered in state.json separately)
# [[seed]]
# path = "/data/era5/*.zarr"
{seeds}
# advanced tuning (docs/config.md); every key is optional
[tuning]
{tuning}""")


def parse_ztnet(s: str) -> tuple[str | None, str, str | None]:
    """ztnet://<node_id>@host:port[?k=<network key>] -> (node_id, http://host:port, key)"""
    s, _, q = s.removeprefix("ztnet://").partition("?")
    key = next((v for k, _, v in (x.partition("=") for x in q.split("&")) if k == "k"), None) or None
    nid, _, hp = s.rpartition("@")
    if ":" not in hp:
        hp = f"{hp}:{DEFAULT_PORT}"
    return nid or None, f"http://{hp}", key


def _progress(job):
    tot = max(job["total"], 1)
    el = (job["t1"] or time.time()) - job["t0"]
    sys.stderr.write(f"\r{job['done']}/{job['total']} chunks {100 * job['done'] / tot:5.1f}%  "
                     f"{_mb(job['bytes'])}  {job['bytes'] / 1e6 / max(el, 1e-3):6.1f} MB/s  "
                     f"peers={len(job['per_peer'])}  missing={job['missing']}   ")
    sys.stderr.flush()


def cmd_init(a):
    from .common import Identity
    home = _home(a)
    ident = Identity(home)
    cfg = _config(home)
    cfg["port"] = a.port or cfg.get("port", DEFAULT_PORT)
    if a.bootstrap_node:
        if not a.public_host:
            sys.exit("--bootstrap-node needs --public-host (DNS name or IP reachable by other nodes)")
        cfg.update(host="0.0.0.0", public=f"http://{a.public_host}:{cfg['port']}", relay_server=True,
                   bootstrap=[], auto=False, relay=None)
        if a.private and not cfg.get("network_key"):
            import secrets
            cfg["network_key"] = secrets.token_urlsafe(24)
        k = f"?k={cfg['network_key']}" if cfg.get("network_key") else ""
        cfg["network"] = f"ztnet://{ident.id}@{a.public_host}:{cfg['port']}{k}"
    elif a.join:
        _, boot, key = parse_ztnet(a.join)
        cfg["network_key"] = key or ""
        cfg.update(host="0.0.0.0" if a.listen_public else "127.0.0.1", bootstrap=[boot], network=a.join,
                   auto=a.listen_public, relay=None if a.listen_public else boot, public=None, relay_server=False)
    else:
        sys.exit("use --bootstrap-node --public-host HOST  or  --join ztnet://ID@HOST:PORT")
    _write_config(home, cfg)
    print(f"node id   {ident.id}\nconfig    {home / 'config.toml'}\nnetwork   {cfg['network']}")
    if a.service:
        unit = Path("~/.config/systemd/user/zt-node.service").expanduser()
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(f"[Unit]\nDescription=zarr-torrent node\nAfter=network-online.target\n\n[Service]\n"
                        f"ExecStart={sys.executable} -m zarr_torrent.cli node --home {home}\nRestart=always\n"
                        f"RestartSec=5\n\n[Install]\nWantedBy=default.target\n")
        print(f"service   {unit}  (systemctl --user enable --now zt-node)")


def cmd_node(a):
    home = _home(a)
    cfg = _config(home)
    for k, x in cfg.get("tuning", {}).items():  # before importing node: its constants are read at import
        if k not in TUNING:
            sys.exit(f"unknown [tuning] key {k!r}; known: {', '.join(TUNING)}")
        os.environ.setdefault(TUNING[k], str(x))
    from .node import Node
    pick = lambda k, d=None: getattr(a, k) if getattr(a, k) not in (None, False, []) else (cfg.get(k) or d)
    mbps = a.upload_mbps if a.upload_mbps is not None else cfg.get("upload_mbps", 0)
    node = Node(home=home, host=pick("host", "127.0.0.1"), port=pick("port", DEFAULT_PORT),
                ctl_port=a.ctl_port or cfg.get("ctl_port"), public=pick("public"),
                bootstrap=pick("bootstrap", []) or [], relay=pick("relay"),
                relay_server=pick("relay_server", False), auto=pick("auto", False), trust=pick("trust", []) or [],
                rate=mbps * 1e6 if mbps else None, seeds=[x["path"] for x in cfg.get("seed", [])],
                network_key=os.environ.get("ZT_NETWORK_KEY") or cfg.get("network_key") or None,
                cache_max=int(float(cfg.get("cache_max_gb") or 0) * 1e9),
                # link emulation for experiments (not in the config on purpose)
                latency=float(os.environ.get("ZT_EMU_LATENCY_MS", 0)) / 1e3, loss=float(os.environ.get("ZT_EMU_LOSS", 0)),
                strategy=os.environ.get("ZT_EMU_STRATEGY", "maxflow"))

    async def run():
        await node.start()
        print(f"zt node {node.ident.id} addr={node.addr} ro={node.ro} ctl=127.0.0.1:{node.ctl_port} "
              f"contacts={len(node.dht.contacts())}", flush=True)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, stop.set)
        await stop.wait()
        await node.stop()
    asyncio.run(run())


def cmd_scan(a):
    from .scan import scan
    r = scan(a.path)
    print(json.dumps({"link": f"zt://{r['grid_id']}", "time": r["grid"]["time"], "dims": r["grid"]["dims"],
                      "arrays": {n: {"dims": x["dims"], "vfid": x["vfid"][:12], "layouts": list(x["layouts"])}
                                 for n, x in r["arrays"].items()},
                      "chunks": len(r["chunks"]), "bytes": sum(c[2] for c in r["chunks"].values())}, indent=1))


def cmd_seed(a):
    r = http(a.ctl, "POST", "/api/seed", {"path": os.path.abspath(a.path)})
    print(f"{r['link']}  arrays={','.join(r['arrays'])}  chunks={r['chunks']}  subgrids={len(r['grids'])}")


def cmd_unseed(a):
    http(a.ctl, "POST", "/api/unseed", {"path": os.path.abspath(a.path)})


def cmd_status(a):
    s = http(a.ctl, "GET", "/api/status")
    print(f"node {s['id']}  addr={s['addr']}  ro={s['ro']}  contacts={s['contacts']}  relayed={s['relayed']}")
    print(f"pubkey {s['pk']}   (publisher key: others put it into trust = [...])")
    for g, x in s["grids"].items():
        print(f"  zt://{g}  {x['chunks']} chunks  {_mb(x['bytes'])}  {','.join(x['arrays'])}")
    for j in s["jobs"]:
        print(f"  job {j['id']} {j['state']} {j['done']}/{j['total']} {_mb(j['bytes'])} {j['label']}")


def cmd_search(a):
    fmt = lambda s: datetime.fromtimestamp(s, timezone.utc).strftime("%Y-%m-%d %H:%M")
    for r in http(a.ctl, "GET", "/api/search?tag=" + urllib.parse.quote(a.tag)):
        tr = f"{fmt(r['tr'][0])} .. {fmt(r['tr'][1])}" if r.get("tr") else "-"
        print(f"zt://{r['grid']}  seeders={r['seeders']}  vars={','.join(r['vars'])}  time={tr}  dims={r['dims']}")


def cmd_peers(a):
    g = http(a.ctl, "GET", "/api/resolve?link=" + urllib.parse.quote(a.link))["grid"]
    for p in http(a.ctl, "GET", f"/api/peers/{g}")["peers"]:
        print(f"{p['id'][:12]}  {p['chunks']:>8} chunks  bw={p['bw'] / 1e6:.1f} MB/s  {p['addr']}")


def _time(s):
    if not s:
        return None, None
    t0, _, t1 = s.partition(":") if s.count(":") == 1 else s.partition("/")
    return t0 or None, t1 or None


def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def _chunking(s):
    if not s:
        return None
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}


def _isel_from_sel(link, ctl, sel: str | None) -> dict:
    """'lat=40:60,lon=0:30' (coordinate values, either order) -> {dim: [i0, i1)} via the lazy union dataset."""
    if not sel:
        return {}
    import numpy as np
    ds = open_dataset(link, ctl=ctl)
    out = {}
    for part in sel.split(","):
        d, _, rng = part.partition("=")
        lo, _, hi = rng.partition(":")
        vals = np.asarray(ds[d].values)
        lo_v = float(lo) if lo else -np.inf
        hi_v = float(hi) if hi else np.inf
        idx = np.nonzero((vals >= min(lo_v, hi_v)) & (vals <= max(lo_v, hi_v)))[0]
        if not len(idx):
            sys.exit(f"--sel {part}: no {d} values in range")
        out[d] = [int(idx[0]), int(idx[-1]) + 1]
    return out


def cmd_get(a):
    t0, t1 = _time(a.time)
    want = a.vars.split(",") if a.vars else None
    isel = _isel_from_sel(a.link, a.ctl, a.sel)
    for grid, view in open_views(a.link, a.ctl):  # every sub-grid of a composite link
        vars_ = [n for n in view.arrays if n not in view.v["grid"]["dims"] and (not want or n in want)]
        for var in vars_:  # server-side cover selection per variable (JLPS: joint layout + peer choice)
            jid = http(a.ctl, "POST", "/api/download", {"grid": grid, "cover": a.cover,
                                                        "region": {"var": var, "t0": t0, "t1": t1, "isel": isel,
                                                                   "step": step_seconds(a.step)},
                                                        "order": "progressive" if a.progressive else "optimal",
                                                        "label": f"get {var} {a.time or ''}"})["job"]
            job = wait_job(a.ctl, jid, _progress)
            sys.stderr.write("\n")
            print(json.dumps({"var": var, "state": job["state"], "chunks": job["done"], "missing": job["missing"], "failed": job.get("failed", 0),
                              "bytes": job["bytes"], "seconds": round(job["t1"] - job["t0"], 2),
                              "plan_T": job.get("plan_T"), "cover": job.get("cover"),
                              "per_peer_MB": {p[:12]: round(b / 1e6, 1) for p, b in job["per_peer"].items()}}))
    if a.out:
        export(a.link, a.ctl, want, t0, t1, isel, a.out, _chunking(a.chunking), a.step)
        print(f"written {a.out}")


def export(link, ctl, vars_, t0, t1, isel, out, chunking=None, step=None):
    """Write a (downloaded) slice to *.zarr or *.nc (NetCDF4/HDF5, needs h5netcdf or netCDF4)."""
    view = open_views(link, ctl, step=step)[0][1]
    ds = open_dataset(link, ctl=ctl, chunking=chunking, step=step)
    sub = ds[vars_] if vars_ else ds
    if view.t and (t0 or t1):
        sub = sub.sel({view.t["name"]: slice(t0, t1)})
    if isel:
        sub = sub.isel({d: slice(*r) for d, r in isel.items() if d in sub.dims})
    for v in sub.variables.values():
        v.encoding.clear()
    sub.attrs = {k: v for k, v in sub.attrs.items() if not k.startswith("zt_")}
    if out.endswith((".nc", ".nc4", ".h5")):
        sub.load().to_netcdf(out, engine="h5netcdf" if _has("h5netcdf") else None)
    else:
        sub.to_zarr(out, mode="w")


def cmd_mean(a):
    ds = open_dataset(a.link, ctl=a.ctl)
    t0, t1 = _time(a.time)
    cb = lambda r: sys.stderr.write(f"\rmean={r['mean']:.6g} ±{r['ci95']:.3g}  chunks {r['chunks']}/{r['of']}  "
                                    f"{_mb(r['bytes'])}  {r['seconds']:.1f}s   ")
    fn = progressive_mean_vas if a.method == "vas" else progressive_mean
    r = fn(ds, a.var, t0, t1, rel_err=a.rel_err, callback=cb)
    sys.stderr.write("\n")
    print(json.dumps(r))


def cmd_parity(a):
    """Volunteer availability: keep one Reed-Solomon parity row of interleaved k-chunk stripes (1/k storage)."""
    for grid, view in open_views(a.link, a.ctl):
        if a.var in view.arrays:
            r = http(a.ctl, "POST", "/api/parity", {"grid": grid, "var": a.var, "k": a.k, "drop": a.drop,
                                                          "row": a.row})
            print(json.dumps(r))


def _dur(s: str) -> float:
    """'7d' / '12h' / '30m' / '3600' -> seconds"""
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}.get(s[-1:].lower())
    return float(s[:-1]) * mult if mult else float(s)


def cmd_follow(a):
    r = http(a.ctl, "POST", "/api/follow", {"link": a.link, "vars": a.vars.split(",") if a.vars else None,
                                            "last_s": _dur(a.last), "isel": _isel_from_sel(a.link, a.ctl, a.sel),
                                            "step": step_seconds(a.step)})
    print(f"subscription {r['id']}: keeping the last {a.last} of {a.vars or 'all variables'} of {a.link} "
          f"up to date ({len(r['jobs'])} catch-up jobs started)")


def cmd_follows(a):
    for s in http(a.ctl, "GET", "/api/follows"):
        print(f"{s['id']}  {s['link']}  vars={','.join(s['vars'] or ['*'])}  last={s['last_s'] / 3600:g} h")


def cmd_unfollow(a):
    print("removed" if http(a.ctl, "POST", "/api/unfollow", {"id": a.id})["ok"] else "no such subscription")


def cmd_name(a):
    print(http(a.ctl, "POST", "/api/name", {"name": a.name, "target": a.link})["link"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zt", description="P2P distribution of Zarr arrays")
    ap.add_argument("--ctl", default=CTL, help="local node control URL (env ZT_CTL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="create identity + config for this host")
    p.add_argument("--home")
    p.add_argument("--port", type=int)
    p.add_argument("--bootstrap-node", action="store_true", help="publicly reachable first node (+relay)")
    p.add_argument("--public-host")
    p.add_argument("--join", help="ztnet://ID@HOST:PORT of any bootstrap node")
    p.add_argument("--listen-public", action="store_true",
                   help="bind 0.0.0.0 and auto-detect reachability (default: outbound-only via relay)")
    p.add_argument("--service", action="store_true", help="write a systemd --user unit")
    p.add_argument("--private", action="store_true",
                   help="closed network: generate a network key; the printed ztnet:// invite carries it")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("node")
    p.add_argument("--home")
    p.add_argument("--host", help="data-port bind address (0.0.0.0 for public)")
    p.add_argument("--port", type=int)
    p.add_argument("--ctl-port", type=int, default=None)
    p.add_argument("--public", help="externally reachable URL of the data port")
    p.add_argument("--bootstrap", nargs="*")
    p.add_argument("--relay", help="relay URL to attach to when not publicly reachable")
    p.add_argument("--relay-server", action="store_true", help="relay traffic for NAT'd peers")
    p.add_argument("--auto", action="store_true", help="detect public reachability via bootstrap")
    p.add_argument("--trust", nargs="*", help="trusted publisher public keys (their values win over majority)")
    p.add_argument("--upload-mbps", type=float, help="upload cap, MB/s (config upload_mbps)")
    p.set_defaults(fn=cmd_node)
    for name, fn in (("scan", cmd_scan), ("seed", cmd_seed), ("unseed", cmd_unseed)):
        p = sub.add_parser(name)
        p.add_argument("path")
        p.set_defaults(fn=fn)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("search")
    p.add_argument("tag")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("peers")
    p.add_argument("link")
    p.set_defaults(fn=cmd_peers)
    p = sub.add_parser("get")
    p.add_argument("link")
    p.add_argument("--vars")
    p.add_argument("--time", help="START:END, ISO dates (use / as separator if times contain ':')")
    p.add_argument("--out", help="output: *.zarr (default) or *.nc (NetCDF4)")
    p.add_argument("--chunking", help="view chunking for --out, e.g. time=8760,lat=1,lon=1")
    p.add_argument("--sel", help="spatial subset by coordinate values, e.g. lat=40:60,lon=0:30")
    p.add_argument("--progressive", action="store_true")
    p.add_argument("--step", help="time step: 6h, 1d, 3600 (default: finest step any replica offers)")
    p.add_argument("--cover", default="jlps", choices=["jlps", "bytes"], help="source-chunk cover selection")
    p.set_defaults(fn=cmd_get)
    p = sub.add_parser("mean")
    p.add_argument("link")
    p.add_argument("var")
    p.add_argument("--time")
    p.add_argument("--rel-err", type=float, default=0.01)
    p.add_argument("--method", default="vas", choices=["vas", "vdc"], help="vas: stratified Neyman (default)")
    p.set_defaults(fn=cmd_mean)
    p = sub.add_parser("parity", help="store erasure-stripe parity (1/k) to raise availability of VAR")
    p.add_argument("link")
    p.add_argument("var")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--drop", action="store_true", help="keep only parity, delete the data chunks")
    p.add_argument("--row", type=int, help="Cauchy row (default: a row no other volunteer publishes yet)")
    p.set_defaults(fn=cmd_parity)
    p = sub.add_parser("follow", help="subscribe: keep the newest LAST of LINK downloaded (feeds, names)")
    p.add_argument("link")
    p.add_argument("--vars")
    p.add_argument("--last", default="7d", help="window before the newest step: 7d, 12h, 30m")
    p.add_argument("--sel", help="spatial subset, e.g. lat=40:60,lon=0:30")
    p.add_argument("--step", help="time step to keep, e.g. 6h (default: finest available)")
    p.set_defaults(fn=cmd_follow)
    sub.add_parser("follows", help="list subscriptions").set_defaults(fn=cmd_follows)
    p = sub.add_parser("unfollow")
    p.add_argument("id")
    p.set_defaults(fn=cmd_unfollow)
    p = sub.add_parser("name")
    p.add_argument("name")
    p.add_argument("link")
    p.set_defaults(fn=cmd_name)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
