"""Ground-station contact traces for the sensing scenario.

Real inputs: current TLEs of polar-orbiting weather satellites with direct broadcast (CelesTrak 'weather' group) and
the ONLINE stations of the SatNOGS network (their real coordinates). For a period, skyfield computes every pass above
a minimum elevation; a station "ingests" an hour if it had at least one contact during it. Output: per station, the
set of hours it holds - the natural, sparse, overlapping replica placement of independently operated sites.

python bench/sat_traces.py --start 2020-01-01 --days 31 --stations 40 [--out bench/sat_traces.json]
(the orbit geometry is propagated from today's TLEs and replayed over the dataset period: the pass pattern, not
the calendar, is what matters for placement)
"""
import argparse
import json
import math
import random
import urllib.request

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

SATS = ("NOAA 20", "NOAA 21", "SUOMI NPP", "METOP-B", "METOP-C", "NOAA 19", "NOAA 18", "FENGYUN 3D")


def tles(names=SATS):
    txt = urllib.request.urlopen("https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle", timeout=60).read().decode()
    lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
    out = {}
    for i in range(0, len(lines) - 2, 3):
        name = lines[i].strip()
        if any(name.startswith(s) for s in names):
            out[name] = (lines[i + 1], lines[i + 2])
    return out


def stations(k, seed):
    d = json.load(urllib.request.urlopen("https://network.satnogs.org/api/stations/?format=json", timeout=120))
    on = [x for x in d if x.get("status") == "Online" and x.get("lat") is not None]
    rng = random.Random(seed)
    pick = [rng.choice(on)]
    while len(pick) < min(k, len(on)):  # farthest-point sampling: geographically spread, like independent sites
        far = max(on, key=lambda x: min(_dist(x, p) for p in pick))
        pick.append(far)
    return [{"id": x["id"], "name": x["name"], "lat": x["lat"], "lon": x["lng"], "alt": x.get("altitude") or 0,
             "min_el": max(10, x.get("min_horizon") or 10)} for x in pick]


def _dist(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lng"], b["lat"], b["lng"]))
    return math.acos(max(-1, min(1, math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(lo1 - lo2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=31)
    ap.add_argument("--stations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench/sat_traces.json")
    ap.add_argument("--sats", default=",".join(SATS), help="comma list of satellite name prefixes")
    a = ap.parse_args()
    ts = load.timescale()
    sats = {n: EarthSatellite(l1, l2, n, ts) for n, (l1, l2) in tles(tuple(a.sats.split(","))).items()}
    sts = stations(a.stations, a.seed)
    t0 = ts.now()
    t0 = ts.utc(t0.utc_datetime().year, t0.utc_datetime().month, t0.utc_datetime().day)
    t1 = ts.tt_jd(t0.tt + a.days)
    res = {"satellites": sorted(sats), "days": a.days, "stations": []}
    for st in sts:
        pos = wgs84.latlon(st["lat"], st["lon"], st["alt"])
        hours, contacts = set(), 0
        for sat in sats.values():
            t, ev = sat.find_events(pos, t0, t1, altitude_degrees=st["min_el"])
            rise = None
            for ti, e in zip(t, ev):
                if e == 0:
                    rise = ti
                elif e == 2 and rise is not None:
                    h0 = int((rise.tt - t0.tt) * 24)
                    h1 = int((ti.tt - t0.tt) * 24)
                    hours.update(range(h0, h1 + 1))
                    contacts += 1
                    rise = None
        res["stations"].append(dict(st, contacts=contacts, hours=sorted(h for h in hours if h < a.days * 24)))
        print(f"{st['name'][:24]:24} lat={st['lat']:7.2f} contacts={contacts:4d} hours={len(hours):4d}", flush=True)
    cover = np.zeros(a.days * 24, int)
    for s in res["stations"]:
        cover[s["hours"]] += 1
    res["hour_replication"] = {"min": int(cover.min()), "median": float(np.median(cover)), "max": int(cover.max()),
                               "uncovered_frac": float((cover == 0).mean())}
    print(json.dumps(res["hour_replication"]))
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
