"""Post-hoc temperature diagnosis; use the frozen seasonal source on PYTHONPATH."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from zarrswarm import codec

work = Path(sys.argv[1])
result = json.loads((work / "provider_seasonal_v5.json").read_text())
out = {"scope": "Post-hoc diagnosis, no estimator tuning; NCAR shared quantizer uses the declared ARCO-count perturbation.",
       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "codec_sha256": hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest(), "fields": {}}
for key, row in result["fields"].items():
    if not key.startswith("2m_temperature@"):
        continue
    var, ts = key.split("@")
    a, b = [np.load(work / "fields" / f"{var}_{ts}_{s}.npy", allow_pickle=False) for s in ("ARCO", "NCAR")]
    lattices = [codec.lattice_of(v) for v in (a, b)]
    step, phase, _ = lattices[1]
    rejected = {"step": 0, "code_hash": 0, "level": 0}
    deltas, positive, negative = [], 0, 0
    for y in range(0, 721, 91):
        for x in range(0, 1440, 180):
            ta, tb = a[y:y + 91, x:x + 180], b[y:y + 91, x:x + 180]
            va, vb = codec.vcid_of(ta), codec.vcid_of(tb)
            pa, pb = json.loads(va.split(":", 2)[2])[0], json.loads(vb.split(":", 2)[2])[0]
            deltas.append([pa[1], pb[1]])
            if not codec.same_vcid(va, vb):
                reason = "step" if abs(pa[1] - pb[1]) > 1e-5 * min(pa[1], pb[1]) else (
                    "code_hash" if va.split(":", 2)[1] != vb.split(":", 2)[1] else "level")
                rejected[reason] += 1
            changed = ta.copy()
            changed.flat[changed.size // 2] += row["ARCO_field_step"]
            quantize = lambda v: np.rint((v.astype("f8") - phase) / step).astype("i8")
            qa = quantize(ta)
            positive += int(np.array_equal(qa, quantize(tb)))
            negative += int(np.array_equal(qa, quantize(changed)))
    out["fields"][key] = {"full_lattices": lattices, "rejected_by_first_diagnostic": rejected,
        "tile_step_ranges": np.array([np.min(deltas, axis=0), np.max(deltas, axis=0)]).T.tolist(),
        "max_absolute_difference": float(np.max(np.abs(a.astype("f8") - b))),
        "shared_NCAR_quantizer": {"accepted_provider_tiles": positive, "accepted_ARCO_count_changed_tiles": negative}}
(work / "provider_seasonal_diagnostic_v5.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
