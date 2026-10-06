"""Known source quantum versus an independently fitted step, and a trusted-packing control.

Run: python bench/identity_boundary.py --out bench/revalidation/identity_boundary_v7.json
These synthetic counterexamples measure an inference limit, not provider prevalence.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zarrswarm import codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", type=Path, default=ROOT / "bench/revalidation/packing_protocol_v8.json")
    args = ap.parse_args()
    assert codec.VALUE_ID == "lattice"
    result = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "estimator": codec.ESTIMATOR, "numpy": np.__version__,
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in ("zarrswarm/codec.py", "bench/identity_boundary.py")},
              "packing_protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              "packing_protocol": json.loads(args.protocol.read_text()),
              "protocol": {"source_quantum": 1, "observed_code_strides": [1, 2, 4, 8, 16],
                           "distinct_values": [2, 4, 16, 256], "dtype": "float32",
                           "values": "arange(distinct_values) * observed_code_stride",
                           "change": "add exactly one source count to the first cell",
                           "selection": "All 20 combinations; no fitting changes or omitted cases."}}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")  # protocol before executing cases
    cases = []
    for stride in result["protocol"]["observed_code_strides"]:
        for count in result["protocol"]["distinct_values"]:
            a = np.arange(count, dtype="f4") * stride
            b = a.copy()
            b[0] += 1
            va, vb = codec.vcid_of(a), codec.vcid_of(b)
            before, after = codec.lattice_of(a), codec.lattice_of(b)
            codec.VALUE_ID = "exact"
            try:
                exact_match = codec.same_vcid(codec.vcid_of(a), codec.vcid_of(b))
            finally:
                codec.VALUE_ID = "lattice"
            packing = (1, 0, .45)
            anchored = codec.vcid_of(a, packing=packing)
            decoded = a.astype("f8") + .4 * np.cos(a.astype("f8"))
            cases.append({"observed_code_stride": stride, "distinct_values": count,
                          "fit_before": before, "fit_after": after,
                          "identity_before": va, "identity_after": vb,
                          "lattice_accepts_changed_cell": codec.same_vcid(va, vb),
                          "exact_accepts_changed_cell": exact_match,
                          "packing_accepts_changed_cell": codec.same_vcid(anchored, codec.vcid_of(b, packing=packing)),
                          "packing_accepts_valid_decoder": codec.same_vcid(anchored, codec.vcid_of(decoded, packing=packing))})
    assert any(c["lattice_accepts_changed_cell"] for c in cases)
    assert not any(c["exact_accepts_changed_cell"] for c in cases)
    assert not any(c["lattice_accepts_changed_cell"] for c in cases if c["observed_code_stride"] == 1)
    assert not any(c["packing_accepts_changed_cell"] for c in cases)
    assert all(c["packing_accepts_valid_decoder"] for c in cases)
    transfer = []
    spec = result["packing_protocol"]["synthetic_transfer"]
    for step in spec["steps"]:
        for stride in spec["strides"]:
            for count in spec["sizes"]:
                codes = np.arange(count, dtype="f8") * stride
                a = spec["origin"] + step * codes
                b = a + spec["decoder_error_bound_steps"] * step * np.cos(codes)
                packing = (step, spec["origin"], spec["acceptance_budget_steps"] * step)
                va = codec.vcid_of(a, packing=packing)
                changed_matches = []
                for direction in spec["negative_directions"]:
                    changed = a.copy(); changed[0] += direction * step
                    changed += spec["decoder_error_bound_steps"] * step * np.cos((changed - spec["origin"]) / step)
                    changed_matches.append(codec.same_vcid(va, codec.vcid_of(changed, packing=packing)))
                transfer.append({"step": step, "stride": stride, "size": count,
                                 "accepts_valid_decoder": codec.same_vcid(va, codec.vcid_of(b, packing=packing)),
                                 "accepts_changed_codes": changed_matches})
    assert len(transfer) == 36 and all(c["accepts_valid_decoder"] for c in transfer)
    assert not any(any(c["accepts_changed_codes"]) for c in transfer)
    result.update(cases=cases, transfer_cases=transfer, completed_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "accepted_source_count_changes":
                      sum(c["lattice_accepts_changed_cell"] for c in cases),
                      "exact_accepted_changes": 0, "packing_accepted_changes": 0,
                      "packing_valid_decoders": len(cases), "transfer_valid_decoders": len(transfer),
                      "transfer_accepted_changes": 0}))


if __name__ == "__main__":
    main()
