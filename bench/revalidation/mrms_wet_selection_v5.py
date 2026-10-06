"""Select a fixed wet crop from a retained precursor, before collecting future fields."""
import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import gribberish
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("out", type=Path)
args = parser.parse_args()
assert not args.out.exists()
record = {"declared_utc": datetime.now(timezone.utc).isoformat(),
          "rule": "Maximize finite positive cells over 1024-square crops starting at latitude indices 0,1024,2048,2476 and longitude indices 0,1024,2048,3072,4096,5120,5976; break ties by first index pair. Freeze the crop before observing future fields.",
          "source_file": str(args.source)}
args.out.write_text(json.dumps(record, indent=2) + "\n")
payload = args.source.read_bytes()
raw = gzip.decompress(payload)
meta = gribberish.parse_grib_message_metadata(raw, 0)
values = gribberish.parse_grib_message(raw, 0).data().reshape(meta.grid_shape)
assert values.shape == (3500, 7000)
counts = [(int(np.count_nonzero(np.isfinite(crop := values[i:i+1024, j:j+1024]) & (crop > 0))), i, j)
          for i in (0, 1024, 2048, 2476) for j in (0, 1024, 2048, 3072, 4096, 5120, 5976)]
count, i, j = max(counts, key=lambda c: c[0])
assert count > 0
record.update(selected_utc=datetime.now(timezone.utc).isoformat(),
              source_sha256=hashlib.sha256(payload).hexdigest(), source_time=str(meta.forecast_date),
              crop=[i, j], precursor_positive_cells=count, candidate_counts=counts)
args.out.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: record[k] for k in ("source_time", "crop", "precursor_positive_cells")}))
