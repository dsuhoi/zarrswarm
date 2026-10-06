import sys,runpy,json,hashlib,statistics,os
from pathlib import Path
if os.environ.get("PYTHONHASHSEED") != "0":
 raise SystemExit("Run with PYTHONHASHSEED=0 from the repository root")
seen={}
def capture(frame,event,arg):
 if event=='return' and frame.f_code.co_name=='test_plan_against_brute_force':
  r=frame.f_locals['ratios'];seen.update(cases=len(r),max_ratio=max(r),above_1_2=sum(x>1.2 for x in r),median_ratio=statistics.median(r))
sys.setprofile(capture)
runpy.run_path('tests/test_algorithms.py')['test_plan_against_brute_force']()
sys.setprofile(None)
seen['python_hash_seed']=os.environ.get('PYTHONHASHSEED','random')
seen['source_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('zarrswarm/plan.py','tests/test_algorithms.py')}
Path('bench/revalidation/assignment_controls_v5.json').write_text(json.dumps(seen,indent=2)+'\n')
print(json.dumps(seen))
