#!/bin/bash
# runs detached on cloudru: generate replica C, start node (via reverse tunnel to relay), seed it
cd ~/zs_src
PY=~/zspkg/py/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
export PYTHONPATH=~/zspkg/site:~/zs_src ZS_ANNOUNCE_EVERY=60
mkdir -p ~/zs_data
[ -f ~/zs_data/era5like_C.zarr/zarr.json ] || $PY bench/make_era5like.py ~/zs_data/era5like_C.zarr --start 2020-01-20 --hours 864 --vars t2m,u10 --chunks 72,181,360 --block 216 > ~/zs_gen.log 2>&1
$PY -m zarrswarm.cli node --home ~/.zs_real --port 17875 --ctl-port 17876 --bootstrap http://127.0.0.1:17881 --relay http://127.0.0.1:17881 > ~/zs_real.log 2>&1 &
for i in $(seq 1 60); do sleep 5; $PY -m zarrswarm.cli --ctl http://127.0.0.1:17876 seed ~/zs_data/era5like_C.zarr > ~/zs_seed.log 2>&1 && break; done
wait
