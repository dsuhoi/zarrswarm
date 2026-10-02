#!/bin/bash
# runs detached on cloudru: generate replica C, start node (via reverse tunnel to relay), seed it
cd ~/zt_src
PY=~/ztpkg/py/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
export PYTHONPATH=~/ztpkg/site:~/zt_src ZT_ANNOUNCE_EVERY=60
mkdir -p ~/zt_data
[ -f ~/zt_data/era5like_C.zarr/zarr.json ] || $PY bench/make_era5like.py ~/zt_data/era5like_C.zarr --start 2020-01-20 --hours 864 --vars t2m,u10 --chunks 72,181,360 --block 216 > ~/zt_gen.log 2>&1
$PY -m zarr_torrent.cli node --home ~/.zt_real --port 17875 --ctl-port 17876 --bootstrap http://127.0.0.1:17881 --relay http://127.0.0.1:17881 > ~/zt_real.log 2>&1 &
for i in $(seq 1 60); do sleep 5; $PY -m zarr_torrent.cli --ctl http://127.0.0.1:17876 seed ~/zt_data/era5like_C.zarr > ~/zt_seed.log 2>&1 && break; done
wait
