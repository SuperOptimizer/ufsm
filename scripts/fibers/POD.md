# Fibre retrain on a rented pod (2026-11-11)

Pod: RTX PRO 4000 Blackwell 24 GB (sm_120), ~10 CPUs (nproc says 48, the host's; the cgroup quota /sys/fs/cgroup/cpu.max is 1020000/100000), 125 GB RAM shown but a 31 GB cgroup limit (/sys/fs/cgroup/memory.max: two 20-thread rasters next to the trainer were OOM-killed), 250 GB at /workspace, Ubuntu 24.04, CUDA 13.2 at /usr/local/cuda.
Self-contained: the source is rsynced from the desk (no build dirs, no .git), the AFVs and the CT come straight from dl.ash2txt.org
(the desk's upload is ~4 MB/s; the pod downloads at ~20 MB/s). Only small results (checkpoints, eval masks) go back to the desk.

## Setup

- Installed: gcc-14 (default gcc is 13; C23 needs 14), libsqlite3-0 (afv.c dlopens it), libblosc-dev, libxxhash-dev, pkg-config.
  Already there: libcurl4-openssl-dev, libzstd-dev, zlib1g-dev, libssl-dev, libjpeg-dev, zstd, jq, python3, curl, rsync.
- Source: `rsync -az --exclude 'build*' --exclude .git ufsm-fib/ root@POD:/workspace/ufsm-fib/` (from the desk's branch fibers-20261110 worktree).
- Build: `cd /workspace/ufsm-fib && make -j40 CC=gcc-14 HOSTCC=gcc-14 build/ufsm` (nvcc picks g++-14 itself).
- AFVs: /workspace/fibers/raw/{PHerc0175A_1mm_v2,0175A_SMALLMORE,PHerc1447_1mm_v2,PHerc1447_all_v2}.afv, from
  https://dl.ash2txt.org/community-uploads/qual/AutomatedFiberVolumes/ with their .sha256 (checked: *.check).
- Meta from the desk: /workspace/fibers/meta/{axis-0175A.json,axis-1447.json,holdouts.json} (the wbench holdouts).
- CT for training: level 0 of both scans copied to /workspace/ct/volcomp/<scroll>/volumes/<scan>.zarr with `wget -r -np -nH --cut-dirs=2`
  (~6 MB/s while the rasters also read; CTROOT=/workspace/ct/volcomp). The rasters read the mirror directly.
- CT: the volcomp mirror https://dl.ash2txt.org/community-uploads/forrest/volcomp (keys `<scroll>/volumes/<scan>.zarr`), training cache
  /workspace/cache. The mirror also has PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr (the wbench / HOLD scan).

## Labels (ring, fibhv-b)

`CT=https://dl.ash2txt.org/community-uploads/forrest/volcomp VOL=volumes/ A=/workspace/fibers/raw O=/workspace/fibers/labels2 MEMCAP= U=/workspace/ufsm-fib/build/ufsm nohup bash scripts/fibers/mklabels2.sh 20 &`
(SEQ=1 MEMCAP= ... mklabels2.sh 10: one scroll at a time, 10 threads, ~10 GB; --resume 1 continues after a kill). Sources (rerun as rows complete):
`CTROOT=https://dl.ash2txt.org/community-uploads/forrest/volcomp VOL=volumes/ CACHE=/workspace/cache AX0175=/workspace/fibers/meta/axis-0175A.json AX1447=/workspace/fibers/meta/axis-1447.json WBH=/workspace/fibers/meta/holdouts.json LAB=/workspace/fibers/labels2 bash scripts/fibers/mksources2.sh /workspace/fibers/sources-fibhv-b.json`

## Training

See the run's `fibhv-b.launch.sh` in /workspace/fibers/runs (scripts/fibers/train_hv.sh with R, SRCJ, U, P, LOCK= MEMCAP=).
Results copied to the desk: /vesuvius/ufsm/fibers/runs/fibhv-b (snapshots), eval on the desk (Paris 4 HOLD GT lives there).
