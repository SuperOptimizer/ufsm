#!/usr/bin/env bash
# The universal run u1 (2026-11-11): U1 + U2 folded (winding + rv 2 + fine stem), both desk GPUs, init wz896-12000. Launcher: u1.train.sh
# (scripts/rv/train.sh with u1's defaults baked in, relaunch inside memcap -t). Sources: sources-univ.json with Paris 4 max_level 3 (19.2 um rung).
cd /vesuvius/ufsm/v2/lat/runs && R=/vesuvius/ufsm/v2/lat/runs U=/vesuvius/ufsm/v2/lat/bin/ufsm.univ1 SRCJ=/vesuvius/ufsm/univ/u1/sources-u1.json GPU=0,1 P=896 ARCH="--widths 96,128,160,192" WORKERS=11 SLOTS=13 MEMMODE=auto ALR=0.0001 MLR=0.002 WIND4L=0.25 ISO=64 \
 EXTRA="--lat 4 --lat-basis /vesuvius/ufsm/v2/lat/codec/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --replay 3 --um-rungs 4.8,9.6:2,19.2 --rung-shift-p 0.2 --film 2 --fstem 16 --fstem-drop 0.25 --uw 1 --grad-cos 500 --init-from /vesuvius/ufsm/v2/lat/runs/wz896/last.ckpt" SNAP=3000 \
 nohup /home/forrest/bin/memcap -t bash /vesuvius/ufsm/v2/lat/runs/u1.train.sh u1 30000 >> /vesuvius/ufsm/v2/lat/runs/u1.nohup 2>&1 &
