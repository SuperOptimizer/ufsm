#!/usr/bin/env bash
# wcrush.sh: crush-augmented winding arm (desk; NOT started by any agent: the coordinator schedules the GPUs). The wrad.sh recipe
# (wind96 trunk via --init-from wind96-wk1-best, fresh zero-init 2-block radial attention stage, scroll RoPE; wtrain.sh: unwrap2,
# P512 9.6 um, wind96 aug, packed weight 8, sources-pk8.json with PH1-3 / HOLD held out, 1667 never trained) + --aug-crush 0.5
# (sheet-aware gap collapse / in-plane squash / bends + kinks on half the windows, src/crush.h). Needs a master build with crush, e.g.
#   git clone (or pull) master into /vesuvius/ufsm/paris4/srcc && make -C /vesuvius/ufsm/paris4/srcc -j16 build/ufsm   (or set U=)
# Install: copy wcrush.sh + wcrushsched.sh to /vesuvius/ufsm/paris4/wnew. Start: setsid nohup ./wcrushsched.sh & (GPU=1 default).
# Resumes from runs/wcrush/last.ckpt when present. Loader: ~15-29 core-s per crushed P512 window -> keep WORKERS 24.
P=/vesuvius/ufsm/paris4/wnew; cd $P
R=""; [ -f runs/wcrush/last.ckpt ] && R=runs/wcrush/last.ckpt
U=${U:-/vesuvius/ufsm/paris4/srcc/build/ufsm} RESUME=$R \
  EXTRA="--init-from /vesuvius/ufsm/paris4/ckpt/wind96-wk1-best.ckpt --attn-blocks 2 --attn-axis radial --attn-pe scroll --aug-crush ${CRUSH:-0.5}" \
  exec ./wtrain.sh wcrush $P/sources-pk8.json ${STEPS:-100000} ${GPU:-1}
