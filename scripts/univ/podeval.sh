#!/usr/bin/env bash
# scripts/univ/podeval.sh CKPT TAG [PREDICT_BIN]: scripts/univ/eval.sh on a pod with the desk's labels (kit: scripts/univ/podkit_eval.sh).
# PODGPU: the pod GPU index (default 0). PODBIN: a predict binary already on the pod (e.g. a pinned pre-fix build), for UP and RVU.
# Copies CKPT (and PREDICT_BIN, the binary that knows the checkpoint, e.g. a run's ufsm.bin) to the pod, runs eval.sh there with the env
# given here (TASKS, DLVS, WZARGS, WIN, HALO, INKP, FIBCH, ...: every eval.sh variable set in this environment is passed through), then copies
# the wbench rows (K/wb/res/TAG*.tsv) and the matrix back next to the desk's (/vesuvius/ufsm/wbench/{res,univ}). POD="HOST PORT" (pod 2).
set -u
CK=${1:?ckpt}; T=${2:?tag}; PB=${3:-}; read -r HOST PORT <<< "${POD:-root@213.173.107.10 27064}"; K=${K:-/workspace/ue}
E="ssh -p $PORT -i $HOME/.ssh/id_ed25519 -o BatchMode=yes"; WB=/vesuvius/ufsm/wbench
# CKPT "pod:PATH": a checkpoint already on the pod (a pod-trained run's snapshot), copied there, nothing uploaded
if [ "${CK#pod:}" != "$CK" ]; then $E $HOST "cp -p ${CK#pod:} $K/ckpt/$T.ckpt" < /dev/null || exit 1
else CK=$(readlink -f "$CK"); rsync -aL -e "$E" $CK $HOST:$K/ckpt/$T.ckpt < /dev/null || exit 1; fi
UPV="RVU=$K/ufsm/build/ufsm"; [ -n "${PODBIN:-}" ] && UPV="UP=$PODBIN RVU=$PODBIN"; [ -n "$PB" ] && { rsync -aL -e "$E" $PB $HOST:$K/ckpt/$T.bin < /dev/null || exit 1; UPV="UP=$K/ckpt/$T.bin RVU=$K/ckpt/$T.bin"; }
PASS=""; for v in TASKS DLVS WZARGS WIN HALO INKP FIBCH LV_INK RVSC REFS THREADS INKC INKGT; do [ -n "${!v+x}" ] && PASS="$PASS $v='${!v}'"; done
$E $HOST "cd $K/ufsm && git pull -q --ff-only; [ -d $K/gpu.shared ] && touch $K/gpu.gate-waiting; env SITE=pod K=$K WB=$K/wb GPU=${PODGPU:-0} INKW=$K/ckpt/ckpt_78k_fullsup.pth FIBPH=$K/wbgt/paris4-wind-20261019 CTR=https://dl.ash2txt.org/community-uploads/forrest/volcomp VOLP=volumes/ CNT=$K/count RVGT=$K/wbgt RU=$K/ufsm/build/ufsm RVE=$K/rv/eval LOCK= TH=8 $UPV $PASS \
  bash scripts/univ/eval.sh $K/ckpt/$T.ckpt $T > $K/wb/univ_$T.out 2>&1; pgrep -f univ/[e]val.sh > /dev/null || rm -f $K/gpu.gate-waiting" < /dev/null
mkdir -p $WB/univ/pod; rsync -a -e "$E" "$HOST:$K/wb/res/$T*.tsv" $WB/univ/pod/ < /dev/null; rsync -a -e "$E" $HOST:$K/wb/univ/$T.matrix.txt $HOST:$K/wb/univ/$T.log $WB/univ/pod/ < /dev/null; rsync -a -e "$E" "$HOST:$K/rv/eval/$T.txt" $WB/univ/pod/rv_$T.txt < /dev/null 2>/dev/null
echo "pod rows and matrix -> $WB/univ/pod/ ($T)"
