#!/usr/bin/env bash
# pseudo-box.sh OUT ROOT GROUP UM AXIS BOX [GTROOT GTKEY GTUM]: winding pseudo labels (dino-train GT SupCon codes) for one box of one scan.
#   1 predict: the winding checkpoint CK (default wind96-best; task unwrap / unwrap2) over BOX (z,y,x,nz,ny,nx in level-LEVEL voxels of the CT
#     pyramid GROUP, UM = its level-0 um; AXIS: umbilicus control points in level-0 voxels)
#     -> fit fields OUT/fields (zarr, h 2), no logits kept
#   2 fit: the spiral fit of the box (dir voted unless DIR), written as band codes 1 + round(18 U) mod 252 at the CT grid (step 1),
#     gated per voxel (GATE = CONF,ERR,MAT: network phase confidence >= CONF, |fit W - network phase| <= ERR turns, material >= MAT;
#     else 0 / 255 = unlabelled): OUT/codes/<um key>
#   3 optional GT check (GTROOT GTKEY GTUM: real band codes on the scan's grid): the gated codes' share within 0.5 turn of the GT, whole
#     box (one gauge) and per 128^3 block (own gauge each: what a training crop sees); the lines go to OUT/check.txt
# Output OUT/source.json: one dino-train source entry (codes origin = -box corner) to merge into a sources file.
# GATE2 (optional, e.g. 0.3,0.15,0.3): a second code set OUT/codes2 + OUT/source2.json from the same fit.
# Env: LEVEL (0), U (binary), CK, GPU (0), TH (8), GATE (0.5,0.1,0.3), DIR (0 = vote), WIN (640), WINZ (384), MEM (capped.sh limit on the laptop), PNAME (source name; NAME is the host on WSL).
set -u
OUT=$1 ROOT=$2 KEY=$3 UM=$4 AX=$5 BOX=$6; GTR=${7:-} GTK=${8:-} GTU=${9:-}
U=${U:-$HOME/ufsm-data/dino/bin/ufsm-fs24}; CK=${CK:-$HOME/ufsm-data/fitkit/ckpt/wind96-best.ckpt}; G=${GPU:-0}; TH=${TH:-8}
GATE=${GATE:-0.5,0.1,0.3}; NAME=${PNAME:-$(basename $OUT)}
case "$(hostname)" in forlindesk2*) RUN=(nice -n 5);; *) RUN=(flock ${LOCK:-$HOME/ufsm-data/gpu.lock} env MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh);; esac   # (laptop: the GPU lock (LOCK) + the memory slice)
mkdir -p $OUT; [ -s $OUT/source.json ] && { echo "$NAME: done"; exit 0; }
IFS=, read -r bz by bx bnz bny bnx <<< "$BOX"
if [ ! -s $OUT/fields/fields.json ]; then
  rm -rf $OUT/fields $OUT/scratch
  "${RUN[@]}" $U predict $CK $ROOT $KEY $OUT/scratch --um $UM --level ${LEVEL:-0} --box $BOX --window-z ${WINZ:-384} --halo-z 64 --window ${WIN:-640} --halo 64 --shard 512 --ema 1 --q 8 \
    --threads $TH --gpu $G --axis $AX --levels 1 --fit-zarr $OUT/fields --fit-h 2 --no-out 1 > $OUT/pred.log 2>&1 || { echo "$NAME: predict failed"; tail -3 $OUT/pred.log; exit 1; }
  rm -rf $OUT/scratch
fi
CHK=(); [ -n "$GTR" ] && CHK=(--winding-check $GTR $GTK $GTU)
rm -rf $OUT/codes $OUT/fit
"${RUN[@]}" env UFSM_FIT_THREADS=$TH $U fit $OUT/fields --gpu $G --out $OUT/fit --dir ${DIR:-0} --winding $OUT/codes --winding-mod14 2 --winding-step 1 \
  --winding-gate $GATE "${CHK[@]}" > $OUT/fit.log 2>&1 || { echo "$NAME: fit failed"; tail -3 $OUT/fit.log; exit 1; }
grep -E "^fit: dir|winding-gate|winding-check" $OUT/fit.log > $OUT/check.txt
if [ -n "${GATE2:-}" ]; then   # a second (looser) code set from the same fit: OUT/codes2, OUT/source2.json
  rm -rf $OUT/codes2
  "${RUN[@]}" env UFSM_FIT_THREADS=$TH $U fit $OUT/fields --gpu $G --model $OUT/fit --winding $OUT/codes2 --winding-mod14 2 --winding-step 1 \
    --winding-gate $GATE2 "${CHK[@]}" > $OUT/fit2.log 2>&1 || { echo "$NAME: fit (GATE2) failed"; tail -3 $OUT/fit2.log; exit 1; }
  grep -E "winding-gate|winding-check" $OUT/fit2.log > $OUT/check2.txt
fi
ck=$(ls $OUT/codes | grep -v zarr.json | head -1)
cos=$(grep -oE 'phase\) [-0-9.]+' $OUT/check.txt | awk '{print $2}'); lab=$(grep -oE 'labelled \([0-9.]+%' $OUT/check.txt | tr -d '(%' | awk '{print $2}')
jq -n --arg n "$NAME" --arg r "$ROOT" --arg k "$KEY/${LEVEL:-0}" --arg cr "$(readlink -f $OUT/codes)" --arg ck "$ck" --argjson b "[$BOX]" --argjson cos "${cos:-0}" --argjson lab "${lab:-0}" --arg g "$GATE" \
  '{name: $n, root: $r, ct: $k, codes: {root: $cr, key: $ck}, origin: [-$b[0], -$b[1], -$b[2]], boxes: [$b], pseudo: {agree_cos: $cos, labelled_pct: $lab, gate: $g}}' > $OUT/source.json
if [ -n "${GATE2:-}" ]; then
  cos2=$(grep -oE 'phase\) [-0-9.]+' $OUT/check2.txt | awk '{print $2}'); lab2=$(grep -oE 'labelled \([0-9.]+%' $OUT/check2.txt | tr -d '(%' | awk '{print $2}')
  jq --arg cr "$(readlink -f $OUT/codes2)" --argjson cos "${cos2:-0}" --argjson lab "${lab2:-0}" --arg g "$GATE2" \
    '.codes.root = $cr | .pseudo = {agree_cos: $cos, labelled_pct: $lab, gate: $g}' $OUT/source.json > $OUT/source2.json
fi
echo "$NAME: $(tr '\n' ' ' < $OUT/check.txt | cut -c1-600)"
