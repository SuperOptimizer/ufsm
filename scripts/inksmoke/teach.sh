#!/usr/bin/env bash
# scripts/inksmoke/teach.sh KIT SCROLL: the 2-D teachers on one scroll's labelled meshes, then its ink-teach windows (the smoke's arm B).
# Per mesh (KIT/inkgt/segs-SCROLL.json grouped by tifxyz, named by the first window, as ink-teach expects): ink-ext --native-only (9.6 um)
# reader3 (Reader V3) and ink9um (seeds 42 + 43 averaged), CT from the public volcomp mirror with one shared --cache, --batch pinned
# -> KIT/maps/SCROLL/<teacher>/<name>.tif; then ufsm ink-teach (AUC-weighted logit ensemble, Platt per scroll, gate p <= 0.1 | p >= 0.8,
# human px and KIT/arms/holdouts-SCROLL.json boxes + 64 voxels removed) -> KIT/teach/SCROLL/segs-teach.json. Restartable (finished maps
# and a finished ink-teach are kept). B2 (11-18): KIT/kit/extra-SCROLL.tsv (or EXTRA=FILE), lines "NAME TIFXYZ" of meshes without human
# labels: their teacher maps too, and ink-teach --extra-list (teacher labels with the labelled meshes' calibration, held-out boxes and
# px within 48 voxels of a labelled mesh removed). Env: HOD (the holdouts-SCROLL.json dir, KIT/arms; podq.sh: KIT/arms/full), U (binary), GPU (0), LOCK (/root/gpu.lock: held per ink-ext call), BATCH (16), TH (8).
set -uo pipefail; K=${1:?KIT}; S=${2:?SCROLL}; U=${U:?U}; R=https://dl.ash2txt.org/community-uploads/forrest/volcomp
W3=$K/weights/reader-v3.pth; W42=$K/weights/ink9um-s42.pth; W43=$K/weights/ink9um-s43.pth; HO=${HOD:-$K/arms}/holdouts-$S.json
for f in $W3 $W42 $W43 $HO $K/inkgt/segs-$S.json; do [ -s $f ] || { echo "teach: missing $f"; exit 1; }; done
CT=$(python3 -I -c "import json; print([s['ct'] for s in json.load(open('$K/inkgt/sources-H0.json'))['sources'] if s['name'].startswith('$S-')][0])") || exit 1
lk() { if [ -n "${LOCK:-}" ]; then flock "$LOCK" "$@"; else "$@"; fi; }
python3 -I -c "
import json; seen = set()
for e in json.load(open('$K/inkgt/segs-$S.json'))['segments']:
    if e['tifxyz'] not in seen: seen.add(e['tifxyz']); print(e['name'], e['tifxyz'])" > $K/maps/.$S.list || exit 1
XL=${EXTRA:-$K/kit/extra-$S.tsv}   # (B2) meshes without human labels, lines "NAME TIFXYZ" (unroll wraps ...): teacher maps + ink-teach --extra-list
XA=(); if [ -s "$XL" ]; then grep -v '^#' "$XL" >> $K/maps/.$S.list; XA=(--extra-list "$XL"); echo "teach: $S: $(grep -vc '^#' "$XL") extra meshes ($XL)"; fi
for t in reader3 ink9um; do
  O=$K/maps/$S/$t; mkdir -p $O; [ $t = reader3 ] && W=$W3 || W=$W42,$W43; M=$( [ $t = reader3 ] && echo reader3 || echo ink9um )
  while read -r nm tx; do
    [ -s $O/$nm.tif ] && continue; t0=$(date +%s); tmp=$O/.tmp_$nm
    case $tmp in $K/maps/*) rm -rf -- "$tmp";; esac
    lk env CUDA_VISIBLE_DEVICES=${GPU:-0} $U ink-ext --model $M --weights $W --surface $tx --ct $R $CT --cache $K/ctcache --out $tmp --native-only \
        --batch ${BATCH:-16} --threads ${TH:-8} > $O/$nm.log 2>&1 && mv $tmp/*_native.tif $O/$nm.tif && rm -rf -- "$tmp" \
      && echo "teach: $S $t $nm $(( $(date +%s) - t0 )) s" || { echo "teach: FAIL $S $t $nm (log $O/$nm.log)"; exit 1; }
  done < $K/maps/.$S.list
done
L=$K/teach/$S; [ -s $L/segs-teach.json ] && { echo "teach: $S: ink-teach done before"; exit 0; }
case $L in $K/teach/*) rm -rf -- "$L";; esac; mkdir -p $L
$U ink-teach --segs $K/inkgt/segs-$S.json --teacher reader3=$K/maps/$S/reader3/%s.tif --teacher ink9um=$K/maps/$S/ink9um/%s.tif \
  --holdouts $HO --ct-key $CT --out $L "${XA[@]}" > $L.log 2>&1 && [ -s $L/segs-teach.json ] && { echo "teach: $S: $(grep -iE 'platt|gate|ensemble' $L.log | tr '\n' ' ' | cut -c1-300)"; exit 0; }
echo "teach: FAIL ink-teach $S (log $L.log)"; exit 1
