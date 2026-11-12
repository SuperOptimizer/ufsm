#!/bin/bash
# scripts/inkgt/baseline.sh PRED [GPU]: ink baselines on the held-out ink crops (16 Paris 4 wbench/inkeval/crops + 8 PHerc0139 / 1667 inkgt/evcrops)
# against the human labels (2026-11-11).
#   PRED = ink3dext: the published dino3d 3-D ink net (ink3d-ext, ckpt_78k_fullsup) over each crop box -> 4.8 um prob254;
#   PRED = ct: the CT itself (level 1) as the score (a density-only floor);  PRED = ckpt:PATH:CH: channel CH of a ufsm checkpoint (predict, 4.8 um).
# Scores: 3-D  ink-vscore vs the crop's ink-splat --face reference (inkgt/evgt/<crop>.zarr)  -> $O/PRED/vscores.tsv
#         2-D  render along the crop surface (--depth 4 at 4.8 um) + ink-score vs labels.tif  -> $O/PRED/scores.tsv   (not for ct)
set -u
P=${1:?ink3dext|ct|ckpt:PATH:CH}; GPU=${2:-1}; U=${U:-/home/forrest/ufsm-inkgt/build/ufsm}; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
D=/vesuvius/ufsm/inkgt; O=${O:-$D/baseline}; tag=${P%%:*}; [ "$tag" = ckpt ] && tag=ckpt_$(basename $(echo $P | cut -d: -f2) .ckpt)_ch$(echo $P | cut -d: -f3)
o=$O/$tag; mkdir -p $o; touch $o/vscores.tsv $o/scores.tsv; CACHE=/vesuvius/ufsm/cache
P4K=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
for c in /vesuvius/ufsm/wbench/inkeval/crops/w*_? $D/evcrops/PHerc*; do
  n=$(basename $c); grep -qP "^$n\t" $o/vscores.tsv && continue
  read key um box gk < <(python3 -I -c "import json,sys; j=json.load(open(sys.argv[1])); u = j.get('um', 2.4); print(j.get('ct','$P4K'), u, ','.join(map(str, j['box_l0'])), '%g' % round(2 * u, 3))" $c/crop.json)
  if [ "$P" = ct ]; then pr=$M; pk=$key/1
  elif [ "$P" = ink3dext ]; then v=$o/vol/$n; pr=$v; pk=$gk
    [ -s $v/$gk/zarr.json ] || ~/bin/memcap -m 24G $U ink3d-ext --weights /vesuvius/ufsm/teachers/dino3d/ckpt_78k_fullsup.pth --ct $M $key --ct-um $um --box $box --out $v --out-um $gk --gpu $GPU --cache $CACHE > $o/ink_$n.log 2>&1 || { echo "FAIL predict $n"; continue; }
  else ck=$(echo $P | cut -d: -f2); ch=$(echo $P | cut -d: -f3); v=$o/vol/$n; pr=$v; pk=$gk
    [ -s $v/$pk/zarr.json ] || ~/bin/memcap -m 40G $U predict $ck $M $key $v --um $um --level 1 --box $box --channel $ch --gpu $GPU --threads 6 --cache $CACHE > $o/ink_$n.log 2>&1 || { echo "FAIL predict $n"; continue; }
  fi
  $U ink-vscore --gt $D/evgt/$n.zarr $gk --pred $pr $pk --box $box --row $n --cache $CACHE 2>> $o/vscore.log >> $o/vscores.tsv
  if [ "$P" != ct ]; then
    s=$o/surf/$n; mkdir -p $s/w; python3 -I - $c $s/w/$n <<'PY'
import sys, os, shutil, numpy as np, tifffile
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
for a in 'xyz':
    v = tifffile.imread(os.path.join(src, a + '.tif')).astype(np.float32); tifffile.imwrite(os.path.join(dst, a + '.tif'), np.where(v >= 0, (v + 0.5) / 2 - 0.5, -1).astype(np.float32))
shutil.copy(os.path.join(src, 'meta.json'), dst)
PY
    nice -n 5 $U render $s/w $s/r --volume $pr $pk --depth 4 --threads 8 --gpu 0 --cache $CACHE > $o/render_$n.log 2>&1 || { echo "FAIL render $n"; continue; }
    mp=$(ls $s/r/*.tif 2>/dev/null | grep -v _fill | head -1)
    [ -n "$mp" ] && $U ink-score --surface $c --labels $c/labels.tif --mask $c/mask.tif --map ink=$mp --row $n 2>> $o/score.log | grep -P "^$n\t" >> $o/scores.tsv
  fi
done
awk -F'\t' '{n++; a+=$6; f+=$8} END {if (n) printf "3-D mean over %d crops: AUC %.4f best F1 %.4f\n", n, a / n, f / n}' $o/vscores.tsv
[ -s $o/scores.tsv ] && awk -F'\t' '{n++; f+=$5; a+=$10} END {if (n) printf "2-D mean over %d crops: f1best %.4f auc %.4f\n", n, f / n, a / n}' $o/scores.tsv
