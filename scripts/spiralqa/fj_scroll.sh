#!/usr/bin/env bash
# fj_scroll.sh OUTDIR FIBRES SRC Z0 Z1 [NZ] [PAR]: the fibre judge (ufsm fiber-judge) over z slabs [Z0, Z1) of NZ (512) level-2 voxels,
# PAR (4) slabs at a time, one thread each. SRC: "--wraps DIR --first K --last K" or "--set FITSET" (quoted). Rows -> OUTDIR/s<z>.tsv,
# logs OUTDIR/s<z>.log; then OUTDIR/tiles.tsv (all tiles: box, H + V arc m, H + V persistent, rate /m) and the totals.
# Env: U (ufsm), UM (2.4: the level-0 um), LEVEL (2), FJX (extra flags, e.g. --tr T.json).
set -u
OD=${1:?out}; FIB=${2:?fibres}; SRC=${3:?src}; Z0=${4:?}; Z1=${5:?}; NZ=${6:-512}; PAR=${7:-4}
U=${U:-$HOME/ufsm-spiralqa/build/ufsm}; mkdir -p $OD
for z in $(seq $Z0 $NZ $((Z1 - 1))); do
  [ -s $OD/s$z.log ] && grep -q '^fiber-judge: .* tiles' $OD/s$z.log && continue
  while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 5; done
  ~/bin/lowprio ~/bin/memcap -m 8G $U fiber-judge $FIB $SRC --um ${UM:-2.4} --level ${LEVEL:-2} --slab $z,$NZ --d-mm 0.3,0.6,1.2,2.4 ${FJX:-} --out $OD/s$z.tsv > $OD/s$z.log 2>&1 &
done
wait
# per tile: H (cols 5-7) + V (cols 12-14) arc / persistent (layout of fiber-judge --out with 4 pair distances: per family arc, exc, pers, 4 pairs)
awk -F'\t' 'FNR == 1 { next } { ah = $5; ph = $7; av = $12; pv = $14; a = ah + av; printf "%s\t%.4f\t%.0f\t%.1f\n", $1, a, ph + pv, (a > 0 ? (ph + pv) / a : 0) }' $OD/s*.tsv | sort -t, -k1,1n -k2,2n > $OD/tiles.tsv
awk -F'\t' '{ a += $2; p += $3; if ($2 >= 0.5) { n++; r[n] = $4 } } END { if (!n || !(a > 0)) { print "fj_scroll: no tiles"; exit } asort(r); printf "fj_scroll: %d tiles with >= 0.5 m of fibre, %.1f m, persistent %.0f = %.1f /m; tile rate median %.0f q75 %.0f q90 %.0f /m\n", n, a, p, p / a, r[int(n / 2) + 1], r[int(0.75 * n) + 1], r[int(0.9 * n) + 1] }' $OD/tiles.tsv
