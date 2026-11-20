#!/bin/bash
# latink.sh O GPU OUT: the unroll's ink stage (INKCMD) with the latent universal ink head (user 11-13: "forget about ift3"), inkB's recipe
# (sched/inkwatch/u1667ink-v3.sh, GATE=1): predict --heads ink,c12,c13 at the run's CT level over its box, window 544 halo 16, on two tile
# grids (origin 0 and 256), each head blended with ufsm ink-blend (sin^2 tile weights, tile 512) -> OUT (ink, key = the level's um) and
# O/ink/c12.zarr, O/ink/c13.zarr (p_recto, p_verso; strip_refill.sh gates the rendered ink by max of the two when INKGATE names them).
# The grid-256 pass writes 256 shards (inkB 11-13: with 512 shards each shard overlapped 8 shifted windows, ~7x the windows of grid 0).
# LATWROUND=1 (OFF by default; infer audit #8): predict --window-round 1, 544 / 16 -> 640 / 64 with the same 512 interior for nets whose
# window must be a multiple of 128 (scratch2, nlev 6); a no-op for u1-class nets. Needs a predict with --window-round (inferfix).
# LATINK_BOX=z,y,x,nz,ny,nx (CT level voxels, origins on the 512 grid): ink only over that box, not the run's (Paris 4 on a 200 GB pod: the
# whole box's three lossless heads are ~0.4 TB, 0332's 51 GB; the desk's P4 u1 ink was the band z 8192 + 2560 too). The renders show
# ink inside it only.
# LATINKQ (default 4 since 2026-11-17; 0 = lossless): volcomp q of the grid passes and the blends (q > 0: encoded from the GPU's levels).
# q 4 vs q 0 on 1667 (band z 3328 + 512, 44 wraps, pairwrap on 207947 held-out samples): ink AUC 0.6574 vs 0.6572 (+0.0002, CI -0.0006..
# +0.0010), renders mean |diff| 1.2-1.5 of 255 (p99 6), ink stage 118 -> 91 s wall, 645 -> 358 CPU s, stores 4.5 GB -> 0.57 GB. 1667 1024^3 box, three heads (podunroll 11-17): q 0 1145 MB,
# q 4 147 MB (ink 21x; mean |diff| ink 1.41 / c12 3.02 of 255, 82 flips across 128 of 1.07 G voxels), q 8 81 MB (1.84 / 4.37),
# q 32 23 MB (3.04 / 7.96, 1.9 M flips: too lossy)
# Env: LATINK_CK (default /root/pod/ckpt/inkpod-15000.ckpt, the newest inkpod snap), ITH (threads), LATINK_X (more predict args, e.g. --ema 0:
# the live weights; predict uses the EMA by default, and a stale EMA (scratch2 try9, accum 10) scored far worse than live on 1667, 11-18).
# LATWIN / LATHALO (544 / 16): the window and halo (a smaller window for a GPU budget: scratch2 try9 at 640 / 64 peaked 8.8 GB, 11-18).
# LATINK_HEADS (ink,c12,c13): the heads predicted (ink only for a net without recto / verso rows: score.sh GATEFROM brings the gate).
# LATINK_TILES=DIR (11-20, scoring in minutes): only the 512 tiles listed in DIR/tiles0.txt and DIR/tiles256.txt ("iz iy ix" per line, the
# tile at origin i * 512 + grid) are predicted, one predict --boxes run per grid into the --full-grid store (x runs merged into one box);
# the blend sees both grids' tiles at every listed point. LATINK_TILELIST (1, 11-20 fullvram): one predict per grid over the run's box with
# --tile-list (the shards outside the tiles skipped as air), so the windows pipeline as a whole-box run; 0: the --boxes runs. For a scorer whose samples all lie in those tiles: 1667's are the shards of
# an ink-splat of all its human labels (grid 0: the shard files present, 200 of 931) and the grid-256 tiles overlapping them (398).
# LATINK_CHANNEL=ink: a full-resolution (voxel) checkpoint: predict --heads takes latent-grid nets only, so predict --channel ink into
# g<grid>/ink (LATINK_HEADS must then be ink; 11-20 FR1 / ladder nets).
set -u
HL=${LATINK_HEADS:-ink,c12,c13}
O=${1:?}; G=${2:-0}; OUT=${3:?}; U=${U:-/root/pod/ufsm/build/ufsm}; CK=${LATINK_CK:-${POD:-/root/pod}/ckpt/inkpod-15000.ckpt}
# the CT: UNROLL_CT="ROOT|KEY|UM|LEVEL" (unroll.sh exports it: CTROOT / CTKEY overrides included), else the scrolls.tsv row of this run's
# scroll (O's basename, a TAG suffix allowed; was `tail -1`, the pods' one-row table: on the desk's full table that is another scroll)
if [ -n "${UNROLL_CT:-}" ]; then IFS='|' read -r R K UM LV <<< "$UNROLL_CT"
else row=$(awk -F'\t' -v o=$(basename $O) 'NR > 1 && index(o, $1) == 1 { if (length($1) > l) { l = length($1); r = $0 } } END { print r }' $O/bin/scrolls.tsv)
  [ -n "$row" ] || { echo "latink: no scrolls.tsv row for $(basename $O) (set UNROLL_CT)"; exit 1; }
  R=$(echo "$row" | cut -f2); K=$(echo "$row" | cut -f3); UM=$(echo "$row" | cut -f4); LV=$(echo "$row" | cut -f5); fi
LUM=$(awk -v u=$UM -v l=$LV 'BEGIN { printf "%.4g", u * 2 ^ l }'); BOX=${LATINK_BOX:-$(cat $O/box.txt)}; T=$O/ink/lat.tmp; TH=${ITH:-12}
echo "latink: ck $CK, $R $K level $LV ($LUM um), box $BOX"
mkdir -p $T
# pbox SHIFT: the run's box moved down onto the pass's 512-voxel tile grid (origin = SHIFT mod 512), as seamless.c pass_box: an off-grid box
# (box.txt origins are multiples of 512, the grid-256 pass's tiles start at 256 mod 512) makes predict's planner run one window per 256
# shard instead of one per 2^3 shards (predict.c kss; audit 2026-11-14: ~6-8x the windows)
pbox() { awk -F, -v s=$1 '{ for (d = 1; d <= 3; d++) { l = $d - s; l = (l >= 0 ? int(l / 512) : -int((-l + 511) / 512)) * 512 + s; if (l < 0) l = s
         o[d] = l; e = $d + $(d + 3); n[d] = e > l ? e - l : 1 } printf "%d,%d,%d,%d,%d,%d\n", o[1], o[2], o[3], n[1], n[2], n[3] }' <<< "$BOX"; }
[ -s $O/axis.json ] || { echo "latink: no $O/axis.json (the net's radial input channels need the scroll axis)"; exit 1; }
for g in 0 256; do
  [ -e $T/g$g.done ] && continue; rm -rf $T/g$g
  # (--q 0 --levels 1: the passes are blended into OUT by ink-blend, which reads level 0 only; their own pyramid / quantisation was thrown away)
  BX=(--box $(pbox $g))
  if [ -n "${LATINK_TILES:-}" ] && [ "${LATINK_TILELIST:-1}" = 1 ]; then   # (one predict over the run's box, every shard outside the listed tiles skipped)
    BX=(--box $(pbox $g) --tile-list $LATINK_TILES/tiles$g.txt --tile-list-size 512); echo "latink: grid $g: $(wc -l < $LATINK_TILES/tiles$g.txt) tiles (--tile-list)"
  elif [ -n "${LATINK_TILES:-}" ]; then   # (x runs of listed tiles -> one box line each, clamped to the volume's box)
    sort -n -k1,1 -k2,2 -k3,3 $LATINK_TILES/tiles$g.txt | awk -v g=$g -v B="$BOX" -v o=$T/g$g${LATINK_CHANNEL:+/$HL} 'BEGIN { split(B, b, ",") }
      function out() { if (n) { z0 = z * 512 + g; y0 = y * 512 + g; x0 = x * 512 + g; x1 = (x + n) * 512 + g; z1 = z0 + 512; y1 = y0 + 512
        if (z0 < b[1]) z0 = b[1]; if (y0 < b[2]) y0 = b[2]; if (x0 < b[3]) x0 = b[3]; if (z1 > b[1] + b[4]) z1 = b[1] + b[4]; if (y1 > b[2] + b[5]) y1 = b[2] + b[5]; if (x1 > b[3] + b[6]) x1 = b[3] + b[6]
        if (z1 > z0 && y1 > y0 && x1 > x0) printf "%d,%d,%d,%d,%d,%d %s\n", z0, y0, x0, z1 - z0, y1 - y0, x1 - x0, o } }
      { if (n && $1 == z && $2 == y && $3 == x + n) { n++; next } out(); z = $1; y = $2; x = $3; n = 1 } END { out() }' > $T/boxes$g.txt
    echo "latink: grid $g: $(wc -l < $T/boxes$g.txt) boxes from $(wc -l < $LATINK_TILES/tiles$g.txt) tiles"; BX=(--boxes $T/boxes$g.txt); fi
  $U predict $CK $R $K $T/g$g${LATINK_CHANNEL:+/$HL} --um $UM --level $LV "${BX[@]}" --full-grid 1 --occupancy $((LV + 3)) $([ -n "${LATINK_CHANNEL:-}" ] && echo --channel $LATINK_CHANNEL || echo --heads $HL) --gpu $G --threads $TH --shard $([ $g = 256 ] && echo 256 || echo 512) \
    --window ${LATWIN:-544} --halo ${LATHALO:-16} $([ "${LATWROUND:-0}" = 1 ] && echo --window-round 1) ${LATINK_X:-} --grid-origin $g,$g,$g --axis $O/axis.json --q ${LATINKQ:-4} --levels 1 || { echo "latink: predict grid $g failed"; exit 1; }
  for h in ${HL//,/ }; do [ -e $T/g$g/$h/zarr.json ] || { echo "latink: no $h in $T/g$g"; exit 1; }; done
  touch $T/g$g.done
done
# the three heads' blends at once (independent outputs, the same bytes): one at a time they were 3 x ~125 s + 35 s of pyramid on 0332
# (2026-11-17), ~45 % of the ink stage, the GPU idle; each blend's shard loop is read -> blend -> encode in turn, so they overlap well
bl() { local h=$1 d=$OUT; [ $h = ink ] || d=$O/ink/$h.zarr; rm -rf $d
  $U ink-blend $d --um $LUM $([ "${LATINKQ:-4}" != 0 ] && echo --q ${LATINKQ:-4}) --tile 512,512,512 --grids 0,0,0:256,256,256 $T/g0/$h $T/g256/$h --threads $TH > $T/blend.$h.log 2>&1 || { cat $T/blend.$h.log; echo "latink: ink-blend $h failed"; return 1; }
  cat $T/blend.$h.log; [ -e $d/zarr.json ] || { echo "latink: no blended $h at $d"; return 1; }; }
bp=(); for h in ${HL//,/ }; do bl $h & bp+=($!); done
rc=0; for p in "${bp[@]}"; do wait $p || rc=1; done; [ $rc = 0 ] || exit 1
rm -rf $T; echo "latink: done -> $OUT (heads $HL)"
