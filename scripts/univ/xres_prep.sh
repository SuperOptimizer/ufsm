#!/usr/bin/env bash
# scripts/univ/xres_prep.sh [SCROLLS]: the cross-resolution wbench set (universal-net eval harness, 2026-11-11).
# For each scroll with 9.36 um winding GT and a registered 2.2-2.4 um scan (ufsm reg maps from the volcomp mirror, held-out p50 ~1 um):
#   - the 9.36 um wbench boxes mapped into the fine scan (reg map --box, level-0 voxels) -> fine level-2 boxes (aligned to 32, 512^3 around the centre)
#     -> $X/boxes-xres.tsv (scroll x0139 / x0814 / x0500P2, wbench boxes.tsv format, the 9.36 box's note kept);
#   - the GT codes and phase warped onto the fine scan's level-3 grid (reg warp, nearest; ~19.2 / 17.7 um, the 18.72 um labels' scale) over the
#     union of the scroll's boxes -> $X/<scroll>/{codes,phase}/<um>;
#   - the umbilicus axis mapped into the fine scan's level-0 voxels -> $X/<scroll>/axis.json.
# wbench.sh then scores a checkpoint on the same sheets from the fine scan (BT=$X/boxes-xres.tsv, scroll entries x0139 / x0814 / x0500P2):
# paired against the 9.36 um rows of the same boxes, that is the cross-res column of scripts/univ/eval.sh.
set -u
SC=${1:-PHerc0139 PHerc0814 PHerc0500P2}
U=${U:-/home/forrest/ufsm-latres/build/ufsm}; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
D=${D:-/vesuvius/ufsm/reg/xres}; X=${X:-/vesuvius/ufsm/wbench/xres}; GT=/vesuvius/ufsm/gt; BT=/vesuvius/ufsm/wbench/boxes.tsv
T=${T:-16}; CACHE=${CACHE:-/vesuvius/ufsm/wbench/cache}; mkdir -p $X
declare -A REG=([PHerc0139]=PHerc0139-20260102150214 [PHerc0814]=PHerc0814 [PHerc0500P2]=PHerc0500P2)
declare -A GTD=([PHerc0139]=PHerc0139-wind-20261016 [PHerc0814]=PHerc0814-wind-20261019 [PHerc0500P2]=PHerc0500P2-wind-20261019)
declare -A AXF=([PHerc0139]=PHerc0139-umb-scan.json [PHerc0814]=PHerc0814-20250804134230.json [PHerc0500P2]=PHerc0500P2-20250820143440.json)
[ -s $X/boxes-xres.tsv ] || printf '# name\tscroll\tz\ty\tx\tnz\tny\tnx\tlevel\tnote  (fine-scan level-2 voxels: the 9.36 um wbench box of the same name mapped by ufsm reg; xres_prep.sh)\n' > $X/boxes-xres.tsv
for S in $SC; do
  r=$D/${REG[$S]}/reg.json; g=$GT/${GTD[$S]}; o=$X/$S; mkdir -p $o
  fk=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['fine']['key'])" $r); fu=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['fine']['um'])" $r)
  l3=$(python3 -I -c "print(round(8 * $fu, 3))"); ck=$(ls $g/codes | grep -E '^[0-9.]+$' | head -1)
  # boxes
  awk -F'\t' -v s=$S '$2 == s' $BT | while IFS=$'\t' read -r b s_ z y x nz ny nx lv note; do
    grep -qP "^x$b\t" $X/boxes-xres.tsv && continue
    m=$($U reg map $r --box $z,$y,$x,$nz,$ny,$nx 2>/dev/null | tail -1)
    python3 -I - "$m" "x$b" "x$S" "$note" >> $X/boxes-xres.tsv <<'PY'
import sys
z, y, x, nz, ny, nx = map(int, sys.argv[1].split(','))
c = [(z + nz / 2) / 4, (y + ny / 2) / 4, (x + nx / 2) / 4]   # level-2 centre
o = [max(0, int(round((v - 256) / 32)) * 32) for v in c]
print('\t'.join([sys.argv[2], sys.argv[3]] + [str(v) for v in o] + ['512', '512', '512', '2', sys.argv[4] + '; cross-res of ' + sys.argv[2][1:]]))
PY
  done
  # union of the scroll's boxes in level-3 voxels
  ub=$(awk -F'\t' -v s=x$S '$2 == s { z0 = $3 / 2; y0 = $4 / 2; x0 = $5 / 2; z1 = z0 + $6 / 2; y1 = y0 + $7 / 2; x1 = x0 + $8 / 2;
       if (!n++ || z0 < a) a = z0; if (n == 1 || y0 < b) b = y0; if (n == 1 || x0 < c) c = x0; if (z1 > d) d = z1; if (y1 > e) e = y1; if (x1 > f) f = x1 }
       END { printf "%d,%d,%d,%d,%d,%d", a - 32, b - 32, c - 32, d - a + 64, e - b + 64, f - c + 64 }' $X/boxes-xres.tsv)
  for k in codes phase; do
    src=$g/$k; [ -d $src/$ck ] || src=$g/${k}; key=$ck
    [ -e $o/$k/$l3/zarr.json ] && continue; mkdir -p $o/$k
    nice -n 5 $U reg warp $r $src $key $ck $o/$k/$l3 --level 3 --box $ub --fill 255 --cache $CACHE --threads $T > $o/warp_$k.log 2>&1 || { echo "$S warp $k failed: $o/warp_$k.log"; continue; }
  done
  # axis
  [ -s $o/axis.json ] || python3 -I - $GT/axis/${AXF[$S]} $r $U $o/axis.json <<'PY'
import json, sys, subprocess
a = json.load(open(sys.argv[1])); cp = a['control_points']
pts = ['%r,%r,%r' % (p['x'], p['y'], p['z']) for p in cp]
out = []
for i in range(0, len(pts), 200):
    r = subprocess.run([sys.argv[3], 'reg', 'map', sys.argv[2]] + pts[i:i + 200], capture_output=True, text=True, check=True)
    out += [l for l in r.stdout.split('\n') if l.count(',') == 2]
assert len(out) == len(cp), (len(out), len(cp))
a['control_points'] = [dict(zip(('x', 'y', 'z'), map(float, l.split(',')))) for l in out]
a['control_points'].sort(key=lambda p: p['z']); a['ufsm_note'] = 'mapped from %s through %s (ufsm reg map, coarse L0 -> fine L0)' % (sys.argv[1], sys.argv[2])
json.dump(a, open(sys.argv[4], 'w'))
PY
  echo "$S: fine $fk ($fu um), labels at level 3 = $l3 um, union box $ub; $(grep -c "x$S" $X/boxes-xres.tsv) boxes"
done
