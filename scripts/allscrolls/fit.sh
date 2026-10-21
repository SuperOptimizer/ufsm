#!/usr/bin/env bash
# fit.sh S [BOX...]: LOCAL spiral fits of scroll S in boxes, from its published prod96 fit fields; resumable (A/<S>/state, per-box FITDONE).
# User (10-20): local fits only (global spiral fits failed on 1299 and got ~20% ids on a Paris 4 cross-section), fastest path to
# rendered cm2: the best boxes first, the cleanest sheets of each box exported for rendering as soon as the box is done.
#  1 fields: the fit-field zarr copied to A/<S>/fields (whole 512^3 shard files, curl --parallel; absent shards = air)
#  2 boxes: N^3 field cells (desk N 1024 = 2048 native voxels, 1.8-1.9 cm; laptop N 512 = 0.9 cm: its 6 GB job cap OOMs larger
#    boxes), stride N - 128, >= 10% material on CT level 5; order (A/<S>/boxes-N.tsv): high material, mid radius (0.55 of the outermost
#    material box's radius from the scroll axis), middle z: score = material - 2 |r / rmax - 0.55| - |z - zmid| / Z
#  3 per box (MAXBOX of them this pass, default 2): CT planes (ct-fields, CT level 1), probe fits of the centre sub-box with the axis
#    2000 native voxels away along its mean sheet normal on both sides (the lower loss wins; or CORE=1 with A/<S>/core-line.json: the
#    core-line half box of that side, ufsm coreline box), the box fit (--core 0, --w-ct 1 --ct-phi 0 --ct-flood 0 --uw-repair 8
#    --w-normal 3, wind4 off = default), wraps at the native um, surf-stats --per-wrap --ct (on papyrus), self-gate (pick: the coherent
#    part of each wrap; REPICK=1 gates boxes fitted before the gate) -> the TOPW (3) best sheets
#    (area x on-papyrus x (1 - phase-residual share), >= 0.5 cm2, on papyrus >= MINPAP 0.5) -> A/<S>/SURF/<box>/<wrap> and a row in
#    A/<S>/SURF/list.tsv (surface, kind, area cm2, on papyrus); SURF = surf (desk) or surf-l (laptop; pushed to the desk for render.sh)
#  4 STITCH=1: ufsm stitch align / quality / solve / sheets over the fitted boxes (only the agreeing neighbours join: --min-agree 0.5)
#  5 GTCHECK=1 (desk): the published segments of this scan (AWS "-on-<scan id>-" tifxyz) vs a raster of the fitted wraps (ufsm wrapdist)
D=$(cd "$(dirname "$0")" && pwd); . $D/common.sh
s=$1; shift; read -r scan um fz <<< "$(scan_of $s)"; [ -n "${fz:-}" ] || { echo "fit.sh: $s not in $A/scrolls.tsv"; exit 2; }
S=$A/$s; F=$S/fields/$fz; CK=$s/volumes/$scan.zarr
if [ $HOST = desk ]; then N=${N:-1024}; TH=${TH:-12}; SURF=surf; else N=${N:-512}; TH=${TH:-8}; SURF=surf-l; fi
ST=$((N - 128)); BX=$S/boxes-$N; mkdir -p $S/state $BX $S/$SURF
FITA=(--w-ct 1 --ct-phi 0 --ct-flood 0 --uw-repair 8 --w-normal 3 --core 0 --threads $TH --quiet)   # (whole box; --slab must be a multiple of the knot spacing)
# FIT ARGS..: ufsm fit; laptop: under the shared GPU lock (ink-model porting agent) and capped.sh MEM; desk: FITCPU=1 --cpu, else the emptier GPU
FIT() { if [ $HOST = laptop ]; then flock $HOME/ufsm-data/gpu.lock env MEM=${MEM:-6G} UFSM_FIT_THREADS=$TH $HOME/ufsm-data/capped.sh $U fit "$@"
        elif [ "${FITCPU:-0}" = 1 ]; then UFSM_FIT_THREADS=$TH nice -n 5 $U fit "$@" --cpu
        else local g=${FITGPU:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)}
             UFSM_FIT_THREADS=$TH nice -n 5 $U fit "$@" --gpu $g; fi; }
g1() { grep -oE "$1" "$2" 2> /dev/null | tail -1 | grep -oE '[0-9.]+$'; }
CT1=(--ct $R $CK/1 --ct-um $(awk -v u=$um 'BEGIN { print 2 * u }') --ct-level 80 --ct-range 15 --cache $S/ctcache-$HOST)

exec 9> $BX.lock; flock 9   # (steps 1-2: one fetcher / list builder; the other fit.sh of the scroll waits)
# ---- 1 fields
if ! done_ $s fields || [ ! -s $F/fields.json ]; then
  mkdir -p $F; P=48
  for f in fields.json README.json; do curl -sf --retry 5 $R/$s/representations/predictions/fit-fields/$fz/$f -o $F/$f; done
  [ -s $F/fields.json ] || { echo "fit.sh: no fields.json for $s"; exit 1; }
  for p in pc ps t0 t1 t2 t3 t4 t5 sp w; do
    base=$R/$s/representations/predictions/fit-fields/$fz/$p; dst=$F/$p; mkdir -p $dst
    [ -f $dst/.complete ] && continue
    curl -sf --retry 5 $base/zarr.json -o $dst/zarr.json || { echo "fit.sh: no $p/zarr.json"; exit 1; }
    J=$(tr -d ' \n' < $dst/zarr.json); read -r Z Y X <<< "$(echo "$J" | grep -oE '"shape":\[[0-9,]+\]' | head -1 | grep -oE '[0-9]+' | tr '\n' ' ')"
    read -r cz <<< "$(echo "$J" | grep -oE '"chunk_shape":\[[0-9,]+\]' | head -1 | grep -oE '[0-9]+' | head -1)"
    nz=$(( (Z + cz - 1) / cz )) ny=$(( (Y + cz - 1) / cz )) nx=$(( (X + cz - 1) / cz ))
    for pass in 1 2 3; do
      cfg=$dst/.curl.cfg; : > $cfg; n=0
      for ((z = 0; z < nz; z++)); do for ((y = 0; y < ny; y++)); do for ((x = 0; x < nx; x++)); do
        k=c/$z/$y/$x; [ -f $dst/$k ] || [ -f $dst/$k.absent ] && continue
        printf 'url = "%s"\noutput = "%s.part"\n' "$base/$k" "$dst/$k" >> $cfg; n=$((n + 1))
      done; done; done
      [ $n = 0 ] && break
      curl -s --http1.1 --parallel --parallel-max $P --create-dirs --retry 3 -K $cfg -w '%{http_code} %{size_download} %{filename_effective}\n' > $dst/.curl.log 2>&1
      while read -r code sz f; do t=${f%.part}
        case $code in 200) [ -s "$f" ] && [ $(stat -c %s "$f") = "$sz" ] && mv "$f" "$t" || rm -f "$f";; 404) rm -f "$f"; : > "$t.absent";; *) rm -f "$f";; esac
      done < $dst/.curl.log
    done
    [ $n = 0 ] || { echo "fit.sh: $s $p: $n shards missing after 3 passes"; exit 1; }
    echo "$nz x $ny x $nx" > $dst/.complete; rm -f $dst/.curl.cfg $dst/.curl.log
  done
  mark $s fields "$(du -sh $F | cut -f1)"
fi
read -r SZ SY SX <<< "$(jq -r '.shape | map(tostring) | join(" ")' $F/fields.json)"; H=$(jq -r .h $F/fields.json)
[ -s $S/axis.tsv ] || jq -r '.axis[] | @tsv' $F/fields.json > $S/axis.tsv   # line z + 1: the axis y x at native z

# ---- 2 box list (corner in cells, material % from CT level 5 = 32 native voxels, radius share), best boxes first
if [ ! -s $BX.tsv ]; then
  t=$BX.tsv.tmp; : > $t; n5=$((N * H / 32))
  for ((z = 0; z + 256 <= SZ; z += ST)); do for ((y = 0; y + 256 <= SY; y += ST)); do for ((x = 0; x + 256 <= SX; x += ST)); do
    pct=$($U read $R $CK/5 $((z * H / 32)) $((y * H / 32)) $((x * H / 32)) $n5 $n5 $n5 /dev/null 2>&1 | grep -oE 'nonzero [0-9.]+' | awk '{print $2}')
    [ -n "$pct" ] || { echo "fit.sh: CT level 5 read failed at $z $y $x"; exit 1; }
    awk -v p=$pct 'BEGIN { exit !(p >= 10) }' || continue
    zc=$(( (z + N / 2) * H )); read -r ay ax <<< "$(sed -n "$((zc < SZ * H ? zc + 1 : SZ * H))p" $S/axis.tsv)"
    r=$(awk -v y=$(( (y + N / 2) * H )) -v x=$(( (x + N / 2) * H )) -v ay=$ay -v ax=$ax 'BEGIN { printf "%.0f", sqrt((y - ay) ^ 2 + (x - ax) ^ 2) }')
    printf 'b%05d_%04d_%04d\t%d\t%d\t%d\t%s\t%s\n' $z $y $x $z $y $x $pct $r >> $t
  done; done; done
  rmax=$(awk -F'\t' '$5 >= 50 && $6 > m { m = $6 } END { print m + 1 }' $t)
  awk -F'\t' -v OFS='\t' -v rmax=$rmax -v zm=$((SZ / 2)) -v Z=$SZ -v n=$N '{ rn = $6 / rmax; d = rn - 0.55; dz = $2 + n / 2 - zm;
      sc = $5 / 100 - 2 * (d < 0 ? -d : d) - (dz < 0 ? -dz : dz) / Z; printf "%.4f\t%s\t%s\t%s\t%s\t%s\t%.2f\n", sc, $1, $2, $3, $4, $5, rn }' $t |
    sort -t$'\t' -k1,1 -gr | cut -f2- > $BX.tsv; rm -f $t
  echo "$s: $(wc -l < $BX.tsv) boxes of $N cells with >= 10% material; best $(head -1 $BX.tsv | tr '\t' ' ')"
fi
exec 9>&-

# fj DIR Z Y X NZ NY NX AY AX: a zarr-backed fields dir for the box (corner Z Y X cells, size NZ NY NX; planes linked to F) with the
# constant box-local axis AY AX (native voxels)
fj() {
  local d=$1 z=$2 y=$3 x=$4 nz=$5 ny=$6 nx=$7; mkdir -p $d
  for p in pc ps t0 t1 t2 t3 t4 t5 sp w; do ln -sfn $F/$p $d/$p; done
  jq -c --argjson z $z --argjson y $y --argjson x $x --argjson nz $nz --argjson ny $ny --argjson nx $nx --argjson h $H --argjson ay $8 --argjson ax $9 \
    '{format: "zarr", origin: [$z * $h, $y * $h, $x * $h], shape: [$nz, $ny, $nx], h: $h, um: .um, zarr_offset: [$z, $y, $x],
      axis: [range(0; $nz * $h + 1) | [$ay, $ax]]}' $F/fields.json > $d/fields.json
}
# several fit.sh of one scroll share the boxes (claims: <box>.claim = host pid; a dead claimant's claim is taken over)
claim() { local c=$BX/$1.claim
  if ! mkdir $c 2> /dev/null; then read -r h p < $c/who 2> /dev/null; [ "${h:-}" = $HOST ] && ! kill -0 ${p:-0} 2> /dev/null || return 1; fi
  echo "$HOST $$" > $c/who; }

# pick BOX KIND: gate the box's wraps to their coherent part (ufsm self-gate: on the network's sheet (conf >= 0.1, |e| < 0.25 turn),
# on papyrus (CT level 1), no fault (|grad W| <= 3 x the sheet density, spacing to the next wrap 0.5..1.5), the box's phase coherence
# vs the network >= 0.5 and not reversed (stitch quality), eroded 2, stretches >= 0.1 cm2: surfaces crossing crumpled layers drop out)
# -> boxes-N/<b>/gated; the TOPW (3) wraps with the most kept area x on papyrus (>= 0.5 cm2 kept, on papyrus >= MINPAP) -> SURF/<b>/<w>
# and SURF/list.tsv rows: surface, kind, kept cm2, on papyrus, masked share (1 - kept / area)
pick() {
  local b=$1 kind=$2 d=$BX/$1 fd=$BX/$1/f
  printf '{"um": %s, "boxes": [{"name": "%s", "dir": "%s"}], "edges": []}\n' $um $b $d > $d/box.json
  C $U stitch quality $d/box.json $d/box_q.json --fields $F --threads $TH > $d/quality.log 2>&1
  rm -rf $d/gated; C $U self-gate $d/tifxyz $d/gated --fields $fd --model $d/fit "${CT1[@]}" $([ -s $d/box_q.json ] && echo --quality $d/box_q.json) \
    --min-conf 0.1 --max-e 0.25 --erode 2 --min-area 0.1 --per-wrap --threads $TH > $d/gate.log 2>&1 || { echo "pick: $b self-gate failed"; return 1; }
  : > $d/best.tsv
  awk '$1 ~ /^w[0-9]+$/ && NF >= 5 {print $1, $4, $5}' $d/gate.log | while read -r w a ka; do   # wrap, area, kept area
    op=$(awk -v w=$w '$1 == w && NF >= 8 {print $NF}' $d/stats.log)
    awk -v a=${a:-0} -v ka=${ka:-0} -v op=${op:-0} -v w=$w -v m=${MINPAP:-0.5} 'BEGIN { if (ka >= 0.5 && op >= m) printf "%.4f\t%s\t%.3f\t%.3f\t%.3f\n", ka * op, w, ka, op, (a > 0 ? 1 - ka / a : 1) }' >> $d/best.tsv
  done
  sort -t$'\t' -k1,1 -gr $d/best.tsv | head -n ${TOPW:-3} | while IFS=$'\t' read -r sc w ka op ms; do
    [ -d $d/gated/$w ] || continue; mkdir -p $S/$SURF/$b; rm -rf $S/$SURF/$b/$w; cp -r $d/gated/$w $S/$SURF/$b/
    printf '%s/%s\t%s-%s\t%s\t%s\t%s\n' $b $w $kind $N $ka $op $ms >> $S/$SURF/list.tsv
  done
  [ -s $d/best.tsv ] || pick_loose $b $kind
  touch $d/PICKED
}
# pick_loose BOX KIND (user 10-21: cm2 for every box): when the strict gate keeps no sheet, a looser one (no phase-confidence floor,
# |e| < 0.5 turn, no coherence map, erode 1, stretches >= 0.05 cm2) -> boxes-N/<b>/gated-loose; its LOOSEW (2) wraps with the most kept
# area (>= 0.2 cm2) join the list as kind "loose-..."
pick_loose() {
  local b=$1 kind=$2 d=$BX/$1 fd=$BX/$1/f
  rm -rf $d/gated-loose; C $U self-gate $d/tifxyz $d/gated-loose --fields $fd --model $d/fit "${CT1[@]}" \
    --min-conf 0 --max-e 0.5 --erode 1 --min-area 0.05 --per-wrap --threads $TH > $d/gate-loose.log 2>&1 || { echo "pick_loose: $b self-gate failed"; return 1; }
  awk '$1 ~ /^w[0-9]+$/ && NF >= 5 && $5 >= 0.2 {print $5, $1, $4}' $d/gate-loose.log | sort -gr | head -n ${LOOSEW:-2} | while read -r ka w a; do
    [ -d $d/gated-loose/$w ] || continue; op=$(awk -v w=$w '$1 == w && NF >= 8 {print $NF}' $d/stats.log)
    mkdir -p $S/$SURF/$b; rm -rf $S/$SURF/$b/${w}l; cp -r $d/gated-loose/$w $S/$SURF/$b/${w}l
    printf '%s/%sl\tloose-%s-%s\t%s\t%s\t%s\n' $b $w $kind $N $ka ${op:-na} $(awk -v a=$a -v k=$ka 'BEGIN { printf "%.3f", (a > 0 ? 1 - k / a : 1) }') >> $S/$SURF/list.tsv
  done
  touch $d/LOOSE
}
# LOOSE=1: boxes picked before the loose fallback existed whose strict pick found nothing
if [ "${LOOSE:-0}" = 1 ]; then
  for d in $BX/*/PICKED; do [ -e "$d" ] || continue; d=$(dirname "$d"); b=$(basename "$d")
    [ -s $d/best.tsv ] || [ -e $d/LOOSE ] || pick_loose $b $(cut -f4 $d/FITDONE | sed 's/[pm]$//')
  done
fi
# boxes fitted before the gate (no PICKED): their ungated sheets leave the list, the gated ones join it (REPICK=1)
if [ "${REPICK:-0}" = 1 ]; then
  for d in $BX/*/FITDONE; do [ -e "$d" ] || continue; d=$(dirname "$d"); b=$(basename "$d"); [ -e $d/PICKED ] && continue
    [ -s $S/$SURF/list.tsv ] && grep -v "^$b/" $S/$SURF/list.tsv > $S/$SURF/list.tmp; mv $S/$SURF/list.tmp $S/$SURF/list.tsv 2> /dev/null
    if [ -s $S/renders/index.tsv ]; then   # the box's ungated renders go (render.sh redoes the gated sheets)
      rm -f $S/renders/w_${s}_${b}_*; rm -rf $S/sheets/w_${s}_${b}_*
      grep -v "^$SURF/$b/" $S/renders/index.tsv > $S/renders/index.tmp; mv $S/renders/index.tmp $S/renders/index.tsv
      [ -s $S/sheets.tsv ] && { grep -v "	$b	" $S/sheets.tsv > $S/sheets.tmp; mv $S/sheets.tmp $S/sheets.tsv; }
    fi
    k=$(cut -f4 $d/FITDONE | sed 's/[pm]$//'); pick $b ${k:-local}
  done
fi

# ---- 3 box fits, best boxes first
# the box order (user 10-20, "likely text"): priority boxes first (A/<S>/priority-N.txt, one box name per line: e.g. the boxes over a
# published segment with known letters), then the boxes ranked for text: material, the middle / outer wraps (radius share near 0.65),
# round robin over the z rows (each row's best box, the rows from the middle outward, then each row's second ...) so a pass of
# MAXBOX boxes spreads over the whole height. PRIO_ONLY=1: the priority boxes only.
ranked() { awk -F'\t' -v OFS='\t' -v zm=$((SZ / 2)) -v n=$N '{ d = $6 - 0.65; sc = $5 / 100 - 2 * (d < 0 ? -d : d); dz = $2 + n / 2 - zm; print $2, sc, (dz < 0 ? -dz : dz), $1 }' $BX.tsv |
  sort -t$'\t' -k1,1n -k2,2gr | awk -F'\t' -v OFS='\t' '{ r = ($1 == z) ? r + 1 : 1; z = $1; print r, $3, $4 }' | sort -t$'\t' -k1,1n -k2,2n | cut -f3; }
ls_boxes() {
  if [ $# -gt 0 ]; then printf '%s\n' "$@"; return; fi
  [ -s $S/priority-$N.txt ] && cat $S/priority-$N.txt
  [ "${PRIO_ONLY:-0}" = 1 ] || ranked | head -n ${MAXBOX:-2}
}
for b in $(ls_boxes "$@"); do
  d=$BX/$b; [ -e $d/FITDONE ] || [ -e $d/FITDONE.failed ] && continue; claim $b || continue
  read -r z y x pct rn <<< "$(awk -v b=$b '$1 == b {print $2, $3, $4, $5, $6}' $BX.tsv)"
  nz=$((SZ - z < N ? SZ - z : N)) ny=$((SY - y < N ? SY - y : N)) nx=$((SX - x < N ? SX - x : N)); t0=$(date +%s)
  rm -rf $d; mkdir -p $d/p; fail() { echo "FAILED $1" > $d/FITDONE.failed; rm -f $d/*/*.raw; }
  # probe: the centre sub-box (half the box in z, 3/8 in y / x), both sides; mean sheet normal + the better side
  pn=$((N * 3 / 8)); pz=$((z + nz / 4)) py=$((y + (ny - pn) / 2)) px=$((x + (nx - pn) / 2))
  fj $d/pf $pz $py $px $((nz / 2)) $((ny < pn ? ny : pn)) $((nx < pn ? nx : pn)) 0 0
  for sd in p m; do Dn=2000; [ $sd = m ] && Dn=-2000; FIT $d/pf --out $d/p/$sd --core 0 --axis-normal $Dn --quiet > $d/p/$sd.log 2>&1; done
  lp=$(g1 'loss [0-9.]+' $d/p/p.log); lm=$(g1 'loss [0-9.]+' $d/p/m.log)
  [ -n "${lp:-}${lm:-}" ] || { fail "both probe fits"; continue; }
  sd=p; sg=-1; awk -v a="${lp:-9e9}" -v b="${lm:-9e9}" 'BEGIN { exit !(b < a) }' && { sd=m; sg=1; }
  read -r ny_ nx_ <<< "$(grep -m1 'mean normal' $d/p/$sd.log | sed -E 's/.*mean normal z [-0-9.]+ y ([-0-9.]+) x ([-0-9.]+);.*/\1 \2/')"
  [[ "${ny_:-}" =~ ^-?[0-9.]+$ && "${nx_:-}" =~ ^-?[0-9.]+$ ]] || { fail "probe normal (${ny_:-} ${nx_:-})"; continue; }
  if [ "${CORE:-0}" = 1 ] && [ -s $S/core-line.json ]; then   # core-line half box on the probe's side
    C $U coreline box $S/core-line.json --fields $F --box $z $y $x $nz $ny $nx --side $sg --out $d/f > $d/core.log 2>&1 || { fail "coreline box"; continue; }
    kind=core
  else
    read -r ay ax <<< "$(awk -v ny=$ny_ -v nx=$nx_ -v g=$sg -v cy=$((ny * H / 2)) -v cx=$((nx * H / 2)) 'BEGIN { l = sqrt(ny * ny + nx * nx); printf "%.1f %.1f", cy + g * 2000 * ny / l, cx + g * 2000 * nx / l }')"
    fj $d/f $z $y $x $nz $ny $nx $ay $ax; kind=local
  fi
  rm -rf $d/pf
  C $U ct-fields $R $CK/1 $d/f --ct-um $(awk -v u=$um 'BEGIN { print 2 * u }') --threads $TH --cache $S/ctcache-$HOST > $d/ct.log 2>&1 || { fail ct-fields; continue; }
  tf=$(date +%s); FIT $d/f --out $d/fit "${FITA[@]}" > $d/fit.log 2>&1 || { fail fit; continue; }; echo "fit wall $(( $(date +%s) - tf ))s" >> $d/fit.log
  rm -f $d/f/*.raw
  C $U wraps $d/fit $d/tifxyz --step 1 --out-um $um --fields $d/f --threads $TH > $d/wraps.log 2>&1 || { fail wraps; continue; }
  C $U surf-stats $d/tifxyz --fields $d/f --model $d/fit --per-wrap "${CT1[@]}" --threads $TH > $d/stats.log 2>&1
  pick $b $kind
  cl=$(g1 'largest [0-9.]+' $d/stats.log); nw=$(ls $d/tifxyz | wc -l); opall=$(grep -oE 'on papyrus \(max CT[^)]*\) [0-9.]+' $d/stats.log | awk '{print $NF}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $b $pct $rn $kind$sd ${cl:-na} ${opall:-na} $nw $(sort -t$'\t' -k1,1 -gr $d/best.tsv | head -1 | cut -f2,3 | tr '\t' :) $(g1 'fit wall [0-9]+' $d/fit.log) $(( $(date +%s) - t0 )) |
    tee $d/FITDONE >> $S/fits-$HOST.tsv
  rm -rf $d/p
done
rm -rf $S/ctcache-$HOST
# all listed boxes fitted (FITDONE) or failed (FITDONE.failed, not retried)? Else another fit.sh still works on them (exit 4)
left=0; for b in $(ls_boxes "$@"); do [ -e $BX/$b/FITDONE ] || [ -e $BX/$b/FITDONE.failed ] || left=$((left + 1)); done
[ $left = 0 ] || { echo "fit.sh: $s: $left boxes still being fitted elsewhere"; exit 4; }
[ $# -gt 0 ] && exit 0

# ---- 4 STITCH=1: connected sheets across the fitted boxes that agree with their neighbours
if [ "${STITCH:-0}" = 1 ]; then
  O=$S/stitch-$N; rm -rf $O; mkdir -p $O/b
  for d in $BX/*/fit; do b=$(basename $(dirname $d)); mkdir -p $O/b/$b; ln -sfn $d $O/b/$b/fit; done
  if [ $(ls $O/b | wc -l) -ge 2 ]; then
    C $U stitch align $O/gauge.json --fields $F --min-agree 0.5 --threads $TH $O/b/* > $O/align.log 2>&1 &&
    C $U stitch quality $O/gauge.json $O/gauge_q.json --fields $F --threads $TH > $O/quality.log 2>&1 &&
    C $U stitch sheets $O/gauge_q.json $O/sheets --fields $F --threads 4 --min-cm2 2 --out-um $um > $O/sheets.log 2>&1 &&
      for w in $(ls $O/sheets); do a=$(jq -r '.area_cm2 // 0' $O/sheets/$w/meta.json); mkdir -p $S/$SURF/stitched; rm -rf $S/$SURF/stitched/$w; cp -r $O/sheets/$w $S/$SURF/stitched/
        printf 'stitched/%s\tstitched\t%s\tna\n' $w $a >> $S/$SURF/list.tsv; done
  fi
fi

# ---- 5 GTCHECK=1 (desk): the published segments of this scan vs the fitted wraps
if [ "${GTCHECK:-0}" = 1 ] && [ $HOST = desk ]; then
  B=https://vesuvius-challenge-open-data.s3.amazonaws.com; GW=$A/gt/$s/w; id=${scan%%-*}
  if [ ! -d $GW ]; then i=0; mkdir -p $GW
    for p in $(curl -s "$B/?list-type=2&prefix=$s/segments/&delimiter=/" | grep -oE "<Prefix>$s/segments/[^<]+" | sed 's/<Prefix>//' | grep -vE '/raw/$|segments/$'); do
      k=$(curl -s "$B/?list-type=2&prefix=${p}mesh/" | grep -oE "<Key>[^<]+on-$id[^<]+/meta.json" | sed 's/<Key>//' | head -1); [ -n "$k" ] || continue
      o=$GW/$(printf w%03d $i); mkdir -p $o; for f in x.tif y.tif z.tif; do curl -sf "$B/${k%meta.json}$f" -o $o/$f; done
      curl -sf "$B/$k" | jq -c --argjson u $um '. + {voxel_size_um: $u}' > $o/meta.json; echo "$(basename $o) $p" >> $GW/../names.txt; i=$((i + 1))
    done; fi
  if [ -n "$(ls $GW 2> /dev/null)" ]; then
    read -r NZ NY NX <<< "$($U info $R $CK 2> /dev/null | awk '$1 == "0" {print $5, $7, $9}')"
    rm -rf $S/gtras; C $U raster $S/gtras --shape $NZ,$NY,$NX --um $um --binary 1 --levels 1 --threads $TH $BX/*/tifxyz/w* > $S/gtcheck.log 2>&1 &&
    C $U wrapdist $GW $S/gtras $(ls $S/gtras | grep -v json) --max 16 --threads $TH --per-wrap >> $S/gtcheck.log 2>&1
    grep -A100 'wrapdist:' $S/gtcheck.log > $S/gtcheck.txt; rm -rf $S/gtras
  fi
fi
if [ "${PRIO_ONLY:-0}" = 1 ]; then mark $s prio-$HOST-$N "$(cat $S/priority-$N.txt | wc -l) priority boxes"; else mark $s fit-$HOST-$N-${MAXBOX:-2} "$(cat $S/$SURF/list.tsv 2> /dev/null | wc -l) surfaces"; fi
