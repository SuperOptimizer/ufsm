#!/usr/bin/env bash
# fit.sh S [BOX...]: spiral fits of scroll S (laptop) from its published prod96 fit fields, resumable (A/<S>/state, per-box FITDONE).
#  1 fields: the fit-field zarr copied to A/<S>/fields (whole 512^3 shard files, curl --parallel; absent shards = air)
#  2 boxes: N^3 field cells (N 768: 1536 native voxels, 1.3-1.4 cm; 1024 needs > 6 GB in ct-fields / fit), stride N - 128, material share from CT level 5; the boxes with
#    >= 10% material in the order: middle z first, then outward (A/<S>/boxes.tsv). MAXBOX (default all) caps the boxes of this pass.
#  3 per box: CT planes (ufsm ct-fields, CT level 1), GLOBAL fit (the scroll's axis from fields.json, box-local) with the best measured
#    settings (--w-ct 1 --ct-phi 0 --ct-flood 0 --uw-repair 8 --w-normal 3, z-slabs) -> wraps (tifxyz at the native um) -> gate
#    (ufsm stitch quality: phase coherence >= COH (0.45), anti-aligned <= ANTI (0.15)). A failing box is refitted LOCALLY (the axis
#    2000 native voxels from the box centre along its mean sheet normal, the better side by a probe fit of the centre, --core 0) and the
#    better of the two by coherence is kept. Per-box stats: surf-stats (clean sheet, spacing errors, phase residual).
#  4 stitch: ufsm stitch align / quality / solve / sheets over the kept box fits -> A/<S>/sheets (connected sheets across boxes, >= 1 cm2)
#  5 A/<S>/surf: per box tifxyz + stitched sheets + list.tsv (surface, kind, area cm2) for the desk (render.sh)
# Every heavy command runs under capped.sh (MEM 6G) on the laptop; one box at a time; the GPU part of each fit under ~/ufsm-data/gpu.lock.
D=$(cd "$(dirname "$0")" && pwd); . $D/common.sh
s=$1; shift; read -r scan um fz <<< "$(scan_of $s)"; [ -n "${fz:-}" ] || { echo "fit.sh: $s not in $A/scrolls.tsv"; exit 2; }
S=$A/$s; F=$S/fields/$fz; mkdir -p $S/state $S/boxes $S/surf; CK=$s/volumes/$scan.zarr
if [ $HOST = desk ]; then N=${N:-1024}; TH=${TH:-12}; else N=${N:-768}; TH=${TH:-8}; fi; ST=$((N - 128)); COH=${COH:-0.45}; ANTI=${ANTI:-0.15}
FITA=(--w-ct 1 --ct-phi 0 --ct-flood 0 --uw-repair 8 --w-normal 3 --threads $TH --quiet)   # (whole box; --slab must be a multiple of the knot spacing)
[ -n "${SLAB:-}" ] && FITA+=(--slab $SLAB --overlap 128)
# FIT ARGS..: ufsm fit; laptop: under the shared GPU lock (ink-model porting agent) and capped.sh MEM; desk: GPU FITGPU (1) or FITCPU=1
FIT() { if [ $HOST = laptop ]; then flock $HOME/ufsm-data/gpu.lock env MEM=${MEM:-6G} UFSM_FIT_THREADS=$TH $HOME/ufsm-data/capped.sh $U fit "$@"
        elif [ "${FITCPU:-0}" = 1 ]; then UFSM_FIT_THREADS=$TH nice -n 5 $U fit "$@" --cpu
        else local g=${FITGPU:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)}   # the emptier GPU
             UFSM_FIT_THREADS=$TH nice -n 5 $U fit "$@" --gpu $g; fi; }
g1() { grep -oE "$1" "$2" 2> /dev/null | tail -1 | grep -oE '[0-9.]+$'; }

# ---- 1 fields
if ! done_ $s fields; then
  mkdir -p $F; P=48
  for f in fields.json README.json zarr.json; do curl -sf --retry 5 $R/$s/representations/predictions/fit-fields/$fz/$f -o $F/$f || [ $f = zarr.json ]; done
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

# ---- 2 box list: corner (cells), material % (CT level 5: 32 native voxels; box = 2N/32 = N/16 voxels), middle z first
if [ ! -s $S/boxes.tsv ]; then
  t=$S/boxes.tsv.tmp; : > $t; n5=$((N * H / 32))
  for ((z = 0; z + 256 <= SZ; z += ST)); do for ((y = 0; y + 256 <= SY; y += ST)); do for ((x = 0; x + 256 <= SX; x += ST)); do
    pct=$($U read $R $CK/5 $((z * H / 32)) $((y * H / 32)) $((x * H / 32)) $n5 $n5 $n5 /dev/null 2>&1 | grep -oE 'nonzero [0-9.]+' | awk '{print $2}')
    [ -n "$pct" ] || { echo "fit.sh: CT level 5 read failed at $z $y $x"; exit 1; }
    awk -v p=$pct 'BEGIN { exit !(p >= 10) }' && printf 'b%05d_%04d_%04d\t%d\t%d\t%d\t%s\n' $z $y $x $z $y $x $pct >> $t
  done; done; done
  awk -v m=$((SZ / 2)) -v n=$N -F'\t' '{ d = $2 + n / 2 - m; print (d < 0 ? -d : d) "\t" (100 - $5) "\t" $0 }' $t | sort -n -k1,1 -k2,2 | cut -f3- > $S/boxes.tsv; rm -f $t
  echo "$s: $(wc -l < $S/boxes.tsv) boxes with >= 10% material (grid $N cells, stride $ST)"
fi

# fj DIR Z Y X NZ NY NX [AY AX]: a zarr-backed fields dir for the box (corner Z Y X cells, size NZ NY NX): the planes linked to F, the
# scroll's axis made box-local (or the constant box-local axis AY AX, native voxels)
fj() {
  local d=$1 z=$2 y=$3 x=$4 nz=$5 ny=$6 nx=$7; mkdir -p $d
  for p in pc ps t0 t1 t2 t3 t4 t5 sp w; do ln -sfn $F/$p $d/$p; done
  if [ $# -ge 9 ]; then
    jq -c --argjson z $z --argjson y $y --argjson x $x --argjson nz $nz --argjson ny $ny --argjson nx $nx --argjson h $H --argjson ay $8 --argjson ax $9 \
      '{format: "zarr", origin: [$z * $h, $y * $h, $x * $h], shape: [$nz, $ny, $nx], h: $h, um: .um, zarr_offset: [$z, $y, $x],
        axis: [range(0; $nz * $h + 1) | [$ay, $ax]]}' $F/fields.json > $d/fields.json
  else
    jq -c --argjson z $z --argjson y $y --argjson x $x --argjson nz $nz --argjson ny $ny --argjson nx $nx --argjson h $H \
      '{format: "zarr", origin: [$z * $h, $y * $h, $x * $h], shape: [$nz, $ny, $nx], h: $h, um: .um, zarr_offset: [$z, $y, $x],
        axis: [.axis[$z * $h : ($z + $nz) * $h + 1][] | [.[0] - $y * $h, .[1] - $x * $h]]}' $F/fields.json > $d/fields.json
  fi
}
# quality DIR: the box's phase coherence / anti-aligned share vs the network (ufsm stitch quality) -> DIR/quality.log; prints "coh anti"
quality() {
  printf '{"um": %s, "boxes": [{"name": "%s", "dir": "%s"}], "edges": []}\n' $um $(basename $1) $1 > $1/box.json
  C $U stitch quality $1/box.json $1/box_q.json --fields $F --threads $TH > $1/quality.log 2>&1
  echo "$(grep -oE 'phase coherence [0-9.]+' $1/quality.log | tail -1 | awk '{print $3}') $(g1 'coherent \(>= 0.5\) [0-9.]+, anti-aligned [0-9.]+' $1/quality.log)"
}
# fitone DIR FIELDDIR [extra fit args]: fit + wraps + quality + surf-stats in DIR
fitone() {
  local o=$1 f=$2; shift 2; rm -rf $o/fit $o/tifxyz
  local t=$(date +%s); FIT $f --out $o/fit "${FITA[@]}" "$@" > $o/fit.log 2>&1 || return 1; echo "fit wall $(( $(date +%s) - t ))s" >> $o/fit.log
  C $U wraps $o/fit $o/tifxyz --step 1 --out-um $um --fields $f --threads $TH > $o/wraps.log 2>&1 || return 1
  C $U surf-stats $o/tifxyz --fields $f --model $o/fit --threads $TH > $o/stats.log 2>&1 || true
}

# ---- 3 box fits
ls_boxes() { if [ $# -gt 0 ]; then printf '%s\n' "$@"; else head -n ${MAXBOX:-100000} $S/boxes.tsv | cut -f1; fi; }
# several fit.sh of one scroll share the boxes (claims: boxes/<b>.claim = host pid; a dead claimant's claim is taken over)
claim() { local c=$S/boxes/$1.claim; mkdir -p $S/boxes
  if ! mkdir $c 2> /dev/null; then read -r h p < $c/who 2> /dev/null; [ "${h:-}" = $HOST ] && ! kill -0 ${p:-0} 2> /dev/null || return 1; fi
  echo "$HOST $$" > $c/who; }
for b in $(ls_boxes "$@"); do
  d=$S/boxes/$b; [ -e $d/FITDONE ] || [ -e $d/FITDONE.failed ] && continue; claim $b || continue
  read -r z y x pct <<< "$(awk -v b=$b '$1 == b {print $2, $3, $4, $5}' $S/boxes.tsv)"
  nz=$((SZ - z < N ? SZ - z : N)) ny=$((SY - y < N ? SY - y : N)) nx=$((SX - x < N ? SX - x : N)); t0=$(date +%s)
  rm -rf $d; mkdir -p $d/g; fj $d/f $z $y $x $nz $ny $nx
  C $U ct-fields $R $CK/1 $d/f --ct-um $(awk -v u=$um 'BEGIN { print 2 * u }') --threads $TH --cache $S/ctcache > $d/ct.log 2>&1 || { echo "FAILED ct-fields" > $d/FITDONE.failed; continue; }
  # global
  if fitone $d/g $d/f; then read -r gc ga <<< "$(quality $d/g)"; else gc=0 ga=1; fi
  kind=global; kc=${gc:-0}; ka=${ga:-1}
  if ! awk -v c=${gc:-0} -v a=${ga:-1} -v C=$COH -v A=$ANTI 'BEGIN { exit !(c >= C && a <= A) }'; then
    # local: probe the centre sub-box on both sides, mean normal + side, then the box with the axis there
    pz=$((z + nz / 4)) py=$((y + ny / 2 - 192)) px=$((x + nx / 2 - 192)); [ $py -lt $y ] && py=$y; [ $px -lt $x ] && px=$x
    mkdir -p $d/p; fj $d/pf $pz $py $px $((nz / 2)) $((ny < 384 ? ny : 384)) $((nx < 384 ? nx : 384)) 0 0
    for sd in p m; do Dn=2000; [ $sd = m ] && Dn=-2000; FIT $d/pf --out $d/p/$sd --core 0 --axis-normal $Dn --quiet > $d/p/$sd.log 2>&1; done
    lp=$(g1 'loss [0-9.]+ in' $d/p/p.log | head -1); lm=$(g1 'loss [0-9.]+ in' $d/p/m.log | head -1)
    sd=p; sg=-1; awk -v a="${lp:-9e9}" -v b="${lm:-9e9}" 'BEGIN { exit !(b < a) }' && { sd=m; sg=1; }
    read -r ny_ nx_ <<< "$(grep -m1 'mean normal' $d/p/$sd.log | sed -E 's/.*mean normal z [-0-9.]+ y ([-0-9.]+) x ([-0-9.]+);.*/\1 \2/')"
    if [ -n "${nx_:-}" ] && [ -n "${lp:-}${lm:-}" ]; then
      read -r ay ax <<< "$(awk -v ny=$ny_ -v nx=$nx_ -v g=$sg -v cy=$((ny * H / 2)) -v cx=$((nx * H / 2)) 'BEGIN { l = sqrt(ny * ny + nx * nx); printf "%.1f %.1f", cy + g * 2000 * ny / l, cx + g * 2000 * nx / l }')"
      mkdir -p $d/l; fj $d/lf $z $y $x $nz $ny $nx $ay $ax; cp $d/f/ct.raw $d/f/cs.raw $d/f/ct.json $d/lf/ 2> /dev/null
      if fitone $d/l $d/lf --core 0; then
        read -r lc la <<< "$(quality $d/l)"
        awk -v l=${lc:-0} -v g=${gc:-0} 'BEGIN { exit !(l > g) }' && { kind=local; kc=$lc; ka=${la:-1}; }
      fi
    fi
    rm -rf $d/p $d/pf
  fi
  k=$d/g; [ $kind = local ] && k=$d/l
  [ -d $k/tifxyz ] || { echo "FAILED fits (global coh ${gc:-na})" > $d/FITDONE.failed; continue; }
  ln -sfn $(basename $k) $d/kept; rm -f $d/f/*.raw $d/lf/*.raw
  cl=$(g1 'largest [0-9.]+' $k/stats.log); nw=$(ls $k/tifxyz | wc -l)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $b $pct $kind ${kc:-na} ${ka:-na} ${gc:-na} ${cl:-na} $nw $(( $(date +%s) - t0 )) | tee $d/FITDONE >> $S/fits.tsv
done
# all listed boxes fitted (FITDONE) or failed (FITDONE.failed, not retried)? Else another fit.sh still works on them: done here (exit 4)
left=0; for b in $(ls_boxes "$@"); do [ -e $S/boxes/$b/FITDONE ] || [ -e $S/boxes/$b/FITDONE.failed ] || left=$((left + 1)); done
[ $left = 0 ] || { echo "fit.sh: $s: $left boxes still being fitted elsewhere"; exit 4; }
[ $# -gt 0 ] && exit 0
claim finish || { echo "fit.sh: $s: another fit.sh finishes it"; exit 4; }

# ---- 4 stitch the kept box fits into connected sheets
if ! done_ $s stitch; then
  O=$S/stitch; rm -rf $O; mkdir -p $O; bx=$(for d in $S/boxes/*/kept; do [ -d $d/fit ] && dirname $d; done)
  if [ $(echo "$bx" | grep -c .) -ge 2 ]; then
    # stitch reads BOXDIR/fit: a dir per box with fit -> kept/fit
    for d in $bx; do mkdir -p $O/b/$(basename $d); ln -sfn $d/kept/fit $O/b/$(basename $d)/fit; done
    C $U stitch align $O/gauge.json --fields $F --threads $TH $O/b/* > $O/align.log 2>&1 &&
    C $U stitch quality $O/gauge.json $O/gauge_q.json --fields $F --threads $TH > $O/quality.log 2>&1 &&
    C $U stitch solve $O/gauge_q.json $O/gauge_tree.json --min-agree 0.1 > $O/solve.log 2>&1 &&
    C $U stitch sheets $O/gauge_q.json $S/sheets.tmp --fields $F --threads 4 --min-cm2 1 --out-um $um > $O/sheets.log 2>&1 && { rm -rf $S/sheets; mv $S/sheets.tmp $S/sheets; } ||
      echo "fit.sh: $s stitch failed (see $O)"
  fi
  mark $s stitch "$(ls $S/sheets 2> /dev/null | wc -l) sheets"
fi

# ---- 5 the surfaces for the desk -> A/<S>/surf: the TOPBOX (5) largest wraps (>= 0.5 cm2) of each box, every stitched sheet (>= 1 cm2);
# list.tsv: surface, kind (box-global / box-local / stitched), area cm2, on papyrus (surf-stats --ct: max CT level 1 within +-15 um
# along the normal >= 80); boxes.tsv: the fitted boxes (name z y x n, field cells) for ink.sh's first phase
CTA=(--ct $R $CK/1 --ct-um $(awk -v u=$um 'BEGIN { print 2 * u }') --ct-level 80 --ct-range 15 --cache $S/ctcache --threads $TH)
rm -rf $S/surf; mkdir -p $S/surf; : > $S/surf/list.tsv; : > $S/surf/boxes.tsv
for d in $S/boxes/*/kept; do b=$(basename $(dirname $d)); [ -d $d/tifxyz ] || continue; fd=$(dirname $d)/f; [ $(readlink $d) = l ] && fd=$(dirname $d)/lf
  awk -v b=$b -v n=$N '$1 == b {print $1, $2, $3, $4, n}' $S/boxes.tsv >> $S/surf/boxes.tsv
  [ -s $d/onpap.log ] || C $U surf-stats $d/tifxyz --fields $fd --per-wrap "${CTA[@]}" > $d/onpap.log 2>&1
  for w in $(for m in $d/tifxyz/*/meta.json; do echo "$(jq -r '.area_cm2 // 0' $m) $(basename $(dirname $m))"; done | sort -gr | awk '$1 >= 0.5' | head -n ${TOPBOX:-5} | awk '{print $2}'); do
    a=$(jq -r '.area_cm2' $d/tifxyz/$w/meta.json); op=$(awk -v w=$w '$1 == w {print $NF}' $d/onpap.log)
    mkdir -p $S/surf/$b; cp -r $d/tifxyz/$w $S/surf/$b/
    printf '%s/%s\tbox-%s\t%s\t%s\n' $b $w $(readlink $d | sed 's/^g$/global/; s/^l$/local/') $a ${op:-na} >> $S/surf/list.tsv
  done
done
if [ -d $S/sheets ]; then for w in $(ls $S/sheets); do [ -e $S/sheets/$w/meta.json ] || continue; a=$(jq -r '.area_cm2 // 0' $S/sheets/$w/meta.json)
  mkdir -p $S/surf/one; ln -sfn $S/sheets/$w $S/surf/one/w_$w   # (surf-stats reads the w* entries)
  op=$(C $U surf-stats $S/surf/one --fields $F "${CTA[@]}" 2> /dev/null | grep -oE 'on papyrus \(max CT[^)]*\) [0-9.]+' | awk '{print $NF}')
  rm -rf $S/surf/one; mkdir -p $S/surf/stitched; cp -r $S/sheets/$w $S/surf/stitched/
  printf 'stitched/%s\tstitched\t%s\t%s\n' $w $a ${op:-na} >> $S/surf/list.tsv; done; fi
sort -t$'\t' -k3,3 -gr $S/surf/list.tsv -o $S/surf/list.tsv; rm -rf $S/ctcache
# ---- 6 check against the published segments of this scan (AWS open data <S>/segments/*/mesh/*-on-<scan id>-*.tifxyz, desk only:
# the raster needs ~30 GB): every kept box wrap rasterised (binary band, native voxels), ufsm wrapdist of the published segments to it
# -> gtcheck.txt (share of segment vertices within 9.6 / 19.2 um of a fitted sheet; vertices outside the fitted boxes count as far)
if [ $HOST = desk ] && ! done_ $s gtcheck; then
  B=https://vesuvius-challenge-open-data.s3.amazonaws.com; GW=$A/gt/$s/w; id=${scan%%-*}
  if [ ! -d $GW ]; then i=0; mkdir -p $GW
    for p in $(curl -s "$B/?list-type=2&prefix=$s/segments/&delimiter=/" | grep -oE "<Prefix>$s/segments/[^<]+" | sed 's/<Prefix>//' | grep -vE '/raw/$|segments/$'); do
      k=$(curl -s "$B/?list-type=2&prefix=${p}mesh/" | grep -oE "<Key>[^<]+on-$id[^<]+/meta.json" | sed 's/<Key>//' | head -1); [ -n "$k" ] || continue
      o=$GW/$(printf w%03d $i); mkdir -p $o; for f in x.tif y.tif z.tif; do curl -sf "$B/${k%meta.json}$f" -o $o/$f; done
      curl -sf "$B/$k" | jq -c --argjson u $um '. + {voxel_size_um: $u}' > $o/meta.json; echo "$(basename $o) $p" >> $GW/../names.txt; i=$((i + 1))
    done; fi
  if [ -n "$(ls $GW 2> /dev/null)" ]; then
    read -r NZ NY NX <<< "$($U info $R $CK 2> /dev/null | awk '$1 == "0" {print $5, $7, $9}')"
    rm -rf $S/gtras; C $U raster $S/gtras --shape $NZ,$NY,$NX --um $um --binary 1 --levels 1 --threads $TH $S/boxes/*/kept/tifxyz/w* > $S/gtcheck.log 2>&1 &&
    C $U wrapdist $GW $S/gtras $(ls $S/gtras | grep -v json) --max 16 --threads $TH --per-wrap >> $S/gtcheck.log 2>&1 &&
      grep -A100 'wrapdist:' $S/gtcheck.log > $S/gtcheck.txt; rm -rf $S/gtras
  fi
  mark $s gtcheck "$(grep -m1 ' all ' $S/gtcheck.txt 2> /dev/null)"
fi
mark $s fit "$(wc -l < $S/surf/list.tsv) surfaces, largest $(head -1 $S/surf/list.tsv | cut -f1,3)"
if [ $HOST = desk ]; then mark $s surf "fitted on the desk"; rm -rf $F $S/state/fields.done; fi   # (the fields are refetched for a later pass)
