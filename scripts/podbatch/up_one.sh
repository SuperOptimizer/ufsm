#!/usr/bin/env bash
# up_one.sh SCROLL SCAN UM LEVEL WINDOW [replace]: the upload lane of fit-field batch v5 (run under flock upload.lock). README.json (written
# if missing, else its codec / produced lines updated), lossless repack of the 10 arrays from 128^3 to 512^3 shards (ufsm repack; checks:
# chunk count = input files, decoded spot regions of pc / sp byte-identical), upload (lftp mirror --parallel=24), remote size check of
# pc/zarr.json, state/<s>.uploaded. "replace": a remote copy of exactly this path left partial today by this batch is first renamed to
# <name>.partial and deleted after the upload.
set -u; s=$1 scan=$2 um=$3 lv=$4 w=$5 rep=${6:-}; T=${TAG:-up}; cd /workspace; U=/workspace/ufsm/build/ufsm; ST=out/STATUS.md
R=https://dl.ash2txt.org/community-uploads/forrest/volcomp; name=${scan%%-*}-fitfields-20261016-ufsm-prod96-h2.zarr
CODEC="zarr v3, sharding_indexed 512^3 shards of 128^3 chunks (losslessly repacked from predict's 128^3 shards by ufsm repack), inner codec volcomp lossy q 8 (all 10 planes)"
GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed -e "s/^NVIDIA //" -e "s/^GeForce //"); RTH=${RTH:-8}
PROD="$(date -u +%F) on an ${GPUN:-RTX PRO 6000} (ufsm 7ae7a47 + predict --fit-q / --no-out / slab ring + repack, branch podbatch-fitq 7dafc9e)"
log() { ( flock 8; echo "$(date -u +%Y-%m-%dT%H:%MZ) $T $*" >> $ST ) 8> status.lock; }
o=out/$s/$name; rp=out/$s/rp/$name; mkdir -p out/$s state
if [ -d $o ]; then
  [ -f $o/fields.json ] && [ $(ls -d $o/*/ | wc -l) = 10 ] || { log "$s: $o incomplete"; exit 1; }
  if [ ! -f $o/README.json ]; then
    ivu=$(awk -v u=$um -v l=$lv 'BEGIN {printf "%.6g", u * 2 ^ l}'); fvu=$(awk -v u=$um -v l=$lv 'BEGIN {printf "%.6g", u * 2 ^ l * 2}')
    cat > $o/README.json <<EOR
{"content": "winding (spiral) fitter input fields (ufsm src/fit.h), the input of ufsm fit -> ufsm wraps (tifxyz surfaces)",
 "scroll": "$s", "scan": "$scan", "source": "$R/$s/volumes/$scan.zarr level $lv", "input_um": $um, "input_level": $lv, "input_voxel_um": $ivu,
 "model": "ufsm prod96 winding / unwrap net (9.6 um, 10.7M params, task unwrap), checkpoint step 94872, EMA weights",
 "fit_h": 2, "field_voxel_um": $fvu, "field_origin_zyx_input_voxels": [0, 0, 0], "window": $w, "halo": 64,
 "codec": "$CODEC",
 "planes": {"pc": "phase cos: 128 + 127 * mean(2 sigma - 1) of the phase-cos channel over material voxels of the cell (fill 128)",
            "ps": "phase sin: as pc for the phase-sin channel (fill 128)",
            "t0": "n n^T xx: 255 * clamp(mean decoded diagonal)", "t1": "n n^T yy", "t2": "n n^T zz",
            "t3": "n n^T off-diagonal: 255 * clamp(0.5 + mean(sigma - 0.5))", "t4": "n n^T off-diagonal", "t5": "n n^T off-diagonal",
            "sp": "sheet spacing code: round(8 / (h * rho)), rho = winding density per input voxel (0 = unknown)",
            "w": "material fraction of the cell (CT > 0) * 255"},
 "produced": "$PROD"}
EOR
  else
    sed -i -e "s|\"codec\": \"[^\"]*\"|\"codec\": \"$CODEC\"|" -e "s|\"produced\": \"[^\"]*\"|\"produced\": \"$PROD\"|" $o/README.json
    grep -q '"window"' $o/README.json || sed -i "s|\"field_origin_zyx_input_voxels\": \[0, 0, 0\],|& \"window\": $w, \"halo\": 64,|" $o/README.json
  fi
  jq -e . $o/README.json > /dev/null || { log "$s: README.json is not valid JSON"; exit 1; }
  rm -rf $rp; mkdir -p $rp; cp $o/README.json $o/fields.json $rp/; ok=1; nin=0; nch=0
  for a in pc ps t0 t1 t2 t3 t4 t5 sp w; do
    m=$($U repack $o/$a $rp/$a 512 --threads $RTH 2>> out/$s/repack.log) || ok=0
    nin=$(( nin + $(find $o/$a/c -type f 2>/dev/null | wc -l) )); nch=$(( nch + $(echo "$m" | awk '{for (i = 1; i < NF; i++) if ($(i + 1) == "chunks,") print $i}') ))
  done
  [ $nin = $nch ] || ok=0
  read -r Fz Fy Fx <<< "$(jq -r '.shape | @tsv' $o/fields.json)"
  for a in pc sp; do for r in "0 0 0" "$(( Fz / 2 )) $(( Fy / 3 )) $(( Fx / 3 ))" "$(( Fz / 3 )) $(( Fy / 2 - 32 )) $(( Fx / 2 - 32 ))"; do set -- $r
    rm -f out/$s/va.raw out/$s/vb.raw
    $U read $o/$a . $1 $2 $3 128 256 256 out/$s/va.raw > /dev/null 2>&1; $U read $rp/$a . $1 $2 $3 128 256 256 out/$s/vb.raw > /dev/null 2>&1
    [ -s out/$s/va.raw ] && cmp -s out/$s/va.raw out/$s/vb.raw || ok=0; done; done; rm -f out/$s/va.raw out/$s/vb.raw
  [ $ok = 1 ] || { log "$s REPACK FAILED or mismatch ($nin input files, $nch chunks); $o kept"; exit 1; }
  log "$s repacked losslessly: $nin -> $(find $rp -type f | wc -l) files ($nch chunks, spot regions identical)"; rm -rf $o
fi
[ -d $rp ] || { log "$s: nothing to upload"; exit 1; }
rd=volcomp/$s/representations/predictions/fit-fields
if [ "$rep" = replace ]; then /root/up.sh "mv $rd/$name $rd/$name.partial" > out/$s/mv.log 2>&1; log "$s: today's partial remote upload renamed to $name.partial (rc $?)"; fi
nf=$(find $rp -type f | wc -l); bytes=$(du -sb $rp | cut -f1); t1=$(date +%s)
/root/up.sh "mkdir -p -f $rd; mirror -R --parallel=24 --no-perms $rp $rd/$name; cls -l $rd/$name/pc/zarr.json $rd/$name/README.json" > out/$s/upload.log 2>&1; urc=$?
dt=$(( $(date +%s) - t1 + 1 )); lsz=$(stat -c %s $rp/pc/zarr.json); rsz=$(grep -F "pc/zarr.json" out/$s/upload.log | grep -v "^sftp" | awk '{print $5}' | head -1)
if [ $urc = 0 ] && [ "$lsz" = "$rsz" ]; then
  log "$s uploaded $nf files, $bytes bytes in ${dt}s ($(awk -v b=$bytes -v t=$dt 'BEGIN {printf "%.1f", b / t / 1e6}') MB/s) -> $rd/$name (pc/zarr.json $rsz bytes matches)"
  touch state/$s.uploaded; rm -rf $rp
  if [ "$rep" = replace ]; then   # (detached: many small files, not holding the upload lane)
    ( t2=$(date +%s); /root/up.sh "rm -r -f $rd/$name.partial" > out/$s/rm.log 2>&1; log "$s: $name.partial deleted (rc $?, $(( $(date +%s) - t2 ))s)" ) < /dev/null > /dev/null 2>&1 & fi
else log "$s UPLOAD CHECK FAILED (rc $urc, local $lsz remote ${rsz:-?}); $rp kept"; exit 1; fi
