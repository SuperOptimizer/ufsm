#!/usr/bin/env bash
# gt.sh S (CAL 11-01, POD-L): winding GT of one scroll from its published segments (AWS open data), ported from desk rv/sense
# (lib.sh / named.sh / each.sh / p4.sh, 2026-10-19 wind-sense). Labels: codes + pair-rule phase at 2x the scan voxel (8-9 um scans: level 1)
# or 9.6 um (2.4 um scans). Never invents adjacency: named = authors' wNNN numbering (set A) + the rest wind-assigned onto A;
# each = every segment its own set, phases fused (gauge-free). Sense from all segments' geometry (wind-seed --k 0 dry); refused if undetermined.
# NATIVE=1 (11-03, no coarse labels): codes + phase at the scan's own voxel (level 0, --gs 2, radius 768 um) for 8-9 um scans.
# Env C (/root/cal: the tree with axis/, seg/, gt/), X (axis dir, default $C/axis).
set -uo pipefail; S=$1; C=${C:-/root/cal}; U=${U:-/root/pod/ufsm/build/ufsm}; T=${T:-12}; X=${X:-$C/axis}
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
step() { echo "== $(date +%T) $*"; }
conf() { case $1 in   # VOLID UM SH AX MODE HIRES
  PHerc0172) echo 20241024131839 7.91 20820,6700,9100 $X/PHerc0172.json named 0;;
  PHerc1447) echo 20250521151220 8.64 24297,8343,8343 $X/PHerc1447-umb-scan.json each 0;;
  PHerc0500P2) echo 20250820143440 9.362 7057,4196,4196 $X/PHerc0500P2-20250820143440.json named 0;;
  PHercMANBp) echo 20251216152116 2.399 17148,12577,12577 $X/PHercMANBp.json named 1;;
  PHerc0009B) echo 20250521125136 8.64 9598,7837,7837 $X/PHerc0009B-20250521125136.json each 0;;
  PHerc0814) echo 20250804134230 9.362 19390,5784,5784 $X/PHerc0814-20250804134230.json each 0;;
  PHercParis4) echo 20260411134726 2.4 75784,32693,32693 $X/paris4-axis-0.json p4 1;;
  *) return 1;; esac; }
read VOLID UM SH AX MODE HI < <(conf $S) || { echo "unknown $S"; exit 2; }
SEG=$C/seg/$S; G=$C/gt/$S-wind; W=$G/sets; mkdir -p $SEG $W
splitk() { tac | awk -v want=$1 '{ k = "x" NR; if (match($0, /[-_\/]w(rap)?[0-9]+/)) { k = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", k); k += 0 }
  s = seen[k]++ ? "B" : "A"; if (s == want) print }' | sort; }
key() { ls $1 | grep -v json | sort -g | head -1; }

# 1. segments (meta.json + x/y/z.tif of mesh/*-on-VOLID-*.tifxyz; MANBp: mesh/intermediate/tifxyz_original)
if [ ! -e $SEG/.done ]; then
  step download $S
  segs=$(curl -s --max-time 120 "$B/?list-type=2&prefix=$S/segments/&delimiter=/&max-keys=1000" | grep -o '<Prefix>[^<]*</Prefix>' | sed 's/<[^>]*>//g' | grep -v "^$S/segments/\$")
  [ $S = PHercParis4 ] && segs=$(echo "$segs" | grep -E '/20260623[0-9]*-w[0-9]' | grep -v jordi)
  for sp in $segs; do
    k=$(curl -s --max-time 120 "$B/?list-type=2&prefix=${sp}mesh/&max-keys=1000" | grep -o '<Key>[^<]*</Key>' | sed 's/<[^>]*>//g')
    if [ "$HI" = 1 ] && [ $S = PHercMANBp ]; then echo "$k" | grep '/intermediate/tifxyz_original/' | grep -E '/(meta\.json|[xyz]\.tif)$'
    else echo "$k" | grep -v intermediate | grep -- "-on-$VOLID-" | grep -E '/(meta\.json|[xyz]\.tif)$'; fi
  done > $SEG/keys.txt
  echo "$(wc -l < $SEG/keys.txt) files"
  xargs -P 16 -I{} sh -c 'f="$1"; d="$2/$(echo "$f" | sed "s|^[^/]*/segments/||")"; mkdir -p "$(dirname "$d")"; [ -s "$d" ] || curl -sf --retry 5 --max-time 3600 -o "$d" "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/$f"' _ {} $SEG < $SEG/keys.txt
  # MANBp: tifxyz_original -> a <seg>/mesh/<ts>-on-VOLID-2.399um.tifxyz dir name so the name filters below work
  if [ $S = PHercMANBp ]; then for d in $SEG/*/mesh/intermediate/tifxyz_original; do s=$(basename $(dirname $(dirname $(dirname $d)))); ts=${s%%-*}
    mkdir -p $(dirname $(dirname $d)) && ln -sfn $d $(dirname $(dirname $d))/$ts-on-$VOLID-2.399um.tifxyz; done; fi
  touch $SEG/.done; du -sh $SEG
fi
ALL=$(ls -d $SEG/*/mesh/*-on-$VOLID-*.tifxyz)
echo "$S: $(echo "$ALL" | grep -c .) surfaces, mode $MODE"

# 2. sense from all segments (dry)
step sense; [ -s $C/gt/dry-$S/set.json ] || nice -n 5 $U wind-seed $C/gt/dry-$S $ALL --k 0 --vol-um $UM --axis $AX > $C/gt/dry-$S.log 2>&1
jq -c '{sense, sense_determined, score, sense_margin, pitch, surfaces}' $C/gt/dry-$S/set.json; rm -rf $C/gt/dry-$S/*.tifxyz
DET=$(jq -r .sense_determined $C/gt/dry-$S/set.json); SN=$(jq -r .sense $C/gt/dry-$S/set.json)
if [ "$HI" = 1 ]; then LV=2; GS=4; LUM=9.6; R=160; [ $S = PHercMANBp ] && LUM=9.596; else LV=1; GS=2; LUM=$(echo "$UM*2" | bc -l | sed 's/0*$//'); R=$(echo "384/$UM" | bc); fi
if [ -n "${NATIVE:-}" ]; then [ "$HI" = 1 ] && { echo "$S: NATIVE is for 8-9 um scans (2.4 um scans already train on their 9.6 level 2)"; exit 2; }
  LV=0; GS=2; LUM=$UM; R=$(echo "768/$UM" | bc); fi
RS=(); [ $S = PHercParis4 ] && RS=(--codes-grid 1 --box 36864,5120,5120,20480,24576,25600 --resume 1)
case $MODE in
p4|named)
  SO=""; [ "$DET" = 1 ] && SO="--sense $SN"
  if [ $MODE = p4 ]; then A="$ALL"; Bs=""; else
    if [ $S = PHerc0172 ]; then NM=$(echo "$ALL" | grep -- '-w0'); else NM=$(echo "$ALL" | grep -E -- '-w[0-9]|wrap[0-9]'); fi
    A=$(echo "$NM" | splitk A); Bs=$( (echo "$ALL" | grep -vxF "$NM" || true; echo "$NM" | splitk B) | grep . || true); fi
  step seed A "($(echo "$A" | grep -c .)) $SO"; [ -s $W/qA/set.json ] && [ -s $W/qA/seed.jsonl ] || nice -n 5 $U wind-seed $W/qA $A --vol-um $UM --axis $AX $SO > $W/seedA.log 2>&1 || { tail -5 $W/seedA.log; echo "$S: seed A refused"; exit 4; }
  grep "spiral sense" $W/seedA.log | tail -1
  step raster A; [ -f $W/codesA/zarr.json ] || nice -n 5 $U raster $W/codesA --shape $SH --um $UM --level $LV --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $W/qA/*.tifxyz 2>&1 | tail -1
  sets=A
  if [ -n "$Bs" ]; then step assign B "($(echo "$Bs" | grep -c .))"; [ -s $W/qB/assign.jsonl ] || nice -n 5 $U wind-assign $W/codesA $(key $W/codesA) $W/qB $Bs --ref-um $LUM --vol-um $UM --axis $AX --tol $(echo "t = 29 / $LUM; if (t < 2) 2 else t" | bc) --step 4 --threads $T > $W/assignB.log 2>&1
    if ls -d $W/qB/*.tifxyz > /dev/null 2>&1; then step raster B; [ -f $W/codesB/zarr.json ] || nice -n 5 $U raster $W/codesB --shape $SH --um $UM --level $LV --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $W/qB/*.tifxyz 2>&1 | tail -1
      [ -f $W/codesB/zarr.json ] && sets="A B"; fi; fi
  PH=""; CO=""
  for s in $sets; do step phase $s; [ -f $W/phase$s/zarr.json ] && [ ! -e $W/phase$s.partial ] || { touch $W/phase$s.partial; nice -n 5 $U phase-raster $W/codes$s $(key $W/codes$s) $W/phase$s --gs $GS --pair 1 --radius $R --axis $AX --threads $T --shards 1 "${RS[@]}" 2>&1 | tail -2 && rm -f $W/phase$s.partial; }
    PH="$PH $W/phase$s $(key $W/phase$s)"; CO="$CO $W/codes$s $(key $W/codes$s)"; done;;
each)
  [ "$DET" = 1 ] || { echo "$S: the spiral sense is undetermined: refused (no invented adjacency)"; exit 4; }
  PH=""; CO=""; i=0
  for t in $ALL; do i=$((i + 1)); n=$(basename $(dirname $(dirname $t)))
    step seed s$i $n; [ -s $W/qs$i/set.json ] || nice -n 5 $U wind-seed $W/qs$i $t --k 0 --sense $SN --seam start --min-surface-score 0.8 --vol-um $UM --axis $AX > $W/seed-s$i.log 2>&1
    ls -d $W/qs$i/*.tifxyz > /dev/null 2>&1 || { echo "   s$i dropped"; continue; }
    [ -f $W/codess$i/zarr.json ] || nice -n 5 $U raster $W/codess$i --shape $SH --um $UM --level $LV --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $W/qs$i/*.tifxyz 2>&1 | tail -1
    [ -f $W/phases$i/zarr.json ] || nice -n 5 $U phase-raster $W/codess$i $(key $W/codess$i) $W/phases$i --gs $GS --pair 1 --radius $R --axis $AX --threads $T --shards 1 2>&1 | tail -1
    ls $W/phases$i/*/c > /dev/null 2>&1 && PH="$PH $W/phases$i $(key $W/phases$i)"; CO="$CO $W/codess$i $(key $W/codess$i)"; done
  [ -n "$PH" ] || { echo "$S: no segment has a phase"; exit 5; };;
esac
if [ $MODE = p4 ]; then ln -sfn sets/phaseA $G/phase; ln -sfn sets/codesA $G/codes; else
  step phase fuse; [ -f $G/phase/zarr.json ] || nice -n 5 $U phase-fuse $G/phase $LUM $PH --tol 0.1 --threads $T
  step codes merge; [ -f $G/codes/zarr.json ] || nice -n 5 $U codes-merge $G/codes $LUM $CO --threads $T; fi
step check; nice -n 5 $U phase-check $G/phase $(key $G/phase) --axis $AX --scan-um $UM --threads $T || true
rm -rf $W/q*/*.tifxyz; touch $G/DONE; step done; du -sh $G/*
