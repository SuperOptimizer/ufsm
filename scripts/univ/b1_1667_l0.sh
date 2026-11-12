#!/usr/bin/env bash
# scripts/univ/b1_1667_l0.sh (pod): PHerc1667 winding codes at level 0 (2.399 um) for the onfly phase at 2.4 um (rmax 160), the stored
# fallback while codes on the fly from the meshes are built (2026-11-11). From the q-assigned surfaces qA (1 mesh) / qB (19 meshes) of
# /vesuvius/ufsm/gt/PHerc1667-wind-20261016 (rsynced from the desk, 0.6 + 0.75 GB), `raster --level 0` with the flags of the 4.798 codes.
# Check: the same raster at --level 1 of set A must be byte-identical to the stored codesA/4.798 (CHK=1).
# Env: K (/workspace/univ/gt/1667), U, T (16). Keeps >= MINFREE (25) GB free: stops before a set if less.
set -u; K=${K:-/workspace/univ/gt/1667}; U=${U:-/workspace/univ/ufsm/build/ufsm}; T=${T:-16}; SH=37076,15229,15229; UM=2.399
step() { echo "== $(date +%T) $* (free $(df --output=avail -BG /workspace | tail -1))"; }
if [ "${CHK:-1}" = 1 ] && [ ! -e $K/chkA1.ok ]; then step check A level 1
  nice -n 5 $U raster $K/chkA1 --shape $SH --um $UM --level 1 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $K/qA/*.tifxyz > $K/chkA1.log 2>&1
  n=0; d=0; for f in $(cd $K/codesA/4.798/c && find . -type f); do n=$((n + 1)); cmp -s $K/codesA/4.798/c/$f $K/chkA1/4.798/c/$f || d=$((d + 1)); done
  echo "check A level 1: $n stored shards, $d differ, $(find $K/chkA1/4.798/c -type f | wc -l) rebuilt"; [ $d = 0 ] && touch $K/chkA1.ok; rm -rf $K/chkA1; fi
for s in A B; do O=$K/codes${s}24; [ -e $O/done ] && continue
  [ $(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9) -ge ${MINFREE:-25} ] || { step "set $s: not enough disk"; exit 1; }
  step raster $s level 0; t0=$(date +%s)
  nice -n 5 $U raster $O --shape $SH --um $UM --level 0 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $K/q$s/*.tifxyz > $K/raster${s}24.log 2>&1 && touch $O/done
  echo "set $s: $(( $(date +%s) - t0 )) s, $(du -sh $O | cut -f1)"; done
step done
