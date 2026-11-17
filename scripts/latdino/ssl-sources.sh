#!/usr/bin/env bash
# scripts/latdino/ssl-sources.sh IN.json OUT.json [SHARE]: IN's sources plus one unlabelled ("ct_only": 1) source per CT volume in
# $CTROOT (default /vesuvius/ufsm/v2/ct) for train --ssl (DESIGN 2026-11-12 "Self-supervision for the latent net").
# - EXCL (default PHerc0175A): scrolls left out entirely (the strict unseen control).
# - EXCLVOL (below): single volumes left out.
# - Every box of $HOLD (default /vesuvius/ufsm/inkgt/holdouts-ink.json: the wbench boxes + the ink eval crops) on that CT rejects
#   training windows ("holdout_reject": all names): an unlabelled window has no labels to mask, so held-out CT is never drawn.
# - Levels as the labelled sources: scans finer than 3 um at levels 2..3 (9.6 / 19.2 um), the 7.9-9.4 um scans at 0..1.
# - The axis only where a labelled IN source on the same CT names one (its frame is known); otherwise none (the radial channels are
#   then constant, as for the u2ink sources without one).
# - SHARE is only echoed: pass it as train --ssl-share (the trainer rescales the ct_only weights; each gets weight 1 here).
# - MIRROR=1 (pods, 2026-11-17): the volumes of the list VOLS (lines SCROLL/SCAN.zarr; default: CTROOT's PH*/*.zarr, i.e. made on the desk and
#   copied as a tiny file) read from the volcomp mirror ($MIR/<scroll>/volumes/<scan>); a volume counts when its zarr.json answers.
#   For the pod A/B: HOLD=holdouts-20261114.json (every per-scroll holdout + wbench box on that CT is rejected).
set -euo pipefail
IN=${1:?IN.json}; OUT=${2:?OUT.json}; SHARE=${3:-0.25}
CTROOT=${CTROOT:-/vesuvius/ufsm/v2/ct}; HOLD=${HOLD:-/vesuvius/ufsm/inkgt/holdouts-ink.json}; EXCL=${EXCL:-PHerc0175A}
# EXCLVOL: volumes left out because held-out boxes of their scroll are given on another scan of it and not mapped into this one
# (0841: the 10 ink eval crops are on the 2.403 um scan; 0500P2's 9.362 um scan is a labelled training source already)
EXCLVOL=${EXCLVOL:-PHerc0841/20250821151531-9.366um-1.2m-113keV-masked.zarr PHerc0500P2/20250820143440-9.362um-1.2m-113keV-masked.zarr}
tmp=$(mktemp); trap 'rm -f $tmp' EXIT
n=0
MIRROR=${MIRROR:-0}; MIR=${MIR:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}
if [ -n "${VOLS:-}" ]; then mapfile -t vols < "$VOLS"; else vols=(); for v in "$CTROOT"/PH*/*.zarr; do vols+=("$(basename "$(dirname "$v")")/$(basename "$v")"); done; fi
for rel in "${vols[@]}"; do
  [ -n "$rel" ] || continue
  vol=$CTROOT/$rel; scroll=${rel%%/*}; key=$rel
  case " $EXCL " in *" $scroll "*) continue;; esac
  case " $EXCLVOL " in *" $key "*) echo "skip $key: EXCLVOL" >&2; continue;; esac
  if [ "$MIRROR" = 1 ]; then
    key=$scroll/volumes/$(basename "$rel")
    curl -sfI "$MIR/$key/zarr.json" > /dev/null || { echo "skip $key: no zarr.json on the mirror" >&2; continue; }
  else
    [ -e "$vol/zarr.json" ] || { echo "skip $key: no zarr.json (an incomplete local copy)" >&2; continue; }
  fi
  um=$(basename "$vol" | sed -n 's/^[0-9]*-\([0-9.]*\)um-.*/\1/p'); [ -n "$um" ] || { echo "skip $key: no voxel size in the name" >&2; continue; }
  if awk -v u="$um" 'BEGIN{exit !(u < 3)}'; then lo=2; hi=3; else lo=0; hi=1; fi
  base=$(basename "$vol")
  rej=$(jq -c --arg b "$base" '[.boxes[] | select((.ct | split("/") | last) == $b) | .name]' "$HOLD")
  ax=$(jq -r --arg b "$base" '[.sources[] | select((.ct | split("/") | last) == $b and (.ct_only // 0) == 0 and .axis != null) | .axis][0] // ""' "$IN")
  root=$CTROOT; [ "$MIRROR" = 1 ] && root=$MIR
  jq -n --arg name "$scroll-$um-u" --arg root "$root" --arg ct "$key" --argjson um "$um" --argjson lo $lo --argjson hi $hi \
        --arg hold "$HOLD" --argjson rej "$rej" --arg ax "$ax" \
        '{name: $name, scope: "global", root: $root, ct: $ct, um: $um, iso: 64, min_level: $lo, max_level: $hi, weight: 1, ct_only: 1} + (if ($rej | length) > 0 then {holdout_file: $hold, holdout_reject: $rej} else {} end) + (if $ax != "" then {axis: $ax} else {} end)' >> $tmp
  # (no holdout_file when no box of HOLD is on this CT: holdout_strict makes a holdout file that matches nothing fatal; 11-17)
  n=$((n + 1))
done
jq --slurpfile u $tmp '.sources += $u | .note = ((.note // "") + " + ssl-sources.sh: \($u | length) ct_only volumes (excluded '"$EXCL"')")' "$IN" > "$OUT"
echo "$OUT: $(jq '.sources | length' "$OUT") sources, $n unlabelled; train with --ssl 16 --ssl-share $SHARE" >&2
