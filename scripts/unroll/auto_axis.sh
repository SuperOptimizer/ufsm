#!/bin/bash
# auto_axis.sh U ROOT KEY UM OUT: the "auto" axis of a scroll without a hand / trusted umbilicus json (corpus desk.sh, unroll.sh prep).
# Default since 2026-11-27: ufsm umbilicus (sheet-normal voting on the CT, villa vc_gen_umbilicus port, --uncrush 1.5; DESIGN
# "ufsm umbilicus"): 0.7-2.5 mm from the trusted axes where the per-z CT centroid was 3.8-11 mm (1667: centroid 0.87 vs 1.04 median).
# AXIS_METHOD=centroid: the old ufsm axis (per-z CT>0 centroid). An existing non-empty OUT is kept byte for byte (live runs never move;
# AXIS_REDO=1 rebuilds it). The method that made the file is in its "method" key and on stdout. AXIS_THREADS (6) caps the vote's threads.
U=${1:?usage: auto_axis.sh U ROOT KEY UM OUT}; R=${2:?}; K=${3:?}; UM=${4:?}; OUT=${5:?}
if [ -s "$OUT" ] && [ "${AXIS_REDO:-0}" != 1 ]; then echo "auto_axis: kept existing $OUT ($(jq -r '.method // "no method key"' "$OUT" 2> /dev/null))"; exit 0; fi
case "${AXIS_METHOD:-umbilicus}" in
  centroid) echo "auto_axis: centroid (ufsm axis, AXIS_METHOD=centroid) -> $OUT"; exec "$U" axis "$R" "$K" "$OUT";;
  umbilicus) echo "auto_axis: umbilicus (ufsm umbilicus, CT sheet-normal voting, level auto) -> $OUT"
             exec "$U" umbilicus "$OUT" --um0 "$UM" --ct-root "$R" --ct-key "$K" --level auto --threads "${AXIS_THREADS:-6}";;
  *) echo "auto_axis: unknown AXIS_METHOD=$AXIS_METHOD (umbilicus | centroid)" >&2; exit 2;;
esac
