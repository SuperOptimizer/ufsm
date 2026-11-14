#!/bin/bash
# axis_for.sh SCROLL: the "--axis FILE" flag for predict (the scroll's umbilicus, level-0 voxels of its scan), or exit 1 with a message on stderr.
# predict refuses to run without an axis since 2026-11-14 (the radial input channels were silently zero); eval scripts call this per scroll.
# Lookup: AXIS_<SCROLL> env, then the table below, then scripts/corpus/list.tsv's axis column (a file, not "auto"), then AXDIR/<SCROLL>.json
# (AXDIR default /vesuvius/ufsm/gt/axis). AXIS_NONE=1 prints "--no-axis 1" instead of failing (an explicit choice, logged by the caller).
S=${1:?usage: axis_for.sh SCROLL}; H=$(cd "$(dirname "$0")" && pwd)
v=AXIS_$S; f=${!v:-}
[ -n "$f" ] || case $S in
  PHercParis4|p4) f=/vesuvius/ufsm/teacher/axis-0.json;;
  PHerc1667|p1667) f=/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json;;
  PHerc0139|p139) f=/vesuvius/ufsm/gt/axis/PHerc0139-umb-scan.json;;
esac
[ -n "$f" ] || f=$(awk -F'\t' -v s=$S 'NR > 1 && $1 == s && $6 != "auto" { print $6 }' $H/../corpus/list.tsv 2> /dev/null)
[ -n "$f" ] || f=${AXDIR:-/vesuvius/ufsm/gt/axis}/$S.json
if [ -s "$f" ]; then echo "--axis $f"; exit 0; fi
[ "${AXIS_NONE:-0}" = 1 ] && { echo "axis_for: no axis for $S ($f): --no-axis 1 (AXIS_NONE=1)" >&2; echo "--no-axis 1"; exit 0; }
echo "axis_for: no axis file for $S (tried $f; set AXIS_$S=FILE or AXIS_NONE=1)" >&2; exit 1
