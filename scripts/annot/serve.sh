#!/usr/bin/env bash
# scripts/annot/serve.sh SCROLL [PORT]: the manual winding annotation page (ufsm annotate-serve) for PHerc1299, PHerc1218 or PHercParis4
# on this desk, with our snapped wraps, the spiral label phase (or the human GT phase for Paris 4), fibres and a work queue of flagged
# regions (fibre judge tiles + wrap-check flags, ufsm annot-regions). Open http://127.0.0.1:PORT/ in the desk's browser (VNC).
# Annotations: /vesuvius/ufsm/annot/<SCROLL>.json (+ .log, append-only). Export: ufsm annot-export (see DESIGN.md 2026-11-11).
set -u
S=${1:?scroll: PHerc1299 | PHerc1218 | PHercParis4}; PORT=${2:-8765}
U=${U:-$(dirname "$(readlink -f "$0")")/../../build/ufsm}; A=/vesuvius/ufsm/annot; mkdir -p $A
CT=/vesuvius/ufsm/v2/ct; W=""; LAB=(); FIB=""; JUDGE=(); WC=""
case $S in
  PHerc1299) KEY=PHerc1299/20260309130042-2.399um-0.2m-78keV-masked.zarr; UM=2.399; L=2
    W=/vesuvius/ufsm/latcal/wraps_snap/PHerc1299/wraps_snap; LAB=(--labels /vesuvius/ufsm/latcal/labels_snap/PHerc1299/phase.zarr 9.596)
    FIB=/vesuvius/ufsm/fibers/work/tr_1299_L3.fib; JUDGE=(/vesuvius/ufsm/fibers/work/fjfull_1299/*.tsv); WC=/mnt/scratch/pods/PHerc1299/run/wrapcheck.tsv;;
  PHerc1218) KEY=PHerc1218/20250521120456-8.640um-1.2m-116keV-masked.zarr; UM=8.64; L=0
    W=/vesuvius/ufsm/v2/wraps_snap/PHerc1218/wraps_snap; LAB=(--labels /vesuvius/ufsm/latcal/labels/PHerc1218/phase.zarr 8.64)
    WC=/mnt/scratch/pods/PHerc1218/unroll/wrapcheck.tsv;;
  PHercParis4) KEY=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr; UM=2.4; L=2
    W=/vesuvius/ufsm/v2/wraps_snap/PHercParis4/wraps_snap; LAB=(--labels /vesuvius/ufsm/gt/paris4-wind-20261019 phaseA/9.6)
    FIB=/vesuvius/ufsm/fibers/work/tr_p4_L4.fib; JUDGE=(/vesuvius/ufsm/fibers/work/fj_p4L4_wraps.tsv);;
  *) echo "unknown scroll $S"; exit 2;;
esac
[ -d "$CT/$KEY" ] || CT=https://dl.ash2txt.org/community-uploads/forrest/volcomp   # (not local: the public volcomp mirror)
R=$A/$S.regions.tsv; JA=(); for j in "${JUDGE[@]}"; do [ -s "$j" ] && JA+=(--judge "$j"); done
WA=(); [ -n "$WC" ] && [ -s "$WC" ] && [ -n "$W" ] && WA=(--wrapcheck "$WC" --wraps "$W" --um0 "$UM")
$U annot-regions $R "${JA[@]}" "${WA[@]}" --level $L > /dev/null || R=""
ARGS=(--um $UM --level $L --port $PORT --store $A/$S.json --scroll $S --cache /vesuvius/ufsm/cache)
[ -n "$W" ] && [ -d "$W" ] && ARGS+=(--wraps "$W"); [ ${#LAB[@]} -gt 0 ] && ARGS+=("${LAB[@]}"); [ -n "$FIB" ] && [ -s "$FIB" ] && ARGS+=(--fibers "$FIB")
[ -n "$R" ] && ARGS+=(--regions "$R")
exec nice -n 5 $U annotate-serve $CT $KEY "${ARGS[@]}"
