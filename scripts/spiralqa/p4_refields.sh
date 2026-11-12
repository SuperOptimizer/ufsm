#!/bin/bash
# p4_refields.sh RUN WORK NEW1 [NEW2 ...]: local re-predicted fit fields (predict --fit-zarr boxes, any level, the run's physical cell
# grid) pasted into RUN/fields/lat, the slabs that see them refitted, re-joined, wraps + snap, scored (GT locate, wrap-check). New
# outputs only under WORK; RUN is read. Stages are resumable by WORK/state/<stage>.done. CPU only (fit --cpu); env:
#   FEATHER (32 cells) TH (12 threads) FITREF (joined: the slabs' --ref-model; "ref-cf" = the run's count reference) FITX (extra fit args)
#   SLABS ("k0 k1": override the slab range) STOP_AFTER (a stage name)
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: p4_refields.sh RUN WORK NEW1 [NEW2 ...]" >&2; exit 2; }
O=$(realpath "$1"); W=$2; shift 2; NEW=("$@"); mkdir -p "$W/state" "$W/logs"; W=$(realpath "$W")
U=${U:-$HOME/ufsm-spiralqa/build/ufsm}; S=$(dirname "$(realpath "$0")"); MK=$O/bin/mkcrop.sh
TH=${TH:-12}; FEATHER=${FEATHER:-32}; FITREF=${FITREF:-joined}; I=64; OV=16
CT=(--ct "/vesuvius/ufsm/v2/ct" "PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2" --ct-um 9.6)
CODES=(--codes /vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr 9.6)
F0=$O/fields/lat; FP=$W/fields_patched
done_() { [ -e "$W/state/$1.done" ]; }; mark() { touch "$W/state/$1.done"; [ "${STOP_AFTER:-}" = "$1" ] && { echo "stop after $1"; exit 0; }; return 0; }
log() { echo "$(date '+%F %T') $*" | tee -a "$W/progress.log"; }

# the union of the new boxes in F0's cells, out to F0's 128-cell shards (python: the cell offsets from origin x um)
read -r BZ BY BX NZ NY NX < <(python3 - "$F0/fields.json" "${NEW[@]}" <<'EOF'
import json, sys, math
f0 = json.load(open(sys.argv[1])); c = f0["h"] * f0["um"]; lo = [1e18] * 3; hi = [-1e18] * 3
for d in sys.argv[2:]:
    j = json.load(open(d + "/fields.json"))
    assert abs(j["h"] * j["um"] - c) < 1e-6 * c, (d, j["h"], j["um"])
    for a in range(3):
        o = (j["origin"][a] * j["um"] - f0["origin"][a] * f0["um"]) / c
        lo[a] = min(lo[a], o); hi[a] = max(hi[a], o + j["shape"][a])
b0 = [max(0, int(math.floor(lo[a] / 128)) * 128) for a in range(3)]
b1 = [min(f0["shape"][a], int(math.ceil(hi[a] / 128)) * 128) for a in range(3)]
print(*b0, *[b1[a] - b0[a] for a in range(3)])
EOF
)
log "union box (cells of fields/lat): z $BZ+$NZ y $BY+$NY x $BX+$NX"

if ! done_ paste; then
  rm -rf "$W/boxed" "$FP"
  /usr/bin/time -v "$U" field-blend "$W/boxed" "$F0" "${NEW[@]}" --paste "$FEATHER" --box "$BZ,$BY,$BX,$NZ,$NY,$NX" --threads "$TH" > "$W/logs/paste.log" 2>&1
  bash "$S/field_paste_link.sh" "$F0" "$W/boxed" "$FP" >> "$W/logs/paste.log"
  "$MK" "$F0" "$W/c_old" $BZ $BY $BX $NZ $NY $NX > /dev/null; "$MK" "$FP" "$W/c_new" $BZ $BY $BX $NZ $NY $NX > /dev/null
  "$U" fit-compare "$W/c_old" "$W/c_new" > "$W/logs/paste_compare.log" 2>&1 || true; rm -rf "$W/c_old" "$W/c_new"
  mark paste
fi

# slabs whose crop [64k - 16, 64k + 80) meets the box in z
ZE=$(jq -r '.shape[0]' "$F0/fields.json"); NY0=$(jq -r '.shape[1]' "$F0/fields.json"); NX0=$(jq -r '.shape[2]' "$F0/fields.json")
if [ -n "${SLABS:-}" ]; then read -r K0 K1 <<< "$SLABS"; else K0=$(( (BZ - OV - I) / I + 1 )); [ $K0 -lt 0 ] && K0=0; K1=$(( (BZ + NZ + OV - 1) / I )); fi
NS=$(( (ZE + I - 1) / I )); [ $K1 -ge $NS ] && K1=$((NS - 1))
log "refit slabs s$(printf %03d $K0)..s$(printf %03d $K1) (ref model $FITREF)"
DIR=$(jq -r .dir "$O/ref-cf/model.json")
for k in $(seq $K0 $K1); do
  s=$(printf 's%03d' $k); d=$W/slabs/$s; [ -e "$d/done" ] && continue; mkdir -p "$d"
  c0=$((k * I)); c1=$((c0 + I)); [ $c1 -gt $ZE ] && c1=$ZE; a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ZE ] && b=$ZE
  rm -rf "$d/f" "$d/fit"; "$MK" "$FP" "$d/f" $a 0 0 $((b - a)) $NY0 $NX0 > /dev/null; echo "$c0 $c1 $a $b" > "$d/cells"; t0=$(date +%s)
  env UFSM_FIT_THREADS=$TH /usr/bin/time -v "$U" fit "$d/f" --cpu --threads "$TH" --out "$d/fit" --dir "$DIR" --ref-model "$O/$FITREF" --ref-snap 1 --uw-repair 0 \
      --steps 300,300,300,300,200,200 --batch 1000000 ${FITX:-} > "$d/fit.log" 2>&1 && rc=0 || rc=$?
  rm -rf "$d/f"; log "$s rc $rc $(( $(date +%s) - t0 )) s"
  if [ $rc = 0 ] && grep -qE '^fit: .*loss [0-9]' "$d/fit.log"; then touch "$d/done"; elif grep -q "no data for the prior" "$d/fit.log"; then touch "$d/empty" "$d/done"; else exit 1; fi
done
mark slabs

if ! done_ join; then   # the run's slabs, the refitted ones in place
  args=()
  for d in $(ls -d "$O"/slabs/s??? | sort); do
    s=$(basename "$d"); [ -e "$W/slabs/$s/done" ] && d=$W/slabs/$s
    [ -e "$d/empty" ] && continue; read -r c0 c1 a b < "$d/cells"; args+=("$d/fit" $((2 * c0)) $((2 * c1)))
  done
  rm -rf "$W/joined"; /usr/bin/time -v "$U" fit-join "$W/joined" "$FP" --blend 16 --ref "$O/ref-cf" "${args[@]}" > "$W/logs/join.log" 2>&1
  mark join
fi
if ! done_ wraps; then
  rm -rf "$W/wraps"; /usr/bin/time -v "$U" wraps "$W/joined" "$W/wraps" --step 4 --out-um 9.6 --sfc 0.25 --fields "$FP" --threads "$TH" --zr 384,18176 --min-cm2 0.5 > "$W/logs/wraps.log" 2>&1
  mark wraps
fi
if ! done_ snap; then
  rm -rf "$W/wraps_snap"; /usr/bin/time -v "$U" wrap-snap "$W/wraps" "$W/wraps_snap" --target mix --axis "$O/axis.json" --axis-um 2.400 "${CT[@]}" --fields "$FP" --threads "$TH" ${SNAPX:-} > "$W/logs/snap.log" 2>&1
  mark snap
fi
if ! done_ score; then   # GT locate on the snapped and raw wraps (the before: RUN/qa/wr), then the switch events
  A=(--axis "$O/axis.json" --axis-um 2.4 "${CT[@]}" "${CODES[@]}" --threads "$TH" --cell-rows 16 --cell-deg 5)
  "$U" wrap-repair "$W/wraps_snap" - "${A[@]}" --shift 0.5 --tsv "$W/loc_snap.tsv" --cells "$W/loc_snap_cells.tsv" > "$W/logs/loc_snap.log" 2>&1
  "$U" wrap-repair "$W/wraps" - "${A[@]}" --shift 0 --tsv "$W/loc_raw.tsv" --cells "$W/loc_raw_cells.tsv" > "$W/logs/loc_raw.log" 2>&1
  python3 "$S/../../tools/switch_locate.py" "$W/loc_snap_cells.tsv" "$W/loc_raw_cells.tsv" --out "$W/sw_snap" > "$W/logs/switch.log" 2>&1
  mark score
fi
log "done: $(grep -h 'GT turns' "$W/logs/loc_snap.log" | sed 's/.*GT turns/GT turns/')"
