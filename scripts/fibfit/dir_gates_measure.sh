#!/usr/bin/env bash
# dir_gates_measure.sh (fibfit-20261114, post-sweep, CPU only, ufsm-dev memcap slice): the label-free spiral-hand gates on fields whose hand
# is known (scripts/unroll/spiral_dir.tsv: 1667 -1 and P4 +1 published; 1299 unknown, reported only). Per scroll:
#   1. count-field --pool-only 1 --dir-loops 1 without --dir: pass 1 into a pool cache + the gates on 32 middle rows with every plane
#      (fit prior: ring vote, phase loops, w4 planes; the "count-field: dir" line is the combined decision);
#   2. two model solves from that cache, --dir +1 and --dir -1: "count-field: model coherence C" each (the DIRCOH=1 gate of unroll.sh ref());
# then one table row: the w4 vote, the loop vote, the gate decision, C+ / C- and the coherence decision (margin DIRCOH_MARGIN, 0.05), vs the tsv.
# Usage: dir_gates_measure.sh [OUT (default /vesuvius/ufsm/fibfit/dirgates)] [SCROLL ...]   env: UFSM (binary), TH (threads, 16), CFMEM (GB, 16)
set -u
H=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-/vesuvius/ufsm/fibfit/dirgates}; shift 2> /dev/null
SCROLLS=("$@"); [ ${#SCROLLS[@]} -gt 0 ] || SCROLLS=(PHerc1667 PHercParis4 PHerc1299)
U=${UFSM:-$H/build/ufsm}; TH=${TH:-16}; MEM=${CFMEM:-16}; G=${DIRCOH_MARGIN:-0.05}
case "$OUT" in /vesuvius/*|/tmp/*) ;; *) echo "OUT must be under /vesuvius or /tmp: $OUT" >&2; exit 2;; esac
mkdir -p "$OUT"; [ -x "$U" ] || { echo "no binary $U" >&2; exit 2; }
MC=(); [ -x ~/bin/memcap ] && MC=(~/bin/memcap -m $((MEM + 6))G)
# the unroll's ref options (unroll.sh refpool), without REFX
RP=(--R 0 --pool 8 --phase-w 0.5 --wide 100,140 --wide-phase 15 --model-only 1 --model-stride 4 --iters 1200 --threads $TH --mem-gb $MEM)
TAB=$OUT/summary.tsv; printf 'scroll\ttsv\tw4\tloops\tgate\tCplus\tCminus\tcoherence\n' > "$TAB"
for S in "${SCROLLS[@]}"; do
  F=/vesuvius/ufsm/unroll/$S-u1/fields/lat; D=$OUT/$S
  [ -d "$F" ] || { echo "$S: no fields at $F" >&2; continue; }
  case "$D" in "$OUT"/?*) rm -rf -- "$D";; esac; mkdir -p "$D"
  tsv=$(awk -F'\t' -v s=$S '$1 == s {print $2}' "$H/scripts/unroll/spiral_dir.tsv" | head -1)
  "${MC[@]}" "$U" count-field "$F" "$D/cf.unused" "${RP[@]}" --dir-loops 1 --pool-only 1 --pool-cache "$D/pool.bin" > "$D/gates.log" 2>&1
  w4=$(grep -oE 'w4 planes around the axis: .*i\.e\. dir [^ ]+' "$D/gates.log" | tail -1 | sed -E 's/.*: ([0-9]+) of ([0-9]+) loops wind \+, ([0-9]+) wind -, i\.e\. dir (.*)/\4 (\1+\/\3- of \2)/')
  lo=$(grep -oE 'walked around the axis: [0-9]+ of [0-9]+ loops wind \+, [0-9]+ wind -' "$D/gates.log" | tail -1 | sed -E 's/.*: ([0-9]+) of ([0-9]+) loops wind \+, ([0-9]+) wind -/\1+\/\3- of \2/')
  gate=$(grep -oE 'count-field: dir [+-]1 \([^)]*\)' "$D/gates.log" | tail -1 | sed 's/count-field: dir //')
  c=()
  for d in 1 -1; do
    "${MC[@]}" "$U" count-field "$F" "$D/cf.unused" "${RP[@]}" --dir $d --pool-cache "$D/pool.bin" --model "$D/ref$d" > "$D/ref$d.log" 2>&1
    c+=("$(grep -oE 'count-field: model coherence [0-9.]+' "$D/ref$d.log" | tail -1 | awk '{print $4}')")
    rm -rf -- "$D/ref$d"   # only the line is wanted
  done
  coh=$(awk -v p="${c[0]:-}" -v m="${c[1]:-}" -v g=$G 'BEGIN{ if (p == "" || m == "") print "n/a"; else print p - m >= g ? "+1" : m - p >= g ? "-1" : "refuse" }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$S" "${tsv:-?}" "${w4:-none}" "${lo:-none}" "${gate:-?}" "${c[0]:-?}" "${c[1]:-?}" "$coh" >> "$TAB"
  rm -f -- "$D/pool.bin"
done
column -t -s $'\t' "$TAB"
# agreement with the published hands (rows whose tsv dir is exact +-1)
awk -F'\t' 'NR > 1 && ($2 == "1" || $2 == "-1") { t = $2 + 0; g = $5; sub(/ .*/, "", g); w = $3; sub(/ .*/, "", w);
  printf "%s: tsv %+d | w4 %s | gate %s | coherence %s\n", $1, t, (w + 0 == t ? "agrees" : w == "none" || w == "undecided" ? "undecided" : "WRONG"),
         (g + 0 == t ? "agrees" : "WRONG"), ($8 + 0 == t ? "agrees" : $8 == "refuse" || $8 == "n/a" ? $8 : "WRONG") }' "$TAB"
