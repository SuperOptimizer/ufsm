#!/usr/bin/env bash
# Audit attn 11-14 #2: does the bottleneck attention behave the same at a larger predict window (729 tokens at 1152) as at the trained one (216 at 768)?
# Predicts ONE box with window 768 (reference), 1152, and 1152 with --attn-lscale 216, then compares the outputs inside the box with `ufsm phase-diff`.
# GPU needed: run in a GPU window. usage: attn_window_compare.sh CKPT ROOT CT_KEY UM AXIS Z,Y,X,NZ,NY,NX OUTDIR [extra predict args...]
set -euo pipefail
[ $# -ge 7 ] || { sed -n 2,5p "$0"; exit 2; }
ckpt=$1 root=$2 key=$3 um=$4 axis=$5 box=$6 out=$7; shift 7
U="$(dirname "$0")/../../build/ufsm"
mkdir -p "$out"
run() { # name window [extra]
  local name=$1 win=$2; shift 2
  [ -e "$out/$name/.done" ] || { "$U" predict "$ckpt" "$root" "$key" "$out/$name" --um "$um" --axis "$axis" --box "$box" --window "$win" "$@" && touch "$out/$name/.done"; }
}
run w768 768 "$@"
run w1152 1152 "$@"
run w1152ls 1152 --attn-lscale 216 "$@"
for v in w1152 w1152ls; do
  echo "== $v vs w768 (smaller is closer to the trained window's result)"
  "$U" phase-diff "$out/w768" "$(ls "$out/w768" | head -1)" "$out/$v" "$(ls "$out/$v" | head -1)" --box "$box" || echo "phase-diff failed: compare $out/w768 and $out/$v by hand"
done
