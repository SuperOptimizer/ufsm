#!/usr/bin/env bash
# test_rhocal.sh (2026-11-18, fix0332): scripts/unroll/unroll.sh's rhocal stage against a fake ufsm (no GPU, no data, ~1 s).
# The fake count-field writes a model whose "k" is the mean of its --rho-r profile; the fake wrap-check reports CT peaks per turn
# 1.2 / k in every radius band (so K 1.2 is right), the fake cf-demod a body drift. Checks: the profile rule (radial / global, the
# K floor and cap, unused bands), convergence, the installed ref is a model DIRECTORY (11-18: `ls -d it*` picked it3.bands.tsv and moved
# a tsv into ref-cf), and both DEFAULTED rules leave the plain ref.
set -u
cd "$(dirname "$0")/.." || exit 1
T=$(mktemp -d "${TMPDIR:-/tmp}/trhocal.XXXXXX") || exit 1
trap '[ -n "${KEEPT:-}" ] || case "$T" in */trhocal.*) rm -rf -- "$T";; esac' EXIT
fails=0; ck() { if eval "$2"; then echo "ok: $1"; else echo "FAIL: $1"; fails=$((fails + 1)); fi; }
cat > $T/ufsm <<'EOF'
#!/usr/bin/env bash
cmd=$1; shift
arg() { local k=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$k" ] && { echo "$2"; return; }; shift; done; }
case $cmd in
  count-field) m=$(arg --model "$@"); p=$(arg --rho-r "$@"); mkdir -p $m
    k=$(echo "$p" | tr ',' '\n' | awk -F: '{s += $2; n++} END {printf "%.4f", n ? s / n : 1}'); echo "{\"dir\": -1, \"k\": $k}" > $m/model.json;;
  wrap-check) m=$1; b=$(arg --bands "$@"); k=$(sed -nE 's/.*"k": ([0-9.]+).*/\1/p' $m/model.json); k=${k:-1}
    { printf 'r0_mm\tr1_mm\tn\tct_per_turn\tempty\tone\ttwo_plus\tgt_n\tgt_per_turn\n'
      for r in 0 1 2 3 4 5; do awk -v r=$r -v k=$k -v n=$([ $r = 5 ] && echo 100 || echo 5000) 'BEGIN {printf "%d\t%d\t%d\t%.4f\t0.1\t0.7\t0.2\t0\tnan\n", r, r + 1, n, 1.2 / k}'; done; } > $b
    echo "wrap-check: verdicts: 50 real, $(awk -v k=$k -v d=${FAKE_DUP:-0} 'BEGIN {print 5 + (k > 1.01 ? d : 0)}') dup, 3 merged (--dup 0.5)"
    echo "  CT papyrus peaks per fitted turn $(awk -v k=$k 'BEGIN {printf "%.3f", 1.2 / k}')";;
  cf-demod) m=$1; k=$(sed -nE 's/.*"k": ([0-9.]+).*/\1/p' $m/model.json)
    echo "cf-demod: z drift body: along-sheet excursion $(awk -v k=${k:-1} -v d=${FAKE_DRIFT:-0} 'BEGIN {printf "%.2f", 0.3 + d * (k - 1)}') turns over bands 2..9";;
esac
EOF
chmod +x $T/ufsm
# the stage's functions from unroll.sh (rcprof, rcscore, rhocal) and the helpers they use
F=$T/fns.sh; { sed -n '/^rcprof() {/,/^}/p; /^rcscore() {/,/^}/p; /^rhocal() {/,/^}/p' scripts/unroll/unroll.sh
  echo 'log() { echo "$*" >> $O/progress.log; }; X() { local l=$1; shift; "$@" >> $l 2>&1; }; refpool() { echo --R 0; }; refcz() { echo 512; }; mset() { :; }'; } > $F
run() {   # run NAME ENV...: a fresh run dir with a plain ref (k 1), then rhocal
  local O=$T/$1; shift; mkdir -p $O/ref-cf $O/logs; echo '{"dir": -1, "k": 1}' > $O/ref-cf/model.json
  env "$@" O=$O REF=$O/ref-cf F=$T/f U=$T/ufsm R=r K=k LV=2 LUM=9.6 TH=1 DRY=0 bash -c "source $F; rhocal" > /dev/null 2>&1; echo $?; }
kof() { sed -nE 's/.*"k": ([0-9.]+).*/\1/p' $1/ref-cf/model.json; }
rc=$(run radial RHOCALMODE=radial)
ck "radial: stage ok" '[ "$rc" = 0 ]'
ck "radial: ref-cf is a model directory" '[ -d $T/radial/ref-cf ] && [ -e $T/radial/ref-cf/model.json ]'
ck "radial: the plain ref kept as rhocal/ref0.model" '[ -e $T/radial/rhocal/ref0.model/model.json ]'
ck "radial: K -> ~1.2" 'awk -v k=$(kof $T/radial) "BEGIN { exit !(k > 1.15 && k < 1.25) }"'
ck "radial: converged before pass 3" 'grep -q "converged after" $T/radial/progress.log && [ ! -d $T/radial/rhocal/it3 ]'
ck "radial: the thin band (100 intervals) holds its neighbour's K" 'grep -q "pass 1: --rho-r 0.50:1.2[0-9]*,1.50:1.2[0-9]*,2.50:1.2[0-9]*,3.50:1.2[0-9]*,4.50:1.2[0-9]*,5.50:1.2" $T/radial/progress.log'
rc=$(run global RHOCALMODE=global)
ck "global: one K, ~1.2, a model directory" '[ "$rc" = 0 ] && [ -e $T/global/ref-cf/model.json ] && grep -q "pass 1: --rho-r 0:1.2" $T/global/progress.log'
rc=$(run cap RHOCALMAX=1.1)
ck "cap: K held at RHOCALMAX 1.1, still installed" '[ "$rc" = 0 ] && awk -v k=$(kof $T/cap) "BEGIN { exit !(k > 1.09 && k < 1.11) }"'
rc=$(run gcap RHOCALMODE=global RHOCALMAX=1.1)
ck "global at the cap: converged after pass 1, K 1.1" '[ "$rc" = 0 ] && [ ! -d $T/gcap/rhocal/it2 ] && awk -v k=$(kof $T/gcap) "BEGIN { exit !(k > 1.09 && k < 1.11) }"'
rc=$(run drift FAKE_DRIFT=1)
ck "drift rule: DEFAULTED, the plain ref kept" 'grep -q "DEFAULTED: the body drift rose" $T/drift/progress.log && [ "$(kof $T/drift)" = 1 ] && [ ! -e $T/drift/rhocal/ref0.model ]'
rc=$(run dup FAKE_DUP=4)
ck "dup rule: DEFAULTED, the plain ref kept" 'grep -q "DEFAULTED: wrap-check dup+merged rose" $T/dup/progress.log && [ "$(kof $T/dup)" = 1 ]'
mkdir -p $T/ok1/ref-cf $T/ok1/logs; echo '{"dir": -1, "k": 1.2}' > $T/ok1/ref-cf/model.json
O=$T/ok1 REF=$T/ok1/ref-cf F=$T/f U=$T/ufsm R=r K=k LV=2 LUM=9.6 TH=1 DRY=0 bash -c "source $F; rhocal" > /dev/null 2>&1
ck "already calibrated: kept, nothing re-solved" 'grep -q "already within" $T/ok1/progress.log && [ ! -d $T/ok1/rhocal/it1 ]'
[ $fails = 0 ] && echo "test_rhocal: OK" || { echo "test_rhocal: $fails FAILED"; exit 1; }
