#!/usr/bin/env bash
# head_slab_pass's target cache (--hs-tcache 1, UFSM_HS_TCACHE: the slabs' derived targets and masks kept from the statistics pass for the gradient
# pass) on the wind fixture, --det, latent grid with the rv pair, the guided head and replays, two head slabs a window:
#  (1) UFSM_HS_TCACHE_CHECK=1: every gradient pass rebuilds each slab's planes and compares them with the cached ones byte for byte
#      (the trainer aborts on a difference); the check must have run on every step, over both slabs;
#  (2) the logged losses of the run with the cache equal those of a run without, line for line;
#  (3) the same with --phase-trunc (the mask the statistics pass truncated is the one the gradient pass reads) when the build has it.
# Usage: test_tcache.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 3 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --sched wsd --cooldown 0 --opt muon
   --muon-lr 0.002 --lr 0.001 --replay 3 --packed-weight 8 --steps 6)
fail() { echo "FAIL: $*"; exit 1; }
run() { local tag=$1; shift; env "$@" "$U" train fx/sources.json --out $tag "${C[@]}" ${EXTRA:-} > $tag.log 2>&1 || fail "$tag ($D/$tag.log)"; }
losses() { grep '^step' $1.log | sed 's/ [0-9.]* samp\/s.*//'; }
for v in "" "--fstem 8"; do
  EXTRA=$v; t=$(echo "x$v" | tr -d ' -')
  run off$t UFSM_HS_TCACHE=0
  run on$t UFSM_HS_TCACHE=1 UFSM_HS_TCACHE_CHECK=1
  grep -q "derived targets kept from the statistics pass" on$t.log || fail "the cache was not taken ($D/on$t.log)"
  n=$(grep -c "^target cache check: .* equal" on$t.log || true); b=$(grep "^target cache check: .* equal" on$t.log | tail -1 | grep -o '[0-9]* bytes' | cut -d' ' -f1)
  [ "$n" -ge 6 ] && [ "${b:-0}" -ge $((6 * 64 * 64 * 64 * 10)) ] || fail "the check ran on $n gradient passes over ${b:-0} bytes ($D/on$t.log)"
  [ "$(losses off$t)" = "$(losses on$t)" ] || fail "losses differ with the cache ($D/off$t.log, $D/on$t.log)"
  [ "$(grep -c '^step' on$t.log)" = 6 ] || fail "not 6 steps ($D/on$t.log)"
  echo "  cache on = rebuilt ($n gradient passes, $b bytes), losses equal to the run without${v:+ ($v)}"
  if [ -z "$v" ]; then   # the flag itself (no UFSM_HS_TCACHE): taken, counted by the planner, same losses; without flag or env: not taken
    EXTRA="--hs-tcache 1"; run flag UFSM_HS_TCACHE_CHECK=1; EXTRA=""; run none UFSM_HS_TCACHE_CHECK=1
    grep -q "derived targets kept from the statistics pass" flag.log || fail "--hs-tcache 1: the cache was not taken ($D/flag.log)"
    grep -q "^memory: --hs-tcache keeps" flag.log || fail "--hs-tcache 1: no planner line ($D/flag.log)"
    [ "$(losses off$t)" = "$(losses flag)" ] || fail "losses differ with --hs-tcache 1 ($D/flag.log)"
    ! grep -q "derived targets kept\|^memory: --hs-tcache\|^target cache check" none.log || fail "the cache was taken without the flag ($D/none.log)"
    [ "$(losses off$t)" = "$(losses none)" ] || fail "losses differ without the flag ($D/none.log)"
    echo "  --hs-tcache 1: taken and planned for; default: off"
  fi
done
echo "test_tcache: ok"
