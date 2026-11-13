# sourced by the training tests: det_cmp A.log B.log WHAT [TOL]: two --det 1 train runs saw the same data and trained alike.
# Exact: every batch plane's hash (UFSM_BATCH_HASH=1) and each step's device input (UFSM_DET_TRACE=1 "x"; its label hashes exist only for
# streamed planes, so a device-kept vs streamed pair could not compare them). Losses only within TOL (default 0.02, relative): --det fixes the windows, not the GPU's float atomics (GN / bias sums,
# split-K weight gradients), so two runs of one binary differ in the last bits of the gradients and later losses can take one of a
# few values (test_ct_gpu_decode: step 4 2.3195 / 2.3271 at random, also CPU vs CPU; 2026-11-13). The runs need both env vars.
det_cmp() {
  local a=$1 b=$2 what=$3 tol=${4:-0.02} ha hb xa xb la lb
  ha=$(grep -E '^batchhash' "$a"); hb=$(grep -E '^batchhash' "$b")
  [ -n "$ha" ] || { echo "FAIL: $what: no batchhash lines (UFSM_BATCH_HASH=1) in $a"; return 1; }
  [ "$ha" = "$hb" ] || { echo "FAIL: $what: the batches differ"; diff <(echo "$ha") <(echo "$hb") | head -4; return 1; }
  xa=$(grep -E '^dettrace' "$a" | sed -E 's/ labels .*//'); xb=$(grep -E '^dettrace' "$b" | sed -E 's/ labels .*//')   # (the input; the label hashes are set only for streamed planes)
  [ -n "$xa" ] || { echo "FAIL: $what: no dettrace lines (UFSM_DET_TRACE=1) in $a"; return 1; }
  [ "$xa" = "$xb" ] || { echo "FAIL: $what: the device inputs differ"; diff <(echo "$xa") <(echo "$xb") | head -4; return 1; }
  la=$(grep -E '^step' "$a" | awk '{print $2, $6}'); lb=$(grep -E '^step' "$b" | awk '{print $2, $6}')
  [ -n "$la" ] && [ "$(echo "$la" | wc -l)" = "$(echo "$lb" | wc -l)" ] || { echo "FAIL: $what: no / unequal step lines"; return 1; }
  paste -d' ' <(echo "$la") <(echo "$lb") | awk -v t="$tol" '{d = $2 - $4; if (d < 0) d = -d; if ($1 != $3 || d > t * ($2 < 0 ? -$2 : $2) + 1e-6) bad = 1} END {exit bad}' \
    || { echo "FAIL: $what: losses differ by more than $tol"; paste <(echo "$la") <(echo "$lb"); return 1; }
  DET_CMP_N=$(echo "$ha" | wc -l)
}
