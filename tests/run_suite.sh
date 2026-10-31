#!/usr/bin/env bash
# tests/run_suite.sh: the ufsm test suite runner (make test / test-slow / test-all call it). Pure bash + flock.
#
#   tests/run_suite.sh [options] [REGEX...]     run the tests of the manifest tests/suite.txt (REGEX: only names matching one)
#     -t TIER   fast (default: everything not tagged slow) | slow (only the slow tag) | all
#     -j N      tests in parallel (default: cgroup-aware CPUs / 3, 1..12; env SUITE_J)
#     -T N      CPU threads per test, exported as UFSM_NCPU / UFSM_FIT_THREADS / UFSM_TIFF_THREADS / OMP_NUM_THREADS
#               (default CPUs / j, at least 2; env SUITE_THREADS)
#     -B DIR    build dir holding the test binaries ($B in the manifest; default build-verify)
#     -o DIR    output dir: logs/<name>.log, results.tsv, summary.txt (default $TMPDIR/ufsm-suite-<pid>)
#     -g LIST   GPU indices to use, e.g. "0" or "0,1" (default: every GPU nvidia-smi lists, most free memory first);
#               "none" runs no GPU test (they are reported SKIP)
#     -G N      GPU tests at a time per GPU (default 2: the GPU tests are small, two overlap well; env SUITE_GPU_SLOTS)
#     -x S      per-test timeout in seconds (default 1800)
#     -P        profile: strace marks the tests that open /dev/nvidia* (gpu column "+nv"), nvidia-smi samples their GPU memory
#     -l        list the selected tests and exit
# Manifest lines: NAME | TAGS | EST_SECONDS | COMMAND   (bash, run from the repo root, $B = the build dir; # comments)
#   tags: gpu (takes a GPU lock, CUDA_VISIBLE_DEVICES = that GPU), gpu2 (both GPUs when there are two), slow (test-slow tier),
#   flaky (a failure is retried once and reported FLAKY when the retry passes), solo (runs with nothing else in this suite).
# GPU locks are machine-wide ($SUITE_LOCK_DIR, default /tmp/ufsm-suite-locks/gpu<i>.<slot>): concurrent suites (several agents
# on one machine) share the GPUs one test at a time instead of piling up. Tests start longest-first (EST) so -j packs well.
# Exit 0 when every selected test passed (FLAKY counts as passed), 1 otherwise. A test that prints a line starting "SKIP" and exits
# 0 is reported SKIP.
set -u
H=$(cd "$(dirname "$0")/.." && pwd); cd "$H" || exit 2
MANIFEST=${SUITE_MANIFEST:-$H/tests/suite.txt}
TIER=fast; J=${SUITE_J:-}; T=${SUITE_THREADS:-}; BDIR=build-verify; OUT=; GPUS=; GSLOTS=${SUITE_GPU_SLOTS:-2}; TO=1800; PROF=0; LIST=0
while getopts "t:j:T:B:o:g:G:x:Plh" o; do case $o in
  t) TIER=$OPTARG;; j) J=$OPTARG;; T) T=$OPTARG;; B) BDIR=$OPTARG;; o) OUT=$OPTARG;; g) GPUS=$OPTARG;; G) GSLOTS=$OPTARG;;
  x) TO=$OPTARG;; P) PROF=1;; l) LIST=1;; *) sed -n '2,22p' "$0"; exit 2;; esac; done
shift $((OPTIND - 1)); FILTERS=("$@")
case $TIER in fast|slow|all) ;; *) echo "run_suite: tier $TIER (fast|slow|all)" >&2; exit 2;; esac

# CPUs this process may use: UFSM_NCPU, else min(nproc (affinity), every cgroup v2 cpu.max quota on the path to the root)
ncpu() {
  [ -n "${UFSM_NCPU:-}" ] && { echo "$UFSM_NCPU"; return; }
  local n q p d cg; n=$(nproc 2> /dev/null || echo 1)
  cg=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2> /dev/null); d=/sys/fs/cgroup${cg%/}
  while :; do
    if [ -r "$d/cpu.max" ] && read -r q p < "$d/cpu.max" && [ "$q" != max ] && [ "${p:-0}" -gt 0 ]; then
      q=$((q / p)); [ $q -lt 1 ] && q=1; [ $q -lt $n ] && n=$q; fi
    [ "$d" = /sys/fs/cgroup ] || [ -z "$d" ] && break; d=${d%/*}; [ -z "$d" ] && d=/sys/fs/cgroup
  done
  echo "$n"
}
NC=$(ncpu)
[ -z "$J" ] && { J=$((NC / 3)); [ $J -lt 1 ] && J=1; [ $J -gt 12 ] && J=12; }
[ -z "$T" ] && { T=$((NC / J)); [ $T -lt 2 ] && T=2; }

# GPUs, most free memory first
if [ -z "$GPUS" ]; then
  GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2> /dev/null | sort -t, -k2,2nr | cut -d, -f1 | tr -d ' ' | paste -sd, -)
  [ -z "$GPUS" ] && GPUS=none
fi
[ "$GPUS" = none ] && GL=() || IFS=, read -ra GL <<< "$GPUS"
NG=${#GL[@]}; GCAP=$((NG * GSLOTS))
LOCKD=${SUITE_LOCK_DIR:-/tmp/ufsm-suite-locks}; mkdir -p "$LOCKD" 2> /dev/null; chmod 1777 "$LOCKD" 2> /dev/null

# manifest -> selected tests
NAMES=(); declare -A TAGS EST CMD
while IFS= read -r line; do
  line=${line%%$'\r'}; [[ -z "${line// }" || $line == \#* ]] && continue
  IFS='|' read -r n tg es cm <<< "$line"
  n=$(echo $n); tg=$(echo $tg | tr -d ' '); es=$(echo $es); cm=$(echo "$cm" | sed 's/^ *//')
  [ -z "$n" ] || [ -z "$cm" ] && { echo "run_suite: bad manifest line: $line" >&2; exit 2; }
  case ",$tg," in *,slow,*) s=1;; *) s=0;; esac
  [ $TIER = fast ] && [ $s = 1 ] && continue
  [ $TIER = slow ] && [ $s = 0 ] && continue
  if [ ${#FILTERS[@]} -gt 0 ]; then m=0; for f in "${FILTERS[@]}"; do [[ $n =~ $f ]] && m=1; done; [ $m = 0 ] && continue; fi
  [ -n "${TAGS[$n]+x}" ] && { echo "run_suite: duplicate test name $n" >&2; exit 2; }
  NAMES+=("$n"); TAGS[$n]=$tg; EST[$n]=${es:-10}; CMD[$n]=$cm
done < "$MANIFEST"
[ ${#NAMES[@]} = 0 ] && { echo "run_suite: no test selected" >&2; exit 2; }
hastag() { case ",${TAGS[$1]}," in *,$2,*) return 0;; esac; return 1; }
isgpu() { hastag "$1" gpu || hastag "$1" gpu2; }
# longest first
mapfile -t QUEUE < <(for n in "${NAMES[@]}"; do echo "${EST[$n]} $n"; done | sort -k1,1nr -s | cut -d' ' -f2-)
if [ $LIST = 1 ]; then for n in "${QUEUE[@]}"; do printf '%-48s %-14s %6s  %s\n' "$n" "${TAGS[$n]:--}" "${EST[$n]}" "${CMD[$n]}"; done; exit 0; fi

[ -z "$OUT" ] && OUT=${TMPDIR:-/tmp}/ufsm-suite-$$
mkdir -p "$OUT/logs" "$OUT/res" "$OUT/tmp" || exit 2
OUT=$(cd "$OUT" && pwd)
TIMEBIN=; [ -x /usr/bin/time ] && TIMEBIN=/usr/bin/time
# a test binary in the build dir that no manifest line runs (a merge that added a test to the Makefile only): say so
for f in "$BDIR"/test_*; do [ -x "$f" ] || continue; grep -qwF "${f##*/}" "$MANIFEST" || echo "run_suite: WARNING: $f is not in $MANIFEST (add a line, or it never runs)"; done
echo "run_suite: ${#QUEUE[@]} tests ($TIER) from $BDIR, -j $J, $T threads/test, GPUs ${GPUS} x $GSLOTS, logs $OUT/logs ($(hostname), $NC CPUs)"

# profile: nvidia-smi samples every 0.5 s, each GPU process mapped to its test through /proc/<pid>/environ (UFSM_SUITE_TEST)
SAMPLER=
if [ $PROF = 1 ] && [ $NG -gt 0 ]; then
  ( while :; do
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2> /dev/null | while IFS=', ' read -r p m; do
        t=$(tr '\0' '\n' < /proc/$p/environ 2> /dev/null | sed -n 's/^UFSM_SUITE_TEST=//p'); [ -n "$t" ] && echo "$t $m"; done
      sleep 0.5; done ) >> "$OUT/gpumem.txt" 2> /dev/null &
  SAMPLER=$!
fi

# one test (runs in a background subshell): GPU lock, timeout, timing, one retry for flaky tests -> $OUT/res/<name>
# res line: status rc wall user sys maxrss_mb gpus attempts
run_one() {
  local n=$1 cm=${CMD[$1]} log=$OUT/logs/$1.log tmp=$OUT/tmp/$1 g= fds=() a rc st=FAIL wall=0 us=0 sy=0 rss=0 want=1
  if isgpu "$n"; then
    hastag "$n" gpu2 && [ $NG -ge 2 ] && want=2
    local got=() fd i k lf
    while [ ${#got[@]} -lt $want ]; do      # first free slot, GPUs in preference order; poll (machine-wide locks)
      for i in "${GL[@]}"; do
        [[ " ${got[*]} " == *" $i "* ]] && continue
        for ((k = 0; k < GSLOTS; k++)); do
          lf=$LOCKD/gpu$i.$k; exec {fd}> "$lf" || continue
          if flock -n $fd; then got+=("$i"); fds+=($fd); break; fi
          exec {fd}>&-
        done
        [ ${#got[@]} -ge $want ] && break
      done
      [ ${#got[@]} -lt $want ] && sleep 0.3
    done
    g=$(IFS=,; echo "${got[*]}")
  fi
  for a in 1 2; do
    rm -rf -- "$tmp"; mkdir -p "$tmp"
    local pre=(); [ $PROF = 1 ] && command -v strace > /dev/null && pre=(strace -f -qq --seccomp-bpf -e trace=openat -o "$OUT/tmp/$n.strace")
    command -v stdbuf > /dev/null && pre=(stdbuf -oL -eL "${pre[@]}")   # line-buffered stdout: logs interleave stdout and stderr in order
    local t0 t1; t0=$(date +%s.%N)
    ( export B="$BDIR" TMPDIR="$tmp" UFSM_TEST_TMP="$tmp" UFSM_SUITE_TEST="$n" UFSM_NCPU=$T UFSM_FIT_THREADS=$T UFSM_TIFF_THREADS=$T \
             OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
      if [ -n "$g" ]; then export CUDA_VISIBLE_DEVICES=$g; elif [ "$GPUS" = none ]; then export CUDA_VISIBLE_DEVICES=; fi
      echo $BASHPID > "$OUT/tmp/$n.pid"
      echo "# $n [${TAGS[$n]}] attempt $a gpu=${g:--} $(date '+%F %T') on $(hostname)"; echo "# $cm"
      if [ -n "$TIMEBIN" ]; then exec $TIMEBIN -f '%e %U %S %M' -o "$OUT/tmp/$n.time" timeout -k 20 "$TO" "${pre[@]}" bash -c "$cm"
      else TIMEFORMAT='%R %U %S 0'; { time timeout -k 20 "$TO" "${pre[@]}" bash -c "$cm" 2>&1; } 2> "$OUT/tmp/$n.time"; fi
    ) > "$log.$a" 2>&1 < /dev/null
    rc=$?; t1=$(date +%s.%N)
    read -r wall us sy rss < <(tail -n1 "$OUT/tmp/$n.time" 2> /dev/null); [[ ${wall:-x} =~ ^[0-9.]+$ ]] || { wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.2f", b - a}'); us=0; sy=0; rss=0; }
    rss=$(( ${rss:-0} / 1024 ))
    if [ $rc = 0 ]; then st=PASS; grep -q '^SKIP' "$log.$a" && st=SKIP; [ $a = 2 ] && st=FLAKY
    elif [ $rc = 124 ] || [ $rc = 137 ]; then st=TIMEOUT; else st=FAIL; fi
    [ $a = 1 ] && mv "$log.1" "$log" || { echo; echo "######## retry (attempt 1 failed above)"; cat "$log.2"; } >> "$log"
    rm -f "$log.2"
    [ $rc = 0 ] || ! hastag "$n" flaky && break
  done
  if [ $PROF = 1 ] && [ -f "$OUT/tmp/$n.strace" ]; then grep -q '/dev/nvidia' "$OUT/tmp/$n.strace" && g="${g:-}+nv"; rm -f "$OUT/tmp/$n.strace"; fi
  rm -f "$OUT/tmp/$n.pid"
  for fd in "${fds[@]}"; do exec {fd}>&-; done
  [ $rc = 0 ] && rm -rf -- "$tmp"
  echo "$st $rc $wall $us $sy $rss ${g:--} $a" > "$OUT/res/$n"
}

# scheduler: start longest-first while slots are free; GPU tests only while this suite has a GPU slot free (a CPU test goes instead)
declare -A PNAME PGPU; RUN=0; GBUSY=0; SOLO=0; DONE=0; TOT=${#QUEUE[@]}; T0=$(date +%s)
finish() {
  local p=$1 n=${PNAME[$1]} r; r=$(cat "$OUT/res/$n" 2> /dev/null || echo "FAIL ? 0 0 0 0 - 1")
  RUN=$((RUN - 1)); [ "${PGPU[$p]}" = 1 ] && GBUSY=$((GBUSY - 1)); hastag "$n" solo && SOLO=0; DONE=$((DONE + 1))
  unset "PNAME[$p]" "PGPU[$p]"
  set -- $r; printf '[%3d/%d %5ds] %-7s %7.1fs  %s\n' $DONE $TOT $(( $(date +%s) - T0 )) "$1" "$3" "$n"
}
cleanup() {   # the tests: timeout leads its own process group, its parent (time) pid is in tmp/<name>.pid
  trap - INT TERM; [ -n "$SAMPLER" ] && kill "$SAMPLER" 2> /dev/null; local f c
  for f in "$OUT"/tmp/*.pid; do [ -f "$f" ] || continue; for c in $(pgrep -P "$(cat "$f")" 2> /dev/null); do kill -TERM -- -"$c" 2> /dev/null; done; done
  for p in "${!PNAME[@]}"; do kill -- "$p" 2> /dev/null; done; }
trap 'cleanup; echo "run_suite: interrupted" >&2; exit 130' INT TERM
while [ ${#QUEUE[@]} -gt 0 ] || [ $RUN -gt 0 ]; do
  started=0
  if [ $SOLO = 0 ] && [ $RUN -lt $J ]; then
    for i in "${!QUEUE[@]}"; do
      n=${QUEUE[$i]}
      if isgpu "$n"; then
        if [ $NG = 0 ]; then echo "SKIP 0 0 0 0 0 - 0" > "$OUT/res/$n"; echo "# no GPU (-g none / no nvidia-smi)" > "$OUT/logs/$n.log"
          unset "QUEUE[$i]"; QUEUE=("${QUEUE[@]}"); DONE=$((DONE + 1)); printf '[%3d/%d] SKIP    (no GPU)  %s\n' $DONE $TOT "$n"; started=1; break; fi
        [ $GBUSY -ge $GCAP ] && continue
      fi
      if hastag "$n" solo; then [ $RUN -gt 0 ] && continue; SOLO=1; fi
      run_one "$n" & p=$!
      PNAME[$p]=$n; PGPU[$p]=0; isgpu "$n" && { PGPU[$p]=1; GBUSY=$((GBUSY + 1)); }; RUN=$((RUN + 1))
      unset "QUEUE[$i]"; QUEUE=("${QUEUE[@]}"); started=1; break
    done
  fi
  [ $started = 1 ] && continue
  [ $RUN = 0 ] && continue
  wait -n -p fin; [ -n "${fin:-}" ] && [ -n "${PNAME[$fin]+x}" ] && finish "$fin"
done
[ -n "$SAMPLER" ] && { kill "$SAMPLER" 2> /dev/null; wait "$SAMPLER" 2> /dev/null; }

# results
{ printf 'name\tstatus\trc\twall_s\tuser_s\tsys_s\tmaxrss_mb\tgpu\tattempts\tgpu_mem_mb\ttags\n'
  for n in "${NAMES[@]}"; do
    r=$(cat "$OUT/res/$n" 2> /dev/null || echo "FAIL ? 0 0 0 0 - 0"); set -- $r
    gm=0; [ -f "$OUT/gpumem.txt" ] && gm=$(awk -v t="$n" '$1 == t && $2 > m {m = $2} END {print m + 0}' "$OUT/gpumem.txt")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$gm" "${TAGS[$n]:--}"
  done; } > "$OUT/results.tsv"
WALL=$(( $(date +%s) - T0 ))
FAILS=$(awk -F'\t' 'NR > 1 && ($2 == "FAIL" || $2 == "TIMEOUT") {print $1}' "$OUT/results.tsv")
{ echo; echo "==== failing tests: logs (last 40 lines; full logs in $OUT/logs)"
  for n in $FAILS; do echo; echo "---- $n: $(awk -F'\t' -v t="$n" '$1 == t {print $2 " rc " $3 " after " $4 " s"}' "$OUT/results.tsv")"; echo "     $OUT/logs/$n.log"
    tail -n 40 "$OUT/logs/$n.log" | sed 's/^/     | /'; done
  echo
  echo "==== summary ($TIER, $BDIR, $(hostname), wall ${WALL} s, -j $J)"
  awk -F'\t' 'NR > 1 {c[$2]++; cpu += $5 + $6} END {for (s in c) printf "  %-8s %d\n", s, c[s]; printf "  cpu      %.0f s (user + sys)\n", cpu}' "$OUT/results.tsv"
  F=$(awk -F'\t' 'NR > 1 && $2 == "FLAKY" {printf " %s", $1}' "$OUT/results.tsv"); [ -n "$F" ] && echo "  flaky (passed on the retry):$F"
  [ -n "$FAILS" ] && echo "  FAILED: $(echo $FAILS)" || echo "  ALL PASSED"
} | tee "$OUT/summary.txt" | { [ -n "$FAILS" ] && cat || sed -n '/^==== summary/,$p'; }
echo "results: $OUT/results.tsv"
[ -z "$FAILS" ]
