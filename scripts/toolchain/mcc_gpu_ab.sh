#!/usr/bin/env bash
# mcc_gpu_ab.sh NVCC_BUILD MCC_BUILD OUT: the GPU A/B of mcc's PTX against nvcc (microc PLAN.md 4.3 / 5; stage B). Builds nothing; it runs two
# finished build dirs (e.g. a clang/nvcc release build and the mcc release build, cmake/mcc.cmake) in one GPU window on the desk, holding all six
# desk GPU locks for the whole run, and writes OUT/ab.tsv (one row per timed kernel / pass) and OUT/summary.txt.
#   jit      mcc only: the first-run cost: MCUDA_PRECOMPILE=JITN (8) of MCC_BUILD/ufsm into an empty cache OUT/jitcache (wall s, cubins, MB),
#            then every mcc run uses that cache (so no bench timing includes the driver's ptxas)
#   bench    every GPU bench_* (+ test_head_tc's timing section: the tensor-core head forward / backward-data), REPS (3) rounds, nvcc and
#            mcc alternating per round; per kernel the min ms, the ratio mcc / nvcc and the output check (hashes, checksums, error lines:
#            "same" when the line with its timing / rate numbers masked is identical across builds, "diff" when not, "nondet" when it
#            differs between rounds of one build). Families: MX-fp4 3^3 fwd / bdata / wgrad (bench_f4conv, bench_wgrad4, bench_wgl0),
#            stride 2 (bench_s2f, bench_s2b), 2:4 sparse (bench_sp24), attention (bench_attn: block, GEMMs, cores 1 / 2 = fp16 / lp_attn8),
#            heads (bench_head, bench_head2, test_head_tc), fine stem, GN, level-0 misc, the old fp16 / fp8 convs, Muon, whole-net steps
#   train    TRAIN_STEPS (200) steps of the live scratch2 config at TRAIN_P (768), from the latest trainer argv / env in
#            scratch2.resume.log (or CMDFILE: one line "BIN train args..."), --init-from the fixed checkpoint, --det 1 --gpu-input 1, both
#            cards; per build: samples/s (median of log.csv rows past step 50), wall s; per logged step the loss of both and |diff|
#   predict  the fixed checkpoint on the first unseen box (scripts/univ/unseen_boxes.tsv, PBOX^3 = 384^3 at its origin): --fit-zarr fields
#            and --heads all, every output file sha1-compared (identical / differing / only in one); wall s per build
# Fixed checkpoint: CKPT, else a copy of scratch2/last.ckpt taken once into OUT/fixed.ckpt (reruns reuse it).
# Prerequisites in BOTH build dirs: ufsm, test_head_tc and the bench_* below (cmake --build DIR --target ufsm test_head_tc bench_f4conv ...;
# missing ones are skipped and listed). Env: PARTS ("jit bench train predict"), REPS (3), GPU (0, the card the benches use), JITN (8),
# BENCH_RE (a regex over the bench lines below), TRAIN_STEPS, TRAIN_P, CMDFILE, CKPT, PBOX, TMO (per bench s, 900), LOCKWAIT (s, 1800).
# Only in a GPU window (the desk cards otherwise belong to scratch2; feedback-desk-gpu-scratch2-only); the merge3 queue line runs it.
set -u
NV=$(readlink -f "${1:?NVCC_BUILD}"); MC=$(readlink -f "${2:?MCC_BUILD}"); O=${3:?OUT}; mkdir -p "$O"; O=$(readlink -f "$O")
H=$(cd "$(dirname "$0")/../.." && pwd)
PARTS=${PARTS:-jit bench train predict}; REPS=${REPS:-3}; GPU=${GPU:-0}; JITN=${JITN:-8}; TMO=${TMO:-900}
TRAIN_STEPS=${TRAIN_STEPS:-200}; TRAIN_P=${TRAIN_P:-768}; PBOX=${PBOX:-384}
RUNS=/vesuvius/ufsm/v2/lat/runs
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$O/ab.log"; }
part() { [[ " $PARTS " == *" $1 "* ]]; }
MCAP=(); [ -x ~/bin/memcap ] && MCAP=("$HOME/bin/memcap" -t)
for b in "$NV" "$MC"; do [ -x "$b/ufsm" ] || { echo "mcc_gpu_ab: no $b/ufsm" >&2; exit 2; }; done
[ "$NV" != "$MC" ] || { echo "mcc_gpu_ab: the two build dirs are the same" >&2; exit 2; }

# ---- the window: all six desk GPU locks (as split_finite.sh / scratch2), held until exit
LOCKS=${AB_LOCKS-"/vesuvius/ufsm/opt/gpu0.lock /vesuvius/ufsm/opt/gpu1.lock /tmp/ufsm-suite-locks/gpu0.0 /tmp/ufsm-suite-locks/gpu0.1 /tmp/ufsm-suite-locks/gpu1.0 /tmp/ufsm-suite-locks/gpu1.1"}   # (AB_LOCKS= : none, dry runs of the parsers only)
mkdir -p /tmp/ufsm-suite-locks
fd=20; for lk in $LOCKS; do eval "exec $fd> $lk"; flock -w ${LOCKWAIT:-1800} $fd || { log "lock $lk busy"; exit 2; }; fd=$((fd + 1)); done
log "mcc_gpu_ab: nvcc $NV ($(sha1sum "$NV/ufsm" | cut -c1-8)) vs mcc $MC ($(sha1sum "$MC/ufsm" | cut -c1-8)); parts [$PARTS], reps $REPS, GPU $GPU"
nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,temperature.gpu,memory.used --format=csv,noheader >> "$O/ab.log" 2>&1
export MCUDA_CACHE=$O/jitcache; mkdir -p "$MCUDA_CACHE"   # (nvcc builds ignore it)

# ---- jit: the cold first-run cost of mcc's PTX (PLAN.md gap 14)
if part jit; then
  if [ -z "$(ls -A "$MCUDA_CACHE")" ]; then
    t0=$(date +%s.%N)
    env CUDA_VISIBLE_DEVICES=$GPU MCUDA_PRECOMPILE=$JITN "${MCAP[@]}" "$MC/ufsm" sysinfo > "$O/jit.log" 2>&1; rc=$?
    t1=$(date +%s.%N)
    log "jit: rc $rc, $(echo "$t1 - $t0" | bc) s wall at $JITN threads, $(ls "$MCUDA_CACHE" | wc -l) cubins, $(du -sm "$MCUDA_CACHE" | cut -f1) MB; $(grep -c 'compiled in' "$O/jit.log") kernels compiled"
    grep 'compiled in' "$O/jit.log" | sed -E 's/^mcuda: (.*): compiled in ([0-9.]+) s, ([0-9]+) bytes of PTX, ([0-9]+) of code/\2\t\3\t\4\t\1/' | sort -rn > "$O/jit_kernels.tsv"
    log "jit: slowest: $(head -3 "$O/jit_kernels.tsv" | awk -F'\t' '{printf "%s %.1f s; ", $4, $1}')"
  else log "jit: $MCUDA_CACHE already warm ($(ls "$MCUDA_CACHE" | wc -l) cubins), not re-timed"; fi
fi

# ---- bench: the kernels, per family (args sized for a 16 GB card; ENV=... prefixes allowed)
BENCHES=(
  "bench_f4conv 384"                       # MX-fp4 level-0 3^3: fwd (+GN stats), bdata, wgrad; out hashes
  "bench_wgrad4 20"                        # wgrad fp8 / fp4 / fp4+Hadamard at the 96^3 shapes
  "bench_wgl0 128 256 256 5"               # fp4 wgrad at the scratch2 level-0 widths (gy fp4)
  "GY8=1 bench_wgl0 128 256 256 5"         # (gy fp8)
  "bench_s2f"                              # stride-2 forward, 256..448 channels; hashes
  "bench_s2b"                              # stride-2 backward-data; hashes
  "bench_sp24"                             # 2:4 sparse vs dense MX-fp4 fwd / bdata
  "bench_attn"                             # attention block (cores 0 / 1, linear-mx 4)
  "BENCH_GEMM=1 bench_attn"                # its twelve GEMM shapes
  "BENCH_CORE_ONLY=1 bench_attn 288 16 1 1 10 0 1:4,2:4"   # attention products: fp16 core vs lp_attn8 (core 2)
  "bench_head 384"                         # 1^3 head + loss kernels; checksums
  "bench_head2 512 32"                     # production head over z-slabs, fused backward; hashes
  "test_head_tc"                           # tensor-core head fwd / backward-data vs the FFMA kernels (timing section + checks)
  "bench_fstem 512"                        # fine stem fwd / bwd; hash
  "bench_gn 96 2 16 50"                    # GN+SiLU backward fp16
  "UFSM_GN_X=4 bench_gn 96 2 16 50"        # (x MX-fp4)
  "bench_l0misc 512 5"                     # level-0 non-conv kernels; hashes
  "bench_conv"                             # fp16 / tf32 convs (old path)
  "bench_lp"                               # fp8 / fp4 vs bf16 accuracy + TFLOP/s
  "bench_muonc 20"                         # optimizer steps, production net
  "bench_muonx 20"                         # optimizer steps, 92 M net
  "bench_mem"                              # whole-net train / inference step, small P
  "bench_l0_ring"                          # level-0 rings, net step
  "bench_s2"                               # one desk card's scratch2 trunk share (the whole step)
)
[ -n "${BENCH_RE:-}" ] && { mapfile -t BENCHES < <(printf '%s\n' "${BENCHES[@]}" | grep -E -- "$BENCH_RE"); }
if part bench; then
  mkdir -p "$O/bench"; : > "$O/bench/index.tsv"; missing=()
  for i in "${!BENCHES[@]}"; do
    read -ra W <<< "${BENCHES[$i]}"; E=(); while [[ "${W[0]}" == *=* ]]; do E+=("${W[0]}"); W=("${W[@]:1}"); done
    exe=${W[0]}; [ -x "$NV/$exe" ] && [ -x "$MC/$exe" ] || { missing+=("$exe"); continue; }
    printf '%d\t%s\n' "$i" "${BENCHES[$i]}" >> "$O/bench/index.tsv"
    for r in $(seq 1 "$REPS"); do for t in nvcc mcc; do
      [ $t = nvcc ] && D=$NV || D=$MC; f=$O/bench/$i.$t.$r
      [ -s "$f.out" ] && grep -q '^# rc 0$' "$f.out" && continue   # (reruns keep finished rounds)
      ( cd "$D" && env CUDA_VISIBLE_DEVICES=$GPU UFSM_GPU=0 "${E[@]}" timeout -k 10 "$TMO" "${MCAP[@]}" "./$exe" "${W[@]:1}" > "$f.out" 2> "$f.err"; echo "# rc $?" >> "$f.out" )
    done; done
    log "bench $i [${BENCHES[$i]}]: rc nvcc $(tail -1 "$O/bench/$i.nvcc.1.out" | cut -c3-) mcc $(tail -1 "$O/bench/$i.mcc.1.out" | cut -c3-)"
  done
  [ ${#missing[@]} -gt 0 ] && log "bench: skipped (not built in both dirs): ${missing[*]}"
  python3 - "$O" <<'PY'
import sys, re, math, os, glob
O = sys.argv[1]; B = O + "/bench"
TIME = re.compile(r"(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(ms|s per)\b")
RATE = re.compile(r"(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(TFLOP/s|TF/s|GF/s|GB/s|MB/s|windows/s|M voxels/s|Mvox/s|TF)\b")
DERIV = re.compile(r"(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)(\s*(?:x\b|%|GB\b|percent))")   # time-derived ratios, GPU idle %, device GB: not checked
TABLE = {"bench_sp24": re.compile(r"(?<![\w.])(\d+\.\d+)(?![\dx.])"), "bench_wgrad4": re.compile(r"(?<![\w.])(\d+\.\d+)(?![\d.])")}
def parse(path, exe):
    rows = []
    try: L = open(path, errors="replace").read().splitlines()
    except OSError: return rows, None
    rc = None
    for i, ln in enumerate(L):
        if ln.startswith("# rc "): rc = int(ln[5:]); continue
        t = [(float(m.group(1)) * (1000 if m.group(2) == "s per" else 1), "ms") for m in TIME.finditer(ln)]
        r = [(float(m.group(1)), "rate") for m in RATE.finditer(ln)]
        if exe in TABLE and not t and re.match(r"^\s*\S", ln) and not re.search(r"\bms\b", ln):
            t = [(float(x), "ms") for x in TABLE[exe].findall(ln)]
        chk = DERIV.sub(r"#\2", RATE.sub(r"#\2", TIME.sub(r"#\2", ln)))
        if exe in TABLE: chk = re.sub(r"\s\d+$", " #", TABLE[exe].sub("#", chk))   # (sp24's last column: TF/s as an integer)
        chk = re.sub(r"\b\d+(\.\d+)?\s*s\b", "# s", chk)
        rows.append((i, ln, t if t else r, chk))
    return rows, rc
idx = [l.rstrip("\n").split("\t", 1) for l in open(B + "/index.tsv") if l.strip()]
out = open(O + "/ab.tsv", "w")
out.write("bench\tline\titem\tkind\tnvcc\tmcc\tratio\tflag\tcheck\tlabel\n")
summ = []
for bi, spec in idx:
    exe = [w for w in spec.split() if "=" not in w][0]
    runs = {t: [parse(f, exe) for f in sorted(glob.glob(f"{B}/{bi}.{t}.*.out"))] for t in ("nvcc", "mcc")}
    rcs = {t: [rc for _, rc in runs[t]] for t in runs}
    if not runs["nvcc"] or not runs["mcc"]: continue
    base = {t: runs[t][0][0] for t in runs}
    nondet = {t: any([c for *_, c in rr[0]] != [c for *_, c in runs[t][0][0]] for rr in runs[t][1:]) for t in runs}
    if len(base["nvcc"]) != len(base["mcc"]): summ.append(f"{spec}: line count differs (nvcc {len(base['nvcc'])}, mcc {len(base['mcc'])}), rc {rcs}")
    worst = []
    for k, (i, ln, tv, chk) in enumerate(base["nvcc"]):
        if k >= len(base["mcc"]): break
        mi, mln, mtv, mchk = base["mcc"][k]
        if not tv and chk == mchk: continue
        check = "same" if chk == mchk else ("nondet" if nondet["nvcc"] or nondet["mcc"] else "diff")
        if not tv: out.write(f"{spec}\t{i}\t-\tcheck\t\t\t\t\t{check}\t{ln.strip()[:160]}\n"); continue
        for j, (v, kind) in enumerate(tv):
            def best(t):
                vals = []
                for rows, _ in runs[t]:
                    if k < len(rows) and j < len(rows[k][2]): vals.append(rows[k][2][j][0])
                return (min(vals) if kind == "ms" else max(vals)) if vals else float("nan")
            a, b = best("nvcc"), best("mcc")
            ratio = (b / a if kind == "ms" else a / b) if a > 0 and b > 0 else float("nan")
            flag = "SLOW" if ratio > 1.03 else ("fast" if ratio < 0.97 else "")
            if ratio == ratio and (kind != "ms" or a >= 0.05): worst.append((ratio, ln.strip()[:70]))   # (sub-50-us timings: launch noise)
            out.write(f"{spec}\t{i}\t{j}\t{kind}\t{a:.4g}\t{b:.4g}\t{ratio:.3f}\t{flag}\t{check}\t{ln.strip()[:160]}\n")
    ndiff = sum(1 for (i, ln, tv, chk), (_, _, _, mchk) in zip(base["nvcc"], base["mcc"]) if chk != mchk)
    if worst:
        g = math.exp(sum(math.log(r) for r, _ in worst) / len(worst)); w = max(worst)
        summ.append(f"{spec}: {len(worst)} timings, geomean mcc/nvcc {g:.3f}, worst {w[0]:.3f} ({w[1]}); {sum(r > 1.03 for r, _ in worst)} > 1.03; "
                    f"{ndiff} check lines differ{' (nondeterministic)' if nondet['nvcc'] or nondet['mcc'] else ''}; rc nvcc {rcs['nvcc']} mcc {rcs['mcc']}")
    else: summ.append(f"{spec}: no timings parsed; {ndiff} check lines differ; rc nvcc {rcs['nvcc']} mcc {rcs['mcc']}")
out.close()
open(O + "/bench_summary.txt", "w").write("\n".join(summ) + "\n")
PY
  log "bench: $(grep -c . "$O/ab.tsv") rows in ab.tsv; $(awk -F'\t' '$8 == "SLOW"' "$O/ab.tsv" | wc -l) timings > 3 % slower under mcc; $(awk -F'\t' '$9 == "diff"' "$O/ab.tsv" | wc -l) output checks differ"
fi

# ---- the fixed checkpoint (train --init-from, predict)
CK=${CKPT:-}
if part train || part predict; then
  if [ -z "$CK" ]; then CK=$O/fixed.ckpt; [ -s "$CK" ] || { cp "$RUNS/scratch2/last.ckpt" "$CK.tmp" && mv "$CK.tmp" "$CK"; log "fixed checkpoint: copy of $RUNS/scratch2/last.ckpt ($(sha1sum "$CK" | cut -c1-8))"; }; fi
  [ -s "$CK" ] || { log "no checkpoint $CK"; exit 2; }
fi

# ---- train: the live config, TRAIN_STEPS steps, both builds
if part train; then
  if [ -n "${CMDFILE:-}" ]; then read -ra A < "$CMDFILE"; TENV=(MALLOC_ARENA_MAX=4 UFSM_HEAD_SLAB_Z=16 UFSM_HOSTMEM=1)
  else
    RL=$RUNS/scratch2.resume.log
    read -ra A <<< "$(grep ' trainer pid [0-9]* argv: ' "$RL" | tail -1 | sed 's/.* argv: //')"
    read -ra TENV <<< "$(grep ' trainer pid [0-9]* env: ' "$RL" | tail -1 | sed 's/.* env: //')"
  fi
  [ "${A[1]:-}" = train ] || { log "train: no trainer argv (CMDFILE or $RUNS/scratch2.resume.log)"; exit 2; }
  A=("${A[@]:2}"); T=(); skip=0
  for a in "${A[@]}"; do   # drop the run-specific / overridden options, then append ours (opt() takes the first occurrence)
    if ((skip)); then skip=0; continue; fi
    case "$a" in --out|--steps|--P|--resume|--init-from|--ckpt-every|--val-every|--log-every|--det|--gpu-input) skip=1; continue;; esac; T+=("$a"); done
  for t in nvcc mcc; do
    [ $t = nvcc ] && D=$NV || D=$MC; R=$O/train_$t
    [ -s "$R.done" ] && continue
    case "$R" in "$O"/train_*) rm -rf -- "$R" ;; esac
    t0=$(date +%s.%N)
    env "${TENV[@]}" timeout -k 30 $((TRAIN_STEPS * 60 + 1800)) "${MCAP[@]}" "$D/ufsm" train "${T[@]}" --out "$R" --steps "$TRAIN_STEPS" --P "$TRAIN_P" \
        --init-from "$CK" --ckpt-every 1000000 --val-every 1000000 --log-every 10 --det 1 --gpu-input 1 > "$R.log" 2>&1; rc=$?
    t1=$(date +%s.%N); echo "rc $rc wall $(echo "$t1 - $t0" | bc)" > "$R.done"; log "train $t: $(cat "$R.done")"
  done
  python3 - "$O" "$TRAIN_STEPS" <<'PY'
import sys, re, csv, statistics
O, N = sys.argv[1], int(sys.argv[2]); res = {}
for t in ("nvcc", "mcc"):
    loss, sps = {}, []
    for ln in open(f"{O}/train_{t}.log", errors="replace"):
        m = re.match(r"step\s+(\d+) lr \S+ loss (\S+)", ln)
        if m: loss[int(m.group(1))] = float(m.group(2))
    try:
        for row in csv.DictReader(open(f"{O}/train_{t}/log.csv")):
            if row.get("samples_per_s") and int(row["step"]) >= 50: sps.append(float(row["samples_per_s"]))
    except OSError: pass
    done = open(f"{O}/train_{t}.done").read().split()
    res[t] = (loss, statistics.median(sps) if sps else float("nan"), int(done[1]), float(done[3]))
(ln, sn, rn, wn), (lm, sm, rm, wm) = res["nvcc"], res["mcc"]
with open(f"{O}/train_ab.tsv", "w") as f:
    f.write("step\tloss_nvcc\tloss_mcc\tabs_diff\n")
    for s in sorted(set(ln) | set(lm)):
        a, b = ln.get(s, float("nan")), lm.get(s, float("nan")); f.write(f"{s}\t{a:.5f}\t{b:.5f}\t{abs(a - b):.5f}\n")
common = [s for s in ln if s in lm]; d = [abs(ln[s] - lm[s]) for s in common]
same = sum(ln[s] == lm[s] for s in common)
msg = (f"train ({N} steps): rc nvcc {rn} mcc {rm}; samples/s median (step >= 50) nvcc {sn:.4g} mcc {sm:.4g} -> mcc step time x{sn / sm if sm > 0 else float('nan'):.3f}; "
       f"wall nvcc {wn:.0f} s mcc {wm:.0f} s; losses at {len(common)} logged steps: {same} identical, mean |diff| {statistics.mean(d) if d else float('nan'):.4g}, "
       f"max {max(d) if d else float('nan'):.4g}; final nvcc {ln.get(max(ln), float('nan')) if ln else float('nan'):.4f} mcc {lm.get(max(lm), float('nan')) if lm else float('nan'):.4f}")
open(f"{O}/train_summary.txt", "w").write(msg + "\n")
PY
  log "$(cat "$O/train_summary.txt")"
fi

# ---- predict: byte-compare on a fixed checkpoint and box
if part predict; then
  IFS=$'\t' read -r un sc ur uk uum ulv uz uy ux unz uny unx uax _ <<< "$(grep -v '^#' "$H/scripts/univ/unseen_boxes.tsv" | head -1)"
  bx=$uz,$uy,$ux,$((unz < PBOX ? unz : PBOX)),$((uny < PBOX ? uny : PBOX)),$((unx < PBOX ? unx : PBOX))
  for t in nvcc mcc; do
    [ $t = nvcc ] && D=$NV || D=$MC; R=$O/pred_$t
    [ -s "$R.done" ] && continue
    case "$R" in "$O"/pred_*) rm -rf -- "$R" ;; esac; mkdir -p "$R"
    t0=$(date +%s.%N)
    env CUDA_VISIBLE_DEVICES=$GPU "${MCAP[@]}" "$D/ufsm" predict "$CK" "$ur" "$uk" "$R/f.tmp" --box $bx --fit-zarr "$R/fields" --um "$uum" --level "$ulv" \
        --window 640 --halo 64 --shard 512 --grid-origin 0,0,0 --ema 1 --q 8 --threads 12 --gpu 0 --axis "$uax" --levels 1 --fit-h 2 --fit-q 8 \
        --fit-q-phase 0 --no-out 1 > "$R/fields.log" 2>&1; r1=$?
    t1=$(date +%s.%N)
    env CUDA_VISIBLE_DEVICES=$GPU "${MCAP[@]}" "$D/ufsm" predict "$CK" "$ur" "$uk" "$R/h" --um "$uum" --level "$ulv" --box $bx --full-grid 1 --levels 1 \
        --heads all --gpu 0 --axis "$uax" --threads 6 > "$R/heads.log" 2>&1; r2=$?
    t2=$(date +%s.%N); rm -rf -- "$R/f.tmp"
    echo "rc $r1 $r2 wall $(echo "$t1 - $t0" | bc) $(echo "$t2 - $t1" | bc)" > "$R.done"; log "predict $t ($un $bx): $(cat "$R.done") (fields, heads)"
  done
  (cd "$O/pred_nvcc" && find fields h -type f 2>/dev/null | sort | xargs -r sha1sum) > "$O/pred_nvcc.sha1"
  (cd "$O/pred_mcc" && find fields h -type f 2>/dev/null | sort | xargs -r sha1sum) > "$O/pred_mcc.sha1"
  join -j 2 -a 1 -a 2 -e MISSING -o 0,1.1,2.1 <(sort -k2 "$O/pred_nvcc.sha1") <(sort -k2 "$O/pred_mcc.sha1") > "$O/pred_cmp.txt"
  log "predict: $(awk '$2 == $3' "$O/pred_cmp.txt" | wc -l) files identical, $(awk '$2 != $3 && $2 != "MISSING" && $3 != "MISSING"' "$O/pred_cmp.txt" | wc -l) differ, $(grep -c MISSING "$O/pred_cmp.txt") only in one build (pred_cmp.txt; differing: $(awk '$2 != $3 {print $1}' "$O/pred_cmp.txt" | head -8 | tr '\n' ' '))"
fi

{ echo "mcc GPU A/B $(date '+%F %H:%M'): nvcc $NV vs mcc $MC"; grep -E '^[0-9-]+ [0-9:]+ (jit|predict|train (nvcc|mcc):)' "$O/ab.log" | cut -d' ' -f3-
  [ -s "$O/train_summary.txt" ] && cat "$O/train_summary.txt"; [ -s "$O/bench_summary.txt" ] && cat "$O/bench_summary.txt"
  [ -s "$O/ab.tsv" ] && { echo "-- timings > 3 % slower under mcc (ratio, nvcc ms, mcc ms, bench, line):"
    awk -F'\t' '$8 == "SLOW" {printf "%s\t%s\t%s\t%s\t%s\n", $7, $5, $6, $1, $10}' "$O/ab.tsv" | sort -rn | head -60; }
} > "$O/summary.txt"
log "done: $O/summary.txt, $O/ab.tsv"
