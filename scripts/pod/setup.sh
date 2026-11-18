#!/usr/bin/env bash
# scripts/pod/setup.sh [SCROLL]: make a rented GPU pod ready for a whole-scroll unroll. Idempotent: every step checks
# its result and skips the work when it is already done. Re-run it after a reboot, a failed download or a master update.
#   1. limits: the REAL cgroup CPU / RAM limits. nproc and free show the host: 256 CPUs and 1 TB on a pod whose cgroup allows
#      13.6 CPUs and 62 GB. Also the GPU / MIG slice, CUDA and disk.
#      -> $POD/limits.txt; then `ufsm sysinfo` once the binary exists.
#   2. deps: apt packages (root only), skipped when they are all present (xxhash: zarr3.c since iospeed): LLVM 22 (clang, lld,
#      llvm, compiler-rt; from apt.llvm.org when the distribution lacks it), cmake, ninja, ccache, libcurl / zstd / blosc / zlib /
#      ssl / jpeg / xxhash dev, jq, curl, rsync, time. The build is mcc (microc, scripts/toolchain/install_mcc.sh); clang stays for mcc's
#      stage 0 / glibc link driver and the llvm-release fallback preset. Python tifffile (pip) for test_img_gate and inksdk's mkarms.
#   3. clone or fast-forward ufsm ($BRANCH, default master) into $POD/ufsm, then build it.
#      CMake (cmake/llvm.cmake) finds the newest clang and /usr/local/cuda-* by itself: cmake --preset release (arch86 for
#      non-12.x GPUs: software MX, since the fp4 / fp8 kernels need sm_120a), cmake --build -j <cgroup CPUs>. Always on the pod:
#      builds are native (-march=native, SASS for the pod's GPU only), so a desk binary does not run there.
#   4. smoke test: ufsm sysinfo, test_sysinfo (cgroup parsing), test_mx_sw (GPU kernels).
#   5. checkpoints from the public mirror, resumable, size-checked: wind96-wk1-best (fit fields) and ift3-24000 (ink) -> $POD/ckpt.
#   6. SCROLL given: its CT from the mirror. `ufsm mirror` copies shard files resumably, about 40 MB/s on a pod; desk -> pod
#      rsync manages only about 1.6 MB/s, so nothing comes via the desk. Levels LV..LV+3 of the scan in scripts/unroll/scrolls.tsv
#      or scripts/corpus/list.tsv -> $POD/vol/<S>/<scan>.zarr/<L>.
#      Also any published ufsm-c2f-20261028 products: box.txt, axis.json, ink.zarr (ink only where uploaded). The box is
#      otherwise made on the pod (scanprep --box-out), the axis by unroll (`ufsm axis`), the ink by unroll.
#      Writes $POD/scrolls/<S>.env for scripts/pod/unroll.sh.
# Env: POD (default /root/pod as root, else ~/pod), BRANCH (master), REPO (https://github.com/SuperOptimizer/ufsm), MIRROR,
#   NOAPT=1 (skip apt), NOBUILD=1 (reuse the binary if built on this machine), FOREIGN_BINARY=1 (reuse it anyway), NOSMOKE=1, LEVELS ("L ..." override), MTH (mirror threads, 12; the mirror answers 429 past 64
#   requests in flight per IP), ARCH (auto). Log: $POD/setup.log.
set -uo pipefail
S=${1:-}
POD=${POD:-$([ "$(id -u)" = 0 ] && echo /root/pod || echo $HOME/pod)}
BRANCH=${BRANCH:-master}; REPO=${REPO:-https://github.com/SuperOptimizer/ufsm}
BASE=${BASE:-https://dl.ash2txt.org/community-uploads/forrest}; MIRROR=${MIRROR:-$BASE/volcomp}
MTH=${MTH:-12}; C2F=ufsm-c2f-20261028
mkdir -p $POD/{ckpt,vol,scrolls,state} || exit 1
exec > >(tee -a $POD/setup.log) 2>&1
say() { echo "$(date '+%F %T') setup: $*"; }
die() { say "FAILED: $*"; exit 1; }

# --- 1. limits (shell: before ufsm exists) ---
cg_path() { local p; p=$(awk -F: '$1 == 0 {print $3}' /proc/self/cgroup 2>/dev/null); [ "$p" = / ] && p=; echo "$p"; }
cg_min() {   # cg_min FILE: the smallest numeric first field of FILE over this cgroup, its ancestors and the root ("max" ignored)
  local p=$(cg_path) best= v
  while :; do
    v=$(awk '{print $1}' /sys/fs/cgroup$p/$1 2>/dev/null)
    [ -n "$v" ] && [ "$v" != max ] && { [ -z "$best" ] || [ "$v" -lt "$best" ]; } && best=$v
    [ -z "$p" ] && break; p=${p%/*}
  done; echo "${best:-}"; }
ncpu_cg() {   # floor(cpu.max quota / period), capped by the affinity count
  local n=$(nproc) p=$(cg_path) best= q per
  while :; do
    [ -r /sys/fs/cgroup$p/cpu.max ] && read -r q per < /sys/fs/cgroup$p/cpu.max && [ "$q" != max ] && [ -n "$q" ] && {
      local c=$(( q / per )); [ $c -lt 1 ] && c=1; { [ -z "$best" ] || [ $c -lt $best ]; } && best=$c; }
    [ -z "$p" ] && break; p=${p%/*}
  done
  [ -n "$best" ] && [ $best -lt $n ] && n=$best; echo $n; }
NCPU=$(ncpu_cg); MEMMAX=$(cg_min memory.max)
MEMGB=$(( ${MEMMAX:-$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)} >> 30 ))
{
  echo "date $(date -u '+%F %T')Z host $(hostname)"
  echo "nproc $(nproc), cgroup $(cg_path) cpu.max: $(cat /sys/fs/cgroup$(cg_path)/cpu.max /sys/fs/cgroup/cpu.max 2>/dev/null | tr '\n' ' ') -> NCPU $NCPU"
  echo "memory.max ${MEMMAX:-none} -> MEMGB $MEMGB (free says $(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo) GB)"
  nvidia-smi -L 2>&1 | sed 's/^/gpu /'
  nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>&1 | sed 's/^/gpu driver, cc: /'
  ls -d /usr/local/cuda*/ 2>/dev/null | tr '\n' ' ' | sed 's/^/cuda installs: /'; echo
  df -h $POD | tail -1 | sed 's/^/disk /'
} > $POD/limits.txt
cat $POD/limits.txt
say "pod $POD: $NCPU CPUs, $MEMGB GB (cgroup)"

# --- 2. deps ---
LLVMV=22
PKGS="clang-$LLVMV lld-$LLVMV llvm-$LLVMV libclang-rt-$LLVMV-dev cmake ninja-build ccache git pkg-config ca-certificates curl wget gnupg jq rsync time libcurl4-openssl-dev libzstd-dev libblosc-dev zlib1g-dev libssl-dev libjpeg-dev libxxhash-dev"
if [ "${NOAPT:-0}" != 1 ] && command -v apt-get > /dev/null && ! apt-cache policy clang-$LLVMV 2>/dev/null | grep -q 'Candidate: [0-9]'; then
  [ "$(id -u)" = 0 ] || die "clang-$LLVMV has no apt candidate and not root: add apt.llvm.org or NOAPT=1"
  say "apt: adding apt.llvm.org for clang-$LLVMV"
  export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq --no-install-recommends wget gnupg lsb-release software-properties-common > /dev/null
  wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && bash /tmp/llvm.sh $LLVMV > /dev/null || die "apt.llvm.org clang-$LLVMV"
fi
if [ "${NOAPT:-0}" != 1 ] && command -v apt-get > /dev/null; then
  miss=$(for p in $PKGS; do dpkg -s $p > /dev/null 2>&1 || echo $p; done | tr '\n' ' ')
  if [ -n "${miss// /}" ]; then
    [ "$(id -u)" = 0 ] || die "missing packages ($miss) and not root: install them or NOAPT=1"
    say "apt: $miss"
    export DEBIAN_FRONTEND=noninteractive; apt-get update -qq || die "apt-get update"
    # one failing package must not drop the rest: retry one by one, then check the ones the build needs
    apt-get install -y -qq --no-install-recommends $miss > /dev/null || for p in $miss; do apt-get install -y -qq --no-install-recommends $p > /dev/null || say "apt: $p failed"; done
  else say "deps present"; fi
fi
# xxhash (zarr3.c) is the one fresh pod images lack: fail here, not halfway through the build
[ -e /usr/include/xxhash.h ] || [ -e /usr/local/include/xxhash.h ] || die "xxhash.h missing: apt-get install libxxhash-dev"
for t in git cmake ninja curl jq pkg-config; do command -v $t > /dev/null || die "$t missing"; done
# python tifffile: test_img_gate and inksdk's mkarms read TIFs with it; pod images ship numpy but not tifffile (pod 11-18: the gate's one
# fail, mkarms A). A failed install is a warning: the unroll itself is all ufsm
python3 -c 'import tifffile' 2> /dev/null || { python3 -m pip install -q tifffile > /dev/null 2>&1 || python3 -m pip install -q --break-system-packages tifffile > /dev/null 2>&1; } ||
  say "pip: tifffile install failed (test_img_gate / mkarms need it)"

# --- 3. clone + build ---
U=$POD/ufsm/build/ufsm
if [ ! -d $POD/ufsm/.git ]; then git clone -q $REPO $POD/ufsm || die "git clone"; fi
( cd $POD/ufsm && git fetch -q origin && git checkout -q $BRANCH && { git diff --quiet && git merge -q --ff-only origin/$BRANCH || echo "setup: local changes in $POD/ufsm: not updated"; } ) || die "git update"
say "ufsm $(git -C $POD/ufsm log --oneline -1)"
CUDAP=$(ls -d /usr/local/cuda/bin /usr/local/cuda-*/bin 2>/dev/null | sort -V | tail -1); [ -n "$CUDAP" ] && export PATH=$PATH:$CUDAP
if [ -z "${ARCH:-}" ]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  case "$cc" in 12*|'') ARCH=120;; *) ARCH=86;; esac   # (lp_* need the 12.x block-scaled MMA; others: software MX)
fi
BD=build; PRESET=release; [ "$ARCH" = 86 ] && BD=build86 && PRESET=arch86 && U=$POD/ufsm/build86/ufsm
# the toolchain: microc (mcc + mcudahw) at cmake/mcc.version into ~/.local/mcc (cmake/mcc.cmake finds it there). Stage 0 and mcc's
# glibc-mode link need a C driver: /usr/bin/cc, else the newest clang (MCC_CC, exported for the build)
if [ ! -x /usr/bin/cc ]; then
  MCC_CC=$(ls /usr/bin/clang-[0-9]* /usr/lib/llvm-*/bin/clang 2>/dev/null | sort -V | tail -1); export MCC_CC
  [ -n "$MCC_CC" ] || die "no /usr/bin/cc and no clang: mcc needs a C driver for stage 0 and its glibc link"
  export CC=$MCC_CC
fi
if [ "${NOBUILD:-0}" != 1 ]; then
  say "mcc $(cat $POD/ufsm/cmake/mcc.version)"
  bash $POD/ufsm/scripts/toolchain/install_mcc.sh > $POD/mcc.log 2>&1 || { tail -20 $POD/mcc.log; die "install_mcc (log $POD/mcc.log)"; }
fi
# Native builds (2026-11-14: -march=native, SASS for this GPU only): a binary runs only where it was built. NOBUILD=1 reuses one
# only when its build dir's native_id (cmake/native_id.sh at configure) matches this machine; FOREIGN_BINARY=1 is the explicit
# override for a binary or objects built elsewhere (e.g. compiled on the desk), at the user's risk (SIGILL / no kernel image).
if [ "${NOBUILD:-0}" = 1 ] && [ "${FOREIGN_BINARY:-0}" != 1 ] && [ "$(cat $POD/ufsm/$BD/native_id 2>/dev/null)" != "$(sh $POD/ufsm/cmake/native_id.sh)" ]; then
  say "NOBUILD=1 but $BD was not built on this machine (native_id differs): building (FOREIGN_BINARY=1 overrides)"; NOBUILD=0
fi
if [ "${NOBUILD:-0}" != 1 ]; then
  say "cmake --preset $PRESET, ninja -j$NCPU"
  ( cd $POD/ufsm && cmake --preset $PRESET > $POD/build.log 2>&1 && nice -n 5 cmake --build $BD -j$NCPU --target ufsm >> $POD/build.log 2>&1 ) ||
    { tail -30 $POD/build.log; die "build (log $POD/build.log)"; }
fi
[ -x $U ] || die "no $U"
$U sysinfo --disk $POD | tee $POD/sysinfo.txt
# mcudahw JIT-compiles each kernel's PTX at its first launch (~/.cache/mcudahw, keyed by PTX + GPU model + driver): fill the cache now,
# on all CPUs, not inside step 1. Every run (also NOBUILD=1: a new driver or GPU model after a reboot starts cold; a warm cache is a scan)
if [ "${NOPRECOMPILE:-0}" != 1 ]; then
  say "mcudahw: precompiling the kernels ($NCPU threads)"
  jm=$(JITLOG=$POD/precompile.log bash $POD/ufsm/scripts/toolchain/jit_warm.sh $U $NCPU); jrc=$?; say "$jm"
  [ $jrc = 0 ] || say "precompile failed (rc $jrc, log $POD/precompile.log); kernels will JIT at first use"
fi

# --- 4. smoke test ---
if [ "${NOSMOKE:-0}" != 1 ] && [ ! -e $POD/state/smoke.$(git -C $POD/ufsm rev-parse --short HEAD) ]; then
  ( cd $POD/ufsm && cmake --build $BD -j$NCPU --target test_sysinfo test_mx_sw >> $POD/build.log 2>&1 &&
    timeout -k 10 120 ./$BD/test_sysinfo && flock -w 1800 /root/gpu.lock timeout -k 10 600 ./$BD/test_mx_sw > $POD/smoke.log 2>&1 )   # (GPU lock + timeout: a hang no longer blocks setup forever; mini audit podscripts #2) || { tail -20 $POD/smoke.log 2>/dev/null; die "smoke test"; }
  touch $POD/state/smoke.$(git -C $POD/ufsm rev-parse --short HEAD); say "smoke test passed"
fi

# --- 5. checkpoints ---
fetch() {   # fetch URL OUT: resumable, complete when the size matches Content-Length
  local want have; want=$(curl -sfIL "$1" | awk 'tolower($1) == "content-length:" {n = $2} END {print n + 0}' | tr -d '\r')
  have=$(stat -c %s "$2" 2>/dev/null || echo 0)
  [ "$want" -gt 0 ] && [ "$have" = "$want" ] && return 0
  curl -sfL --retry 5 -C - -o "$2" "$1" || curl -sfL --retry 5 -o "$2" "$1" || return 1
  have=$(stat -c %s "$2"); [ "$want" = 0 ] || [ "$have" = "$want" ] || { echo "fetch: $2 has $have of $want bytes"; return 1; }; }
for c in wind96-wk1-best; do fetch $BASE/ufsm/checkpoints/$c/$c.ckpt $POD/ckpt/$c.ckpt || die "checkpoint $c"; done   # (ift3 dropped 11-13: the latent ink head, /root/pod/latink.sh)
# the production ink (unroll INKMODE=r3, 2026-11-17): Reader V3 (MIT) from HuggingFace at the pinned revision, sha256-checked
R3URL=https://huggingface.co/domenicor046/reader-v3/resolve/aee1d64bced4eb35546348f3d276b9e2a0b99c46/reader-v3.pth
R3SHA=ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a
[ "$(sha256sum $POD/ckpt/reader-v3.pth 2> /dev/null | cut -d' ' -f1)" = $R3SHA ] || { fetch $R3URL $POD/ckpt/reader-v3.pth && [ "$(sha256sum $POD/ckpt/reader-v3.pth | cut -d' ' -f1)" = $R3SHA ] || die "reader-v3 (sha256)"; }
say "checkpoints: $(ls -l $POD/ckpt | awk 'NR > 1 {printf "%s %.0f MB  ", $9, $5 / 1e6}')"

# --- 6. the scroll ---
[ -n "$S" ] || { say "done (no SCROLL: CT not fetched)"; exit 0; }
row=; for t in $POD/ufsm/scripts/unroll/scrolls.tsv $POD/ufsm/scripts/corpus/list.tsv; do
  [ -e $t ] && row=$(awk -F'\t' -v s=$S '$1 == s' $t) && [ -n "$row" ] && break; done
[ -n "$row" ] || die "$S is in neither scripts/unroll/scrolls.tsv nor scripts/corpus/list.tsv"
IFS=$'\t' read -r _ _ K UM LV AXS TBOX TZR TSTRIP <<< "$row"
# the mirror key: SCROLL/volumes/SCAN; a key that already names volumes/ (scrolls.tsv rows of http CT, e.g. PHerc0332) is kept as is
# (podunroll 11-17: it became PHerc0332/volumes/volumes/..., a 404)
case $K in */volumes/*) MK=$K;; *) MK=$(dirname $K)/volumes/$(basename $K);; esac; V=$POD/vol
mkdir -p $V/$K; fetch $MIRROR/$MK/zarr.json $V/$K/zarr.json || die "$MK/zarr.json"
for l in ${LEVELS:-$(seq $LV $((LV + 3)))}; do
  curl -sfI $MIRROR/$MK/$l/zarr.json > /dev/null || continue
  [ -e $POD/state/ct.$S.$l ] && continue
  say "CT $MK level $l"
  nice -n 5 $U mirror $MIRROR $MK/$l $V/$K/$l --threads $MTH --verify 2 || die "mirror $MK/$l"
  touch $POD/state/ct.$S.$l
done
P=$POD/scrolls/$S; mkdir -p $P
for f in box.txt axis.json; do curl -sf $MIRROR/$S/representations/$C2F/$f -o $P/$f.tmp && mv $P/$f.tmp $P/$f; rm -f $P/$f.tmp; done
if curl -sfI $MIRROR/$S/representations/$C2F/ink.zarr/zarr.json > /dev/null && [ ! -e $POD/state/ink.$S ]; then
  mkdir -p $P/ink.zarr; fetch $MIRROR/$S/representations/$C2F/ink.zarr/zarr.json $P/ink.zarr/zarr.json
  for l in 0 1 2 3 4 5; do curl -sfI $MIRROR/$S/representations/$C2F/ink.zarr/$l/zarr.json > /dev/null || continue
    $U mirror $MIRROR $S/representations/$C2F/ink.zarr/$l $P/ink.zarr/$l --threads $MTH || die "ink level $l"; done
  touch $POD/state/ink.$S; say "published ink fetched"
fi
BOX=${TBOX:-}; [ "$BOX" = - ] && BOX=; [ -z "$BOX" ] && [ -s $P/box.txt ] && BOX=$(cat $P/box.txt)
if [ -z "$BOX" ]; then   # the corpus prep: scanprep object mask (cell 32) -> box of the object cells +-1 cell, origin on the 512 grid
  say "box: scanprep"
  nice -n 5 $U scanprep $V $K --level $LV --um $UM --out $P/prep --min-inside 0.25 --cell 32 --threads $NCPU --box-out $P/box.txt > $P/prep.log 2>&1 || die "scanprep (log $P/prep.log)"
  BOX=$(cat $P/box.txt)
fi
AX=auto; [ -s $P/axis.json ] && AX=$P/axis.json; [ -n "${AXS:-}" ] && [ -e "${AXS:-/nonexistent}" ] && AX=$AXS
cat > $P.env <<EOV
# $S (scripts/pod/setup.sh $(date '+%F %T'))
S=$S; R=$V; K=$K; UM=$UM; LV=$LV; AXS=$AX; BOX=$BOX; ZR=${TZR:--}; STRIPARGS="${TSTRIP:--}"
INKPUB=$([ -e $POD/state/ink.$S ] && echo $P/ink.zarr)
U=$U; CK=$POD/ckpt/wind96-wk1-best.ckpt; INKCK=$POD/ckpt/ift3-24000.ckpt; NCPU=$NCPU; MEMGB=$MEMGB; POD=$POD
EOV
say "ready: $P.env (box $BOX, axis $AX); CT $(du -sh $V/$K | cut -f1); disk $(df -h $POD | awk 'NR == 2 {print $4}') free"
say "next: bash $POD/ufsm/scripts/pod/unroll.sh $S"
