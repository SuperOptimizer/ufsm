#!/usr/bin/env bash
# scripts/pod/setup.sh [SCROLL]: make a rented GPU pod ready for a whole-scroll unroll. Idempotent: every step checks
# its result and skips the work when it is already done. Re-run it after a reboot, a failed download or a master update.
#   1. limits: the REAL cgroup CPU / RAM limits. nproc and free show the host: 256 CPUs and 1 TB on a pod whose cgroup allows
#      13.6 CPUs and 62 GB. Also the GPU / MIG slice, CUDA and disk.
#      -> $POD/limits.txt; then `ufsm sysinfo` once the binary exists.
#   2. deps: apt packages (root only), skipped when they are all present:
#      gcc-14 (C23), libcurl / zstd / blosc / zlib / ssl / jpeg dev, jq, curl, rsync, time.
#   3. clone or fast-forward ufsm ($BRANCH, default master) into $POD/ufsm, then build it.
#      The Makefile finds gcc-14, /usr/local/cuda-*, -ccbin and -lstdc++ by itself.
#      make -j <cgroup CPUs>. ARCH=86 is picked for sm_80/86/89 GPUs: software MX, since the fp4 / fp8 kernels need sm_120a.
#   4. smoke test: ufsm sysinfo, test_sysinfo (cgroup parsing), test_mx_sw (GPU kernels).
#   5. checkpoints from the public mirror, resumable, size-checked: wind96-wk1-best (fit fields) and ift3-24000 (ink) -> $POD/ckpt.
#   6. SCROLL given: its CT from the mirror. `ufsm mirror` copies shard files resumably, about 40 MB/s on a pod; desk -> pod
#      rsync manages only about 1.6 MB/s, so nothing comes via the desk. Levels LV..LV+3 of the scan in scripts/unroll/scrolls.tsv
#      or scripts/corpus/list.tsv -> $POD/vol/<S>/<scan>.zarr/<L>.
#      Also any published ufsm-c2f-20261028 products: box.txt, axis.json, ink.zarr (ink only where uploaded). The box is
#      otherwise made on the pod (scanprep --box-out), the axis by unroll (`ufsm axis`), the ink by unroll.
#      Writes $POD/scrolls/<S>.env for scripts/pod/unroll.sh.
# Env: POD (default /root/pod as root, else ~/pod), BRANCH (master), REPO (https://github.com/SuperOptimizer/ufsm), MIRROR,
#   NOAPT=1 (skip apt), NOBUILD=1, NOSMOKE=1, LEVELS ("L ..." override), MTH (mirror threads, 12; the mirror answers 429 past 64
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
    read -r q per < /sys/fs/cgroup$p/cpu.max 2>/dev/null && [ "$q" != max ] && [ -n "$q" ] && {
      local c=$(( q / per )); [ $c -lt 1 ] && c=1; { [ -z "$best" ] || [ $c -lt $best ]; } && best=$c; }
    [ -z "$p" ] && break; p=${p%/*}
  done
  [ -n "$best" ] && [ $best -lt $n ] && n=$best; echo $n; }
NCPU=$(ncpu_cg); MEMMAX=$(cg_min memory.max)
MEMGB=$(( ${MEMMAX:-$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)} >> 30 ))
{
  echo "date $(date -u '+%F %T')Z host $(hostname)"
  echo "nproc $(nproc) (host), cgroup $(cg_path) cpu.max: $(cat /sys/fs/cgroup$(cg_path)/cpu.max /sys/fs/cgroup/cpu.max 2>/dev/null | tr '\n' ' ') -> NCPU $NCPU"
  echo "memory.max ${MEMMAX:-none} -> MEMGB $MEMGB (free says $(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo) GB)"
  nvidia-smi -L 2>&1 | sed 's/^/gpu /'
  nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>&1 | sed 's/^/gpu driver, cc: /'
  ls -d /usr/local/cuda*/ 2>/dev/null | tr '\n' ' ' | sed 's/^/cuda installs: /'; echo
  df -h $POD | tail -1 | sed 's/^/disk /'
} > $POD/limits.txt
cat $POD/limits.txt
say "pod $POD: $NCPU CPUs, $MEMGB GB (cgroup)"

# --- 2. deps ---
PKGS="build-essential gcc-14 g++-14 make git pkg-config ca-certificates curl jq rsync time libcurl4-openssl-dev libzstd-dev libblosc-dev zlib1g-dev libssl-dev libjpeg-dev"
if [ "${NOAPT:-0}" != 1 ] && command -v apt-get > /dev/null; then
  miss=$(for p in $PKGS; do dpkg -s $p > /dev/null 2>&1 || echo $p; done | tr '\n' ' ')
  if [ -n "${miss// /}" ]; then
    [ "$(id -u)" = 0 ] || die "missing packages ($miss) and not root: install them or NOAPT=1"
    say "apt: $miss"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $miss > /dev/null || die "apt-get install"
  else say "deps present"; fi
fi
for t in git make curl jq pkg-config; do command -v $t > /dev/null || die "$t missing"; done

# --- 3. clone + build ---
U=$POD/ufsm/build/ufsm
if [ ! -d $POD/ufsm/.git ]; then git clone -q $REPO $POD/ufsm || die "git clone"; fi
( cd $POD/ufsm && git fetch -q origin && git checkout -q $BRANCH && { git diff --quiet && git merge -q --ff-only origin/$BRANCH || echo "setup: local changes in $POD/ufsm: not updated"; } ) || die "git update"
say "ufsm $(git -C $POD/ufsm log --oneline -1)"
CUDAP=$(ls -d /usr/local/cuda/bin /usr/local/cuda-*/bin 2>/dev/null | sort -V | tail -1); [ -n "$CUDAP" ] && export PATH=$PATH:$CUDAP
if [ -z "${ARCH:-}" ]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  case "$cc" in 8*) ARCH=86;; *) ARCH=120;; esac
fi
BD=build; [ "$ARCH" = 86 ] && BD=build86 && U=$POD/ufsm/build86/ufsm
if [ "${NOBUILD:-0}" != 1 ]; then
  say "make -j$NCPU ARCH=$ARCH"
  ( cd $POD/ufsm && nice -n 5 make -j$NCPU ARCH=$ARCH $BD/ufsm > $POD/build.log 2>&1 ) || { tail -30 $POD/build.log; die "build (log $POD/build.log)"; }
fi
[ -x $U ] || die "no $U"
$U sysinfo --disk $POD | tee $POD/sysinfo.txt

# --- 4. smoke test ---
if [ "${NOSMOKE:-0}" != 1 ] && [ ! -e $POD/state/smoke.$(git -C $POD/ufsm rev-parse --short HEAD) ]; then
  ( cd $POD/ufsm && make -j$NCPU ARCH=$ARCH VERIFY=0 $BD/test_sysinfo $BD/test_mx_sw >> $POD/build.log 2>&1 &&
    ./$BD/test_sysinfo && ./$BD/test_mx_sw > $POD/smoke.log 2>&1 ) || { tail -20 $POD/smoke.log 2>/dev/null; die "smoke test"; }
  touch $POD/state/smoke.$(git -C $POD/ufsm rev-parse --short HEAD); say "smoke test passed"
fi

# --- 5. checkpoints ---
fetch() {   # fetch URL OUT: resumable, complete when the size matches Content-Length
  local want have; want=$(curl -sfIL "$1" | awk 'tolower($1) == "content-length:" {n = $2} END {print n + 0}' | tr -d '\r')
  have=$(stat -c %s "$2" 2>/dev/null || echo 0)
  [ "$want" -gt 0 ] && [ "$have" = "$want" ] && return 0
  curl -sfL --retry 5 -C - -o "$2" "$1" || curl -sfL --retry 5 -o "$2" "$1" || return 1
  have=$(stat -c %s "$2"); [ "$want" = 0 ] || [ "$have" = "$want" ] || { echo "fetch: $2 has $have of $want bytes"; return 1; }; }
for c in wind96-wk1-best ift3-24000; do fetch $BASE/ufsm/checkpoints/$c/$c.ckpt $POD/ckpt/$c.ckpt || die "checkpoint $c"; done
say "checkpoints: $(ls -l $POD/ckpt | awk 'NR > 1 {printf "%s %.0f MB  ", $9, $5 / 1e6}')"

# --- 6. the scroll ---
[ -n "$S" ] || { say "done (no SCROLL: CT not fetched)"; exit 0; }
row=; for t in $POD/ufsm/scripts/unroll/scrolls.tsv $POD/ufsm/scripts/corpus/list.tsv; do
  [ -e $t ] && row=$(awk -F'\t' -v s=$S '$1 == s' $t) && [ -n "$row" ] && break; done
[ -n "$row" ] || die "$S is in neither scripts/unroll/scrolls.tsv nor scripts/corpus/list.tsv"
IFS=$'\t' read -r _ _ K UM LV AXS TBOX TZR TSTRIP <<< "$row"
MK=$(dirname $K)/volumes/$(basename $K); V=$POD/vol
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
