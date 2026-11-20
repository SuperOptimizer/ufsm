#!/usr/bin/env bash
# scripts/test/gpubox_setup.sh HOST [ROOT]: make another machine a suite box for scripts/test/desk_suite.sh -H HOST -a 86 -R ROOT
# (forlindesk's 3090s, the laptop's 4060): checks the tools, installs mcc at this clone's cmake/mcc.version into the host's ~/.local/mcc
# (scripts/toolchain/install_mcc.sh, ~1 min), and a python venv ROOT/pyenv with the test references' modules (numpy, tifffile, scipy,
# pillow; desk_suite puts it on PATH). Idempotent. ROOT default /home/forrest/suite (no /vesuvius needed).
# Then: SUITE_ARGS="-J 8" scripts/test/desk_suite.sh -H HOST -a 86 -R ROOT -m 16 -l 24 BRANCH   (the first GPU run JITs every kernel
# for that GPU model: ~1.5-2 CPU-h, mcuda 11-20; -J 8 compiles the recorded kernels 8 at a time after the first pass has recorded them)
set -u
H=${1:?HOST}; ROOT=${2:-/home/forrest/suite}
here=$(cd "$(dirname "$0")/../.." && pwd)
ssh -o ConnectTimeout=10 "$H" true || { echo "gpubox_setup: $H unreachable"; exit 2; }
ssh "$H" 'for t in git cmake ninja cc python3 nvidia-smi flock timeout; do command -v $t > /dev/null || echo "MISSING $t"; done;
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader; nproc; free -g | awk "NR == 2 {print \$2 \" GB\"}"'
ssh "$H" "mkdir -p $ROOT/setup/scripts/toolchain $ROOT/setup/cmake" || exit 2
scp -q "$here/scripts/toolchain/install_mcc.sh" "$H:$ROOT/setup/scripts/toolchain/" || exit 2
scp -q "$here/cmake/mcc.version" "$H:$ROOT/setup/cmake/" || exit 2
ssh "$H" "bash $ROOT/setup/scripts/toolchain/install_mcc.sh" || { echo "gpubox_setup: mcc install failed"; exit 1; }
ssh "$H" "[ -x $ROOT/pyenv/bin/python3 ] || python3 -m venv $ROOT/pyenv; $ROOT/pyenv/bin/pip install -q numpy tifffile scipy pillow &&
  $ROOT/pyenv/bin/python3 -c 'import numpy, tifffile, scipy, PIL; print(\"pyenv ok\")'" || { echo "gpubox_setup: venv failed"; exit 1; }
echo "gpubox_setup: $H ready: scripts/test/desk_suite.sh -H $H -a 86 -R $ROOT BRANCH"
