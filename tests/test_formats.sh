#!/bin/sh
# Runs build/test_formats against references generated with tifffile/numcodecs when those are available.
# Inputs: a Kaggle label+image pair and a tifxyz x.tif are fetched from the HF bucket if UFSM_NET=1 and
# ~/huggingfacetoken exists; otherwise only the writer round-trip runs.
set -e
D=${UFSM_TEST_DIR:-/tmp/ufsm-test-formats}
mkdir -p "$D"
PY=${UFSM_PY:-$HOME/usrm2/.venv/bin/python}
if [ "${UFSM_NET:-0}" = "1" ] && [ -f "$HOME/huggingfacetoken" ] && $PY -c "import tifffile, numcodecs" 2>/dev/null; then
  T=$(cat "$HOME/huggingfacetoken")
  R=https://huggingface.co/buckets/scrollprize/datasets/resolve
  [ -f "$D/lab1.tif" ] || curl -sL -H "Authorization: Bearer $T" "$R/surfaces/kaggle/labels/sample_00001.tif" -o "$D/lab1.tif"
  [ -f "$D/img1.tif" ] || curl -sL -H "Authorization: Bearer $T" "$R/surfaces/kaggle/images/sample_00001.tif" -o "$D/img1.tif"
  [ -f "$D/c_blosc.bin" ] || curl -sL -H "Authorization: Bearer $T" "$R/surfaces/2um_032726/0500p2_5217.zarr/0/100.61.77" -o "$D/c_blosc.bin"
  X=$(find /vesuvius/usrm/tifxyz -name x.tif 2>/dev/null | head -1); [ -n "$X" ] && cp "$X" "$D/x.tif"
  $PY - "$D" <<'PYEOF'
import sys, os, numpy as np, tifffile, numcodecs
d = sys.argv[1]
for f in ("lab1", "img1", "x"):
    if os.path.exists(f"{d}/{f}.tif"): tifffile.imread(f"{d}/{f}.tif").tofile(f"{d}/{f}.raw")
b = open(f"{d}/c_blosc.bin", "rb").read()
np.frombuffer(numcodecs.Blosc().decode(b), np.uint8).tofile(f"{d}/c_blosc.raw")
PYEOF
fi
rm -rf "$D/z3w_test.zarr"
UFSM_NET=${UFSM_NET:-0} ./build/test_formats "$D"
