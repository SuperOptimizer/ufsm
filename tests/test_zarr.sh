#!/bin/sh
# Reads a local teacher region (volcomp+zstd sharded) and, if UFSM_NET=1, a shard of a published
# prob pyramid over HTTPS with the disk cache, and checks the two code paths agree with python/zarr
# when that is available. Exits nonzero on any failure.
set -e
B=./build/ufsm
CACHE=${UFSM_CACHE:-/tmp/ufsm-test-cache}
LOCAL_ROOT=/vesuvius/usrm2/teacher_regions
LOCAL_KEY=recto/region_10240_10240_10240.zarr
if [ -d "$LOCAL_ROOT/$LOCAL_KEY" ]; then
    $B info "$LOCAL_ROOT" "$LOCAL_KEY"
    $B read "$LOCAL_ROOT" "$LOCAL_KEY" 100 200 300 64 96 128 /tmp/ufsm-test-local.raw
    [ "$(stat -c %s /tmp/ufsm-test-local.raw)" = "786432" ]
    # cross-check against the python zarr + volcomp_zarr reader when installed
    PY=${UFSM_PY:-$HOME/usrm2/.venv/bin/python}; export VOLCOMP_LIB=${VOLCOMP_LIB:-$HOME/volume-compressor/build/release/libvolcomp.so}
    if $PY -c "import zarr, volcomp_zarr" 2>/dev/null; then
        $PY - <<EOF
import numpy as np, zarr, volcomp_zarr
z = zarr.open("$LOCAL_ROOT/$LOCAL_KEY", mode="r")
ref = np.asarray(z[100:164, 200:296, 300:428]).ravel()
got = np.fromfile("/tmp/ufsm-test-local.raw", np.uint8)
assert got.shape == ref.shape and np.array_equal(got, ref), "mismatch vs python reader"
print("local read matches python reader")
EOF
    fi
fi
if [ "${UFSM_NET:-0}" = "1" ]; then
    ROOT=https://dl.ash2txt.org/community-uploads/forrest/volcomp
    KEY=PHerc0800/representations/predictions/surfaces/20250521135224-surface-20260925170000-surface-m7-L0-prob.zarr
    $B info "$ROOT" "$KEY" --cache "$CACHE"
    $B read "$ROOT" "$KEY/138.24" 700 250 250 64 64 64 /tmp/ufsm-test-net.raw --cache "$CACHE"
    # second read must come from the cache: same bytes
    $B read "$ROOT" "$KEY/138.24" 700 250 250 64 64 64 /tmp/ufsm-test-net2.raw --cache "$CACHE"
    cmp /tmp/ufsm-test-net.raw /tmp/ufsm-test-net2.raw
fi
echo "zarr ok"
