#!/bin/sh
# Reads a local teacher region (volcomp+zstd sharded) and, if UFSM_NET=1, a shard of a published
# prob pyramid over HTTPS with the disk cache, and checks the two code paths agree with python/zarr
# when that is available. Exits nonzero on any failure.
# Off the desk (no teacher region) and without UFSM_NET=1 nothing is read: it prints SKIP, not "zarr ok" (mini audit suite.md 4).
# Outputs go to a per-run dir ($TMPDIR in the suite), so concurrent suites do not share them.
set -e
U=${B:-build}/ufsm; [ -x "$U" ] || U=./build/ufsm   # the suite's $B, else the release dir
T=$(mktemp -d "${TMPDIR:-/tmp}/ufsm-test-zarr.XXXXXX")
CACHE=${UFSM_CACHE:-$T/cache}
LOCAL_ROOT=/vesuvius/usrm2/teacher_regions
LOCAL_KEY=recto/region_10240_10240_10240.zarr
RAN=0
if [ -d "$LOCAL_ROOT/$LOCAL_KEY" ]; then
    RAN=1
    $U info "$LOCAL_ROOT" "$LOCAL_KEY"
    $U read "$LOCAL_ROOT" "$LOCAL_KEY" 100 200 300 64 96 128 $T/local.raw
    [ "$(stat -c %s $T/local.raw)" = "786432" ]
    # cross-check against the python zarr + volcomp_zarr reader when installed
    PY=${UFSM_PY:-$HOME/usrm2/.venv/bin/python}; export VOLCOMP_LIB=${VOLCOMP_LIB:-$HOME/volume-compressor/build/release/libvolcomp.so}
    # strict-math build (-DUFSM_STRICT_MATH=ON, $BUILD/strict_math = 1): byte-identical; the default fast-math build decodes the lossy
    # chunks with a contracted / reassociated IDCT: +-1 on at most 1e-4 of the voxels (the codec's cross-build allowance)
    STRICT=$(cat "$(dirname "$B")/strict_math" 2>/dev/null || echo 0)
    if $PY -c "import zarr, volcomp_zarr" 2>/dev/null; then
        $PY - <<EOF
import math, numpy as np, zarr, volcomp_zarr
z = zarr.open("$LOCAL_ROOT/$LOCAL_KEY", mode="r")
ref = np.asarray(z[100:164, 200:296, 300:428]).ravel()
got = np.fromfile("$T/local.raw", np.uint8)
assert got.shape == ref.shape, "shape mismatch vs python reader"
if "$STRICT" == "1":
    assert np.array_equal(got, ref), "mismatch vs python reader"
else:
    d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    assert d.max() <= 1 and (d > 0).sum() <= math.ceil(1e-4 * d.size), f"mismatch vs python reader: {(d > 0).sum()} differ, worst {d.max()}"
print("local read matches python reader")
EOF
    fi
fi
if [ "${UFSM_NET:-0}" = "1" ]; then
    RAN=1
    ROOT=https://dl.ash2txt.org/community-uploads/forrest/volcomp
    KEY=PHerc0800/representations/predictions/surfaces/20250521135224-surface-20260925170000-surface-m7-L0-prob.zarr
    $U info "$ROOT" "$KEY" --cache "$CACHE"
    $U read "$ROOT" "$KEY/138.24" 700 250 250 64 64 64 $T/net.raw --cache "$CACHE"
    # second read must come from the cache: same bytes
    $U read "$ROOT" "$KEY/138.24" 700 250 250 64 64 64 $T/net2.raw --cache "$CACHE"
    cmp $T/net.raw $T/net2.raw
fi
rm -rf "$T"
[ $RAN = 1 ] || { echo "SKIP: test_zarr: no local teacher region ($LOCAL_ROOT/$LOCAL_KEY) and UFSM_NET not 1: nothing read"; exit 0; }
echo "zarr ok"
