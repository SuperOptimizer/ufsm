#!/bin/bash
# ufsm latvol end to end on the synthetic spiral fixture (the kernels and the codec: tests/test_latvol.c):
#  1. the CT group encoded as latent cells with the transform on the device and on the CPU: the same shard bytes; latvol-check's error bound;
#     ufsm read of the latvol level = the rebuilt CT (the host decoder), also over HTTP with a chunk cache.
#  2. predict takes the latvol group as its CT (no flag). Three ways of the same window:
#       dev:  the chunk streams decoded on the device, the trunk's input from the stored indices (the default);
#       host: UFSM_PRED_LATVOL=0, the rebuilt CT from z3_read (host decoder), the trunk's input by the CT -> latent kernels;
#       ct:   the original CT.
#     UFSM_PRED_LATVOL=2 (device decode, but the inputs built from the rebuilt CT like host) = host byte for byte: the device decode and
#     the window placement in predict. dev vs host differ only by the second quantiser of the host way (its step follows the window's sd, the store's the level's: within
#     half a step, test_lat), dev vs ct by the store's quantiser and the dropped axes: bounds on the disagreement. --rv nets with the
#     plain head and with the fine stem + sub-cell head (the stem reads the rebuilt CT), and fit fields of a winding net with both.
#     A C 64 store (near-lossless rebuilt CT) feeds the same C 29 checkpoints. The device way reads over HTTP + cache the same bytes.
# Usage: test_latvol.sh <ufsm binary> <make_wind_fixture binary>
set -u
U=$1 F=$2
D=$(mktemp -d /tmp/test_latvol_XXXXXX)
SRV=
fail() { echo "FAIL: $*"; echo "kept $D"; [ -n "$SRV" ] && kill "$SRV" 2> /dev/null; exit 1; }
"$F" "$D/fx" 256 2 1 > "$D/fx.log" 2>&1 || fail "fixture"
T="--task unwrap2 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed ${LT_SEED:-5} --wind4-tile 0 --levels 1 --lat 4"
# shellcheck disable=SC2086
"$U" train "$D/fx/sources.json" --out "$D/fit" --P 128 --widths 32,64,128 --steps 20 $T --lat-fit "$D/basis.lcb" --lat-fit-n 8 > "$D/fit.log" 2>&1 || fail "lat-fit ($D/fit.log)"
L="--lat-basis $D/basis.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3"
# shellcheck disable=SC2086
"$U" train "$D/fx/sources.json" --out "$D/n0" --P 64 --widths 32,64,128 --steps 6 $T $L --rv 1 --rv-iso 64 > "$D/n0.log" 2>&1 || fail "train rv ($D/n0.log)"
# shellcheck disable=SC2086
"$U" train "$D/fx/sources.json" --out "$D/n3" --P 64 --widths 32,64,128 --steps 6 $T $L --rv 2 --rv-iso 64 --lat-guide 3 --fstem 16 --fstem-drop 0.25 > "$D/n3.log" 2>&1 || fail "train rv 2 + stem ($D/n3.log)"
# shellcheck disable=SC2086
"$U" train "$D/fx/sources.json" --out "$D/w3" --P 64 --widths 32,64,128 --steps ${LT_STEPS:-6} $T $L --lat-guide 3 --fstem 16 --fstem-drop 0.25 > "$D/w3.log" 2>&1 || fail "train winding + stem ($D/w3.log)"

# ---- 1. the stores ----
mkdir -p "$D/lv" "$D/lvc" "$D/lv64"
"$U" latvol "$D/basis.lcb" "$D/fx" ct "$D/lv/ct" --levels 9.6 --threads 4 --gpu 0 > "$D/lv.log" 2>&1 || fail "latvol ($D/lv.log)"
"$U" latvol "$D/basis.lcb" "$D/fx" ct "$D/lvc/ct" --levels 9.6 --threads 4 --gpu -1 > "$D/lvc.log" 2>&1 || fail "latvol --gpu -1 ($D/lvc.log)"
"$U" latvol "$D/basis.lcb" "$D/fx" ct "$D/lv64/ct" --levels 9.6 --threads 4 --gpu 0 --c 64 > "$D/lv64.log" 2>&1 || fail "latvol --c 64 ($D/lv64.log)"
n=0; while IFS= read -r f; do n=$((n + 1)); cmp -s "$f" "$D/lvc/ct/9.6/c/${f#"$D/lv/ct/9.6/c/"}" || fail "device and CPU transforms wrote different shards ($f)"; done < <(find "$D/lv/ct/9.6/c" -type f)
[ "$n" -gt 0 ] || fail "no shards written"
cmp -s "$D/lv/ct/9.6/latvol.bin" "$D/lvc/ct/9.6/latvol.bin" || fail "side files differ"
"$U" latvol-check "$D/lv" ct/9.6 "$D/fx" ct/9.6 --n 4 --threads 4 > "$D/chk.log" 2>&1 || fail "latvol-check ($D/chk.log)"
"$U" latvol-check "$D/lv64" ct/9.6 "$D/fx" ct/9.6 --n 4 --threads 4 > "$D/chk64.log" 2>&1 || fail "latvol-check C 64 ($D/chk64.log)"
python3 -I - "$D" <<'EOF' || fail "latvol-check bounds ($D/chk.log, $D/chk64.log)"
import re, sys
d = sys.argv[1]; bad = 0
for f, k in (("chk.log", 0.9), ("chk64.log", 0.3)):   # C 64: the quantiser alone (step / sqrt 12 = 0.289 step); C 29: + the dropped axes
    t = open(f"{d}/{f}").read(); m = re.search(r"rms error ([0-9.]+) grey levels", t); s = re.search(r"step_raw ([0-9.]+)", t)
    ok = m and s and float(m.group(1)) <= k * float(s.group(1))
    print(f"{f}: rebuilt CT rms error {m.group(1) if m else '?'} grey levels (<= {k} x the step {s.group(1) if s else '?'}): {'ok' if ok else 'FAIL'}"); bad += not ok
sys.exit(1 if bad else 0)
EOF
"$U" read "$D/lv" ct/9.6 0 0 0 256 256 256 "$D/lv.raw" > /dev/null 2>&1 || fail "read of the latvol level"
"$U" read "$D/fx" ct/9.6 0 0 0 256 256 256 "$D/ct.raw" > /dev/null 2>&1 || fail "read of the CT"
# the voxel reader's GPU path (UFSM_LATVOL_GPU: the box's streams decoded and rebuilt on the device below z3_read) = the host decoder,
# for the whole level, a box off the cell grid and a box over the array's end
UFSM_LATVOL_GPU=0 "$U" read "$D/lv" ct/9.6 0 0 0 256 256 256 "$D/lvg.raw" > "$D/lvg.log" 2>&1 || fail "GPU read of the latvol level"
grep -q "decoded on GPU" "$D/lvg.log" || fail "the GPU voxel reader did not run ($D/lvg.log)"
cmp -s "$D/lv.raw" "$D/lvg.raw" || fail "the GPU voxel reader differs from the host decoder"
for b in "3 17 30 131 97 150" "130 100 60 200 180 250"; do
    # shellcheck disable=SC2086
    "$U" read "$D/lv" ct/9.6 $b "$D/bh.raw" > /dev/null 2>&1 || fail "read $b"
    # shellcheck disable=SC2086
    UFSM_LATVOL_GPU=0 "$U" read "$D/lv" ct/9.6 $b "$D/bg.raw" > "$D/bg.log" 2>&1 || fail "GPU read $b"
    grep -q "decoded on GPU" "$D/bg.log" || fail "the GPU voxel reader did not run (box $b)"
    cmp -s "$D/bh.raw" "$D/bg.raw" || fail "the GPU voxel reader differs from the host decoder (box $b)"
done
# a range-serving HTTP root over the store (suffix ranges for the shard indices)
python3 -I - "$D/lv" "$D/port" <<'EOF' &
import http.server, os, re, sys
root, portf = sys.argv[1], sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_HEAD(self): self.serve(False)
    def do_GET(self): self.serve(True)
    def serve(self, body):
        p = os.path.normpath(os.path.join(root, self.path.lstrip("/")))
        if not p.startswith(root) or not os.path.isfile(p):
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
        n = os.path.getsize(p); r = self.headers.get("Range"); lo, hi, code = 0, n - 1, 200
        if r:
            m = re.match(r"bytes=(\d*)-(\d*)$", r.strip())
            if m.group(1) == "": lo = max(0, n - int(m.group(2)))
            else:
                lo = int(m.group(1)); hi = min(n - 1, int(m.group(2))) if m.group(2) else n - 1
            code = 206
        self.send_response(code); self.send_header("Content-Length", str(hi - lo + 1)); self.send_header("Accept-Ranges", "bytes")
        if code == 206: self.send_header("Content-Range", f"bytes {lo}-{hi}/{n}")
        self.end_headers()
        if body:
            with open(p, "rb") as f: f.seek(lo); self.wfile.write(f.read(hi - lo + 1))
s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
open(portf + ".tmp", "w").write(str(s.server_address[1])); os.rename(portf + ".tmp", portf)
s.serve_forever()
EOF
SRV=$!
for _ in $(seq 100); do [ -s "$D/port" ] && break; sleep 0.1; done
[ -s "$D/port" ] || fail "the HTTP server did not start"
H="http://127.0.0.1:$(cat "$D/port")"
"$U" read "$H" ct/9.6 0 0 0 256 256 256 "$D/lvh.raw" --cache "$D/cache" > "$D/lvh.log" 2>&1 || fail "read over HTTP ($D/lvh.log)"
cmp -s "$D/lv.raw" "$D/lvh.raw" || fail "the latvol level over HTTP differs from the local read"
[ -n "$(find "$D/cache" -type f | head -1)" ] || fail "no chunk cached"

# ---- 2. predict ----
P="--um 9.6 --axis $D/fx/axis.json --levels 1 --q 0"
run() {   # TAG NET ROOT [args]: predict --channel rv, then the box as raw bytes
    local tag=$1 net=$2 root=$3; shift 3
    # shellcheck disable=SC2086
    "$U" predict "$D/$net/last.ckpt" "$root" ct "$D/p_$tag" $P --channel rv "$@" > "$D/p_$tag.log" 2>&1 || return 1
    "$U" read "$D/p_$tag" 9.6 0 0 0 256 256 256 "$D/p_$tag.raw" > /dev/null 2>&1
}
for net in n0 n3; do
    run ${net}ct $net "$D/fx" --window 128 --halo 16 || fail "predict $net CT ($D/p_${net}ct.log)"
    run ${net}dev $net "$D/lv" --window 128 --halo 16 || fail "predict $net latvol ($D/p_${net}dev.log)"
    UFSM_PRED_LATVOL=0 run ${net}host $net "$D/lv" --window 128 --halo 16 || fail "predict $net latvol, host decode ($D/p_${net}host.log)"
    run ${net}d64 $net "$D/lv64" --window 128 --halo 16 || fail "predict $net latvol C 64 ($D/p_${net}d64.log)"
    grep -q "latvol CT (C 29.*the trunk's input from the stored indices" "$D/p_${net}dev.log" || fail "predict $net did not take the cells ($D/p_${net}dev.log)"
    grep -q "latvol CT (C 64.*the trunk's input from the stored indices" "$D/p_${net}d64.log" || fail "predict $net did not take the C 64 cells"
    grep -q "latvol CT" "$D/p_${net}host.log" && fail "UFSM_PRED_LATVOL=0 still took the device path"
done
# the device decode + rebuilt CT alone (inputs from the rebuilt CT as the host way builds them): the host way's bytes
UFSM_PRED_LATVOL=2 run n0chk n0 "$D/lv" --window 128 --halo 16 || fail "predict latvol, check mode ($D/p_n0chk.log)"
cmp -s "$D/p_n0chk.raw" "$D/p_n0host.raw" || fail "the device-decoded rebuilt CT gives other bytes than the host-decoded one"
# a window whose streams exceed the slot comes as host-decoded voxels (here: every window): the host way's bytes again
UFSM_PRED_LATVOL_CAP=4096 run n0fb n0 "$D/lv" --window 128 --halo 16 || fail "predict latvol, voxel fallback ($D/p_n0fb.log)"
cmp -s "$D/p_n0fb.raw" "$D/p_n0host.raw" || fail "the voxel fallback gives other bytes than the host-decoded CT"
grep -q "[1-9][0-9]* windows as host-decoded voxels" "$D/p_n0fb.log" || fail "the voxel fallback did not run"
UFSM_PRED_LATVOL=2 run n3chk n3 "$D/lv" --window 128 --halo 16 || fail "predict latvol stem net, check mode"
cmp -s "$D/p_n3chk.raw" "$D/p_n3host.raw" || fail "stem net: the device-decoded rebuilt CT gives other bytes than the host-decoded one"
run n0big n0 "$D/lv" --window 288 --halo 16 --shard 128 || fail "predict latvol, one window per 2^3 shards ($D/p_n0big.log)"
run n0bct n0 "$D/fx" --window 288 --halo 16 --shard 128 || fail "predict CT, one window per 2^3 shards"
run n0off n0 "$D/lv" --window 128 --halo 18 || fail "predict latvol off the cell grid ($D/p_n0off.log)"
grep -q "off the 4-voxel cell grid" "$D/p_n0off.log" || fail "no off-grid fallback message ($D/p_n0off.log)"
run n0http n0 "$H" --window 128 --halo 16 --cache "$D/cache2" || fail "predict latvol over HTTP ($D/p_n0http.log)"
cmp -s "$D/p_n0dev.raw" "$D/p_n0http.raw" || fail "predict over HTTP + cache differs from the local store"
kill "$SRV" 2> /dev/null; SRV=
python3 -I - "$D" <<'EOF' || fail "outputs ($D)"
import sys, numpy as np
d = sys.argv[1]
def ld(t): return np.fromfile(f"{d}/p_{t}.raw", np.uint8).astype(int).reshape(256, 256, 256)
bad = 0
for a, b, agree_min, md_max in (("n0dev", "n0host", 0.93, 12), ("n0ct", "n0dev", 0.95, 16), ("n0ct", "n0d64", 0.95, 16), ("n0bct", "n0big", 0.85, 16),
                                ("n3dev", "n3host", 0.90, 12), ("n3ct", "n3dev", 0.90, 16), ("n3ct", "n3d64", 0.90, 16)):
    x, y = ld(a), ld(b)
    m = (x > 0) & (y > 0); nzx, nzy = (x > 0).mean(), (y > 0).mean()
    agree = ((x[m] > 127) == (y[m] > 127)).mean(); md = np.abs(x - y)[m].mean()
    ok = agree >= agree_min and md <= md_max and abs(nzx - nzy) < 0.03 and m.mean() > 0.5
    print(f"{a} / {b}: written {nzx:.4f} / {nzy:.4f}, side agreement {agree:.4f} (>= {agree_min}), mean |d| {md:.2f} (<= {md_max}): {'ok' if ok else 'FAIL'}")
    bad += not ok
sys.exit(1 if bad else 0)
EOF
# fit fields of the winding net with the stem + sub-cell head
for src in ct lv; do
    r="$D/fx"; [ $src = lv ] && r="$D/lv"
    "$U" predict "$D/w3/last.ckpt" "$r" ct "$D/fo_$src" --fit-zarr "$D/f_$src" --um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 1 --threads 4 --fit-h 4 \
        --axis "$D/fx/axis.json" > "$D/f_$src.log" 2>&1 || fail "predict --fit-zarr $src ($D/f_$src.log)"
    for a in pc ps sp t0 t1 t2 t3 t4 t5 w dq dx dy dz; do [ -d "$D/f_$src/$a" ] && "$U" read "$D/f_$src" $a 0 0 0 64 64 64 "$D/f_$src.$a.raw" > /dev/null 2>&1; done
done
python3 -I - "$D" <<'EOF2' || fail "fit fields ($D)"
# the correlation of CT-path and store-path values per group of planes (test_latents.sh's measure and bounds)
import sys, os, numpy as np
d = sys.argv[1]; bad = 0
groups = {"phase": (["pc", "ps"], .70), "normal": (["t0", "t1", "t2", "t3", "t4", "t5"], .70), "spacing": (["sp"], .60), "material": (["w"], .90), "w4": (["dq"], .65), "w4dir": (["dx", "dy", "dz"], .45)}
for name, (planes, m) in groups.items():
    xs, ys = [], []
    for a in planes:
        f = f"{d}/f_ct.{a}.raw"
        if not os.path.exists(f): continue
        x = np.fromfile(f, np.uint8).astype(float); y = np.fromfile(f"{d}/f_lv.{a}.raw", np.uint8).astype(float)
        mu, sd = x.mean(), x.std() + 1e-6; xs.append((x - mu) / sd); ys.append((y - mu) / sd)
    if not xs: continue
    c = np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1]
    ok = c >= m; bad += not ok
    print(f"fit fields {name:8s}: corr {c:.4f} (>= {m}){'' if ok else ': FAIL'}")
sys.exit(1 if bad else 0)
EOF2
rm -rf "$D"
echo "test_latvol.sh: ok"
