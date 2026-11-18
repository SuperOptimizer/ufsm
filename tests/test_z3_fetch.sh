#!/bin/bash
# ufsm z3-fetch (2026-11-17, the unroll's m7 source prefetch beside the fields): a volcomp array served over a local range-serving HTTP root;
# z3-fetch of a box fills the chunk cache so that ufsm read of that box over the same root + cache requests no chunk data and equals the
# local read byte for byte (a box outside it does request chunks); a second z3-fetch fetches nothing; --scale maps the box into the array's voxels (exactly the box's present chunks fetched).
set -u
U=${1:?ufsm binary}; U=$(readlink -f "$U")
D=$(mktemp -d /tmp/test_z3_fetch_XXXXXX); SRV=
fail() { echo "test_z3_fetch: FAIL: $*" >&2; [ -n "$SRV" ] && kill $SRV 2> /dev/null; echo "(kept $D)" >&2; exit 1; }
python3 -I - "$D/in.raw" <<'EOF' || fail "fixture"
import sys, numpy as np
z, y, x = np.mgrid[0:256, 0:256, 0:256]
v = ((x * 7 + y * 3 + z * 5) % 200 + 20 * np.sin(x * 0.1) * np.cos(y * 0.07)).astype(np.uint8)
v[:, :, :128] = 5   # (the x 0 chunks: tiny once encoded, so the wanted x 1 chunks between them coalesce)
v.tofile(sys.argv[1])
EOF
mkdir -p "$D/s/a"
"$U" z3-put "$D/s/a/0" "$D/in.raw" --shape 384,256,256 --z0 0 --nz 256 --shard 256 > "$D/put.log" 2>&1 || fail "z3-put ($D/put.log)"
python3 -I - "$D/s" "$D/port" "$D/req.log" <<'EOF' &
import http.server, os, re, sys
root, portf, reqf = sys.argv[1], sys.argv[2], sys.argv[3]
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_HEAD(self): self.serve(False)
    def do_GET(self): self.serve(True)
    def serve(self, body):
        with open(reqf, "a") as f: f.write(self.path + " " + str(self.headers.get("Range")) + "\n")
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
# the box z 10..140, y 0..100, x 64..124 at half resolution (--scale 2: array voxels z 20..280, y 0..200, x 128..248): 128^3 chunks
# z 0..2 x y 0..1 x x 1 = 6 in the box, 4 present (z 256.. is an absent shard), all 4 in shard 0 (256^3 shards): coalesced into fewer
# than 4 data requests (their neighbours in file order, the x 0 chunks, are small: within UFSM_Z3_GAP)
"$U" z3-fetch "$H" a/0 --cache "$D/c" --box 10,0,64,130,100,60 --scale 2 --margin 0 --threads 4 > "$D/f1.log" 2>&1 || fail "z3-fetch ($D/f1.log)"
nd=$(grep '/c/0/0/0 ' "$D/req.log" | grep -vc 'bytes=-'); [ "$nd" -ge 1 ] && [ "$nd" -lt 4 ] || fail "z3-fetch: $nd data requests for the 4 chunks of shard 0 (expected 1..3: coalesced): $(cat "$D/req.log" | tr '\n' ' ')"
grep -q "fetched, 0 failed" "$D/f1.log" || fail "z3-fetch: $(cat "$D/f1.log")"
grep -q "6 chunks in the box (3 x 2 x 1), 4 fetched" "$D/f1.log" || fail "z3-fetch: expected 6 chunks in the box, 4 fetched: $(cat "$D/f1.log")"; n1=4
"$U" z3-fetch "$H" a/0 --cache "$D/c" --box 10,0,64,130,100,60 --scale 2 --margin 0 --threads 4 > "$D/f2.log" 2>&1 || fail "z3-fetch again ($D/f2.log)"
grep -q " 0 fetched" "$D/f2.log" || fail "the second z3-fetch fetched again: $(cat "$D/f2.log")"
# the fetched box read over the same root + cache: no chunk data requested (only metadata), and it equals the local array (z 20..256 holds
# data, 256..280 an absent shard)
: > "$D/req.log"
"$U" read "$H" a/0 20 0 128 260 200 128 "$D/net.raw" --cache "$D/c" > "$D/ro.log" 2>&1 || fail "read of the fetched box ($D/ro.log)"
"$U" read "$D/s" a/0 20 0 128 260 200 128 "$D/loc.raw" > "$D/rl.log" 2>&1 || fail "local read ($D/rl.log)"
cmp -s "$D/loc.raw" "$D/net.raw" || fail "the box read through the cache differs from the local array"
nc=$(grep -c '/c/' "$D/req.log"); [ "$nc" = 0 ] || fail "the read of the fetched box still requested $nc chunk objects: $(grep /c/ "$D/req.log" | head -3 | tr '\n' ' ')"
# a box beyond the fetched one does request chunks (the check above can fail)
: > "$D/req.log"
"$U" read "$H" a/0 0 0 0 16 16 16 "$D/out.raw" --cache "$D/c" > "$D/rx.log" 2>&1 || fail "read outside ($D/rx.log)"
[ "$(grep -c '/c/' "$D/req.log")" -gt 0 ] || fail "a box outside the fetch requested no chunk"
kill $SRV; wait $SRV 2> /dev/null; SRV=
echo "test_z3_fetch: $n1 chunks fetched in $nd requests, the fetched box reads from the cache alone and equals the local array; refetch 0"
rm -rf -- "$D"
echo "test_z3_fetch: OK"
