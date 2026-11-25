#!/bin/bash
# ufsm prefetch on sharded arrays (2026-11-25, the pod data wait): the marked chunks of each shard come in coalesced range requests
# (z3_prefetch_shard_sel) instead of one request per chunk. Fixture: a ct_only source over a local range-serving HTTP root; level 0 a
# volcomp array (z3-put: 128^3 chunks in one 256^3 shard), levels 1..5 plain uint8 arrays, papyrus in z < 128 only. Checks: the shard path
# caches exactly the chunk files the old per-chunk path (UFSM_PREFETCH_CHUNKS=1) does, with fewer level-0 data requests than chunks;
# its summary line reports the tally and coverage 1.0000; a second run fetches 0.00 GB with every stored chunk already cached.
set -u
U=${1:?ufsm binary}; U=$(readlink -f "$U")
D=$(mktemp -d /tmp/test_prefetch_shards_XXXXXX); SRV=
fail() { echo "test_prefetch_shards: FAIL: $*" >&2; [ -n "$SRV" ] && kill $SRV 2> /dev/null; echo "(kept $D)" >&2; exit 1; }
python3 -I - "$D" <<'EOF' || fail "fixture"
import sys, os, json, numpy as np
d = sys.argv[1]; z, y, x = np.mgrid[0:256, 0:256, 0:256]
v = ((x * 7 + y * 3 + z * 5) % 50 + 150).astype(np.uint8); v[128:] = 0   # (papyrus >= iso 64 in z < 128 only)
v.tofile(d + "/l0.raw"); ds = []
for l in range(6):
    s = 2.4 * 2 ** l; ds.append({"path": str(l), "coordinateTransformations": [{"scale": [s, s, s], "type": "scale"}]})
    if l == 0: continue
    a = np.ascontiguousarray(v[::2 ** l, ::2 ** l, ::2 ** l]); n = a.shape[0]; ch = min(n, 64); p = f"{d}/s/s.zarr/{l}"; os.makedirs(p, exist_ok=True)
    json.dump({"zarr_format": 3, "node_type": "array", "shape": [n] * 3, "data_type": "uint8", "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [ch] * 3}},
               "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}}, "fill_value": 0, "codecs": [{"name": "bytes", "configuration": {}}],
               "attributes": {}, "dimension_names": ["z", "y", "x"]}, open(p + "/zarr.json", "w"))
    for cz in range(n // ch):
        for cy in range(n // ch):
            for cx in range(n // ch):
                os.makedirs(f"{p}/c/{cz}/{cy}", exist_ok=True)
                a[cz * ch:(cz + 1) * ch, cy * ch:(cy + 1) * ch, cx * ch:(cx + 1) * ch].tofile(f"{p}/c/{cz}/{cy}/{cx}")
json.dump({"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": "0.5", "multiscales": [{"version": "0.5",
           "axes": [{"name": a, "type": "space"} for a in "zyx"], "datasets": ds}]}}}, open(f"{d}/s/s.zarr/zarr.json", "w"))
EOF
"$U" z3-put "$D/s/s.zarr/0" "$D/l0.raw" --shape 256,256,256 --z0 0 --nz 256 --shard 256 > "$D/put.log" 2>&1 || fail "z3-put ($D/put.log)"
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
echo "{\"sources\":[{\"name\":\"s\",\"root\":\"$H\",\"ct\":\"s.zarr\",\"um\":2.4,\"iso\":64,\"ct_only\":1,\"min_level\":0,\"max_level\":0,\"weight\":1}]}" > "$D/src.json"
pf() { UFSM_OCC_CACHE=0 UFSM_HTTP_CACHE="$D/$1" "$U" prefetch "$D/src.json" --levels 1 --P 128 --threads 4 > "$D/$2.log" 2>&1; }
UFSM_PREFETCH_CHUNKS=1 pf cc chunks || fail "per-chunk prefetch ($D/chunks.log)"
: > "$D/req.log"
pf cs shards || fail "shard prefetch ($D/shards.log)"
nd=$(grep '/0/c/0/0/0 ' "$D/req.log" | grep -vc 'bytes=-')
l0=$(grep 'level 0 done' "$D/shards.log")
[ -n "$l0" ] || fail "no level-0 summary line: $(cat "$D/shards.log")"
ns=$(echo "$l0" | sed -E 's/.*: ([0-9]+) stored.*/\1/'); [ "$ns" -ge 2 ] || fail "level 0: $ns stored chunks needed (fixture: expected >= 2): $l0"
[ "$nd" -ge 1 ] && [ "$nd" -lt "$ns" ] || fail "level 0: $nd data requests for $ns stored chunks (expected coalesced, fewer): $l0"
echo "$l0" | grep -q "0 shard failures; coverage 1.0000" || fail "level 0 summary: $l0"
grep -q "0 failures$" "$D/shards.log" || fail "total line: $(tail -1 "$D/shards.log")"
(cd "$D/cc" && find . -type f | sort) > "$D/cc.list"; (cd "$D/cs" && find . -type f | sort) > "$D/cs.list"
[ -s "$D/cs.list" ] || fail "the shard prefetch cached nothing"
cmp -s "$D/cc.list" "$D/cs.list" || fail "the cached chunk files differ from the per-chunk path's: $(diff "$D/cc.list" "$D/cs.list" | head -5 | tr '\n' ' ')"
pf cs again || fail "second prefetch ($D/again.log)"
l1=$(grep 'level 0 done' "$D/again.log")
echo "$l1" | grep -q "$ns stored, [0-9]* absent; $ns already cached, 0.00 GB fetched now" || fail "the second prefetch fetched again: $l1"
kill $SRV; wait $SRV 2> /dev/null; SRV=
echo "test_prefetch_shards: level 0 $ns chunks in $nd data requests; $(wc -l < "$D/cs.list") cache files = the per-chunk path's; refetch 0"
rm -rf -- "$D"
echo "test_prefetch_shards: OK"
