#!/usr/bin/env python3
"""Deterministic local HTTP qualification for sparse Zarr reads, range validation and lifetime."""
import concurrent.futures
import http.server
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/ufsm"
HELPER = ROOT / "build/test_http_reader"
SHAPE = (13, 15, 12)
FILL = 42
MAX = (1 << 64) - 1


def crc32c(data):
    crc = 0xffffffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    return crc ^ 0xffffffff


def absent(sz, sy, sx, ci):
    return (sz, sy, sx) == (1, 1, 1) or ci == 2


def voxel(z, y, x, sparse=True, limit=SHAPE):
    if not all(0 <= c < dim for c, dim in zip((z, y, x), limit)):
        return FILL
    ci = ((z % 8 // 4) * 2 + y % 8 // 4) * 2 + x % 8 // 4
    if sparse and absent(z // 8, y // 8, x // 8, ci):
        return FILL
    return (z * 7 + y * 11 + x * 13) % 251 + 1


def expected(origin, shape, sparse=True, limit=SHAPE):
    return bytes(voxel(z, y, x, sparse, limit) for z in range(origin[0], origin[0] + shape[0])
                 for y in range(origin[1], origin[1] + shape[1])
                 for x in range(origin[2], origin[2] + shape[2]))


def metadata(sharded=True, one=False):
    chunk = [4, 4, 4]
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    if sharded:
        codecs = [{"name": "sharding_indexed", "configuration": {
            "chunk_shape": chunk, "codecs": codecs, "index_location": "end",
            "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]}}]
    return {"zarr_format": 3, "node_type": "array", "shape": SHAPE, "data_type": "uint8",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk if one or not sharded else [8, 8, 8]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": FILL, "codecs": codecs, "attributes": {}}


def fixtures():
    objects = {"blob": bytes(range(128))}
    for key, sharded, one in [("array", True, False), ("plain", False, False), ("single", True, True)]:
        objects[key + "/zarr.json"] = json.dumps(metadata(sharded, one)).encode()
        width = 8 if sharded and not one else 4
        for sz, sy, sx in itertools.product(*(range((n + width - 1) // width) for n in SHAPE)):
            if sharded and not one and (sz, sy, sx) == (1, 1, 1):
                continue
            body = bytearray(); idx = bytearray()
            for ci, (cz, cy, cx) in enumerate(itertools.product(range(width // 4), repeat=3)):
                if sharded and not one and absent(sz, sy, sx, ci):
                    idx.extend(struct.pack("<QQ", MAX, MAX)); continue
                chunk = bytes(((z * 7 + y * 11 + x * 13) % 251 + 1)
                              for z in range(sz * width + cz * 4, sz * width + cz * 4 + 4)
                              for y in range(sy * width + cy * 4, sy * width + cy * 4 + 4)
                              for x in range(sx * width + cx * 4, sx * width + cx * 4 + 4))
                if sharded:
                    # Gaps exercise slicing; one large gap must prevent coalescing.
                    body.extend(b"gap" * (3 if ci != 4 else 23000))
                    idx.extend(struct.pack("<QQ", len(body), len(chunk)))
                body.extend(chunk)
            if sharded:
                body.extend(idx); body.extend(struct.pack("<I", crc32c(idx)))
            objects[f"{key}/c/{sz}/{sy}/{sx}"] = bytes(body)
    # Each gap is legal to combine, but the whole sequence exceeds the 2 MiB request bound.
    meta = metadata(); meta["shape"] = [256, 4, 4]
    meta["chunk_grid"]["configuration"]["chunk_shape"] = [256, 4, 4]
    objects["long/zarr.json"] = json.dumps(meta).encode()
    body = bytearray(); idx = bytearray()
    for ci in range(64):
        body.extend(b"g" * 65436)
        chunk = bytes(voxel(z, y, x, False, (256, 4, 4)) for z in range(ci * 4, ci * 4 + 4)
                      for y in range(4) for x in range(4))
        idx.extend(struct.pack("<QQ", len(body), len(chunk))); body.extend(chunk)
    body.extend(idx); body.extend(struct.pack("<I", crc32c(idx)))
    objects["long/c/0/0/0"] = bytes(body)
    return objects


OBJECTS = fixtures()
REQUESTS = []
COUNTS = {}
LOCK = threading.Lock()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except ConnectionResetError:
            pass  # A rejected oversized response deliberately closes the connection.

    def reply(self, code, body=b"", headers=None, head=False):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def serve(self, head=False):
        mode, _, key = self.path.lstrip("/").partition("/")
        request_range = self.headers.get("Range")
        with LOCK:
            REQUESTS.append((mode, key, self.command, request_range))
            count_key = (mode, key, self.command)
            COUNTS[count_key] = COUNTS.get(count_key, 0) + 1
            count = COUNTS[count_key]
        if mode == "redirect":
            return self.reply(302, headers={"Location": "/ok/" + key}, head=head)
        data = OBJECTS.get(key)
        chunk = "/c/" in key
        if mode == "auth" and self.headers.get("Authorization") != "Bearer fixture-token":
            return self.reply(403, head=head)
        if (mode == "forbidden" and (chunk or key == "blob")) or (mode == "plain-forbidden" and chunk):
            return self.reply(403, head=head)
        if data is None or (mode == "missing" and (chunk or key == "blob")):
            return self.reply(404, head=head)
        if mode == "retry" and not head and (chunk or key == "blob") and count == 1:
            return self.reply(503, head=head)
        if chunk and mode in ("crc", "bounds", "overflow"):
            data = bytearray(data)
            if key.startswith("array/"):
                base = len(data) - 132
                if mode == "crc":
                    data[-1] ^= 1
                else:
                    struct.pack_into("<QQ", data, base, (1 << 63) if mode == "overflow" else len(data), 64)
                    struct.pack_into("<I", data, len(data) - 4, crc32c(data[base:-4]))
                data = bytes(data)
        if head:
            return self.reply(200, data, head=True)
        if not request_range:
            return self.reply(200, data)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request_range)
        if not match:
            return self.reply(416)
        a, b = map(int, match.groups())
        if a > b or b >= len(data):
            return self.reply(416)
        if mode == "payload-missing" and chunk and a < len(data) - 132:
            return self.reply(404)
        if mode == "ignored" and (chunk or key == "blob"):
            return self.reply(200, data)
        body = data[a:b + 1]
        total = len(data)
        range_header = f"bytes {a}-{b}/{total}"
        if mode == "wrong-range":
            range_header = f"bytes {a + 1}-{b + 1}/{total}"
        if mode == "wrong-total":
            range_header = f"bytes {a}-{b}/{b}"
        if mode == "short":
            body = body[:-1]
        headers = {} if mode == "no-range" else {"Content-Range": range_header}
        return self.reply(206, body, headers)

    def do_GET(self):
        self.serve()

    def do_HEAD(self):
        self.serve(True)


def run(cmd, env=None, success=True):
    result = subprocess.run(list(map(str, cmd)), env=env, capture_output=True, timeout=40)
    assert (result.returncode == 0) == success, (cmd, result.returncode, result.stderr.decode())
    return result


def main():
    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    clean = {k: v for k, v in os.environ.items() if not k.startswith("UFSM_")}
    try:
        with tempfile.TemporaryDirectory(prefix="ufsm-http-test-") as tmp:
            tmp = Path(tmp)
            counter = 0
            manifests = []
            def read(mode="ok", key="array", origin=(-2, -1, -3), shape=(17, 18, 18), threads=4,
                     cache=None, coalesce="1", parallel="1", success=True):
                nonlocal counter
                with LOCK:
                    counter += 1; out = tmp / f"out-{counter}.raw"
                cmd = [BINARY, "read", url + "/" + mode, key, *origin, *shape, out, "--threads", threads]
                if cache is not None:
                    cmd += ["--cache", cache]
                env = dict(clean, UFSM_Z3_COALESCE=coalesce, UFSM_Z3_INDEX_PARALLEL=parallel)
                result = run(cmd, env, success)
                if success:
                    limit = (256, 4, 4) if key == "long" else SHAPE
                    want = bytes([FILL]) * (shape[0] * shape[1] * shape[2]) if mode == "missing" else expected(origin, shape, key == "array", limit)
                    assert out.read_bytes() == want, (mode, key, threads, coalesce, parallel)
                else:
                    assert result.stderr.strip(), "missing propagated error"
                return result

            # Independent voxel reference, cold/hot caches, outside bounds, thread counts and both controls.
            for coalesce, parallel, threads in itertools.product(("0", "1"), ("0", "1"), (1, 4, 16)):
                cache = tmp / f"cache-{coalesce}-{parallel}-{threads}"
                read(cache=cache, coalesce=coalesce, parallel=parallel, threads=threads)
                manifests.append({str(p.relative_to(cache)): p.read_bytes() for p in cache.rglob("*") if p.is_file()})
                before = len(REQUESTS)
                read(cache=cache, coalesce=coalesce, parallel=parallel, threads=threads)
                assert all("/c/" not in r[1] for r in REQUESTS[before:]), "warm cache fetched shard data"
            assert all(m == manifests[0] for m in manifests), "encoded cache differs across execution modes"
            for threads in (1, 16):
                read(threads=threads)
                read(key="plain", threads=threads)
                read(key="single", cache=tmp / f"single-{threads}", threads=threads)
            for origin, shape in [((3, 5, 7), (8, 7, 4)), ((20, 20, 20), (2, 3, 4)), ((-8, -8, -8), (2, 3, 4))]:
                read(origin=origin, shape=shape, cache=tmp / f"box-{counter}")
            partial = tmp / "partial"; shutil.copytree(tmp / "cache-0-0-1", partial)
            for i, p in enumerate(sorted(partial.rglob("*"))):
                if p.is_file() and i % 3 == 0:
                    p.unlink()
            read(cache=partial)
            run([HELPER, "prefetch", url + "/ok", "single", tmp / "prefetch"], clean)
            repeat_out = tmp / "repeat.raw"
            run([HELPER, "repeat", url + "/ok", "array", tmp / "repeat", repeat_out], clean)
            assert repeat_out.read_bytes() == expected((-2, -1, -3), (17, 18, 18))
            result = run([HELPER, "resources", url + "/auth"], clean)
            print(result.stdout.decode().strip())
            # Distinguish true sparse absence from failures, with both metadata/chunk scheduling paths.
            for coalesce, parallel in itertools.product(("0", "1"), repeat=2):
                for mode in ("wrong-range", "wrong-total", "no-range", "ignored", "short", "forbidden", "crc", "bounds", "overflow", "payload-missing"):
                    cache = tmp / f"bad-{mode}-{coalesce}-{parallel}"
                    read(mode, cache=cache, coalesce=coalesce, parallel=parallel, success=False)
                    assert all(str(p.relative_to(cache))[:-8] not in OBJECTS for p in cache.rglob("*.missing")), "failure was cached as absent"
                read("missing", cache=tmp / f"missing-{coalesce}-{parallel}", coalesce=coalesce, parallel=parallel)
            read("plain-forbidden", key="plain", success=False)
            read("retry", cache=tmp / "retry")
            read("redirect", cache=tmp / "redirect")
            corrupt = tmp / "corrupt-cache"; shutil.copytree(tmp / "cache-0-0-1", corrupt)
            p = next(corrupt.rglob("*.idx")); data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
            read(cache=corrupt, success=False)
            # Multiple processes can populate one cache without temporary-file collisions.
            shared = tmp / "shared"
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for future in [pool.submit(read, cache=shared, threads=16) for _ in range(4)]:
                    future.result()
            assert {str(p.relative_to(shared)): p.read_bytes() for p in shared.rglob("*") if p.is_file()} == manifests[0]
            # Direct store contract, including legal whole responses at offset zero and redirects.
            for mode, off, length, result in [("ok", 7, 64, 64), ("redirect", 7, 64, 64),
                    ("ignored", 0, 128, 128), ("ignored", 7, 64, -1), ("short", 7, 64, -1),
                    ("wrong-range", 7, 64, -1), ("no-range", 7, 64, -1), ("wrong-total", 7, 64, -1),
                    ("missing", 7, 64, -2), ("forbidden", 7, 64, -1), ("ok", -1, 64, -1), ("ok", 0, 0, 0)]:
                run([HELPER, "range", url + "/" + mode, "blob", off, length, result], clean)
            # Coalescing reduces payload requests and respects the large-gap boundary in the fixture.
            counts = {}
            for coalesce in ("0", "1"):
                start = len(REQUESTS)
                read(cache=tmp / f"request-count-{coalesce}", coalesce=coalesce, parallel="0", threads=1)
                rows = [r for r in REQUESTS[start:] if r[2] == "GET" and "/c/" in r[1] and r[3]]
                payloads = []
                for row in rows:
                    a, b = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", row[3]).groups())
                    if b != len(OBJECTS[row[1]]) - 1:
                        payloads.append(row)
                        assert b - a + 1 <= 2 << 20
                        assert not (a < 69000 and b > 69000), "coalesced across excessive gap"
                counts[coalesce] = len(payloads)
            assert counts["1"] < counts["0"], counts
            start = len(REQUESTS)
            read(key="long", origin=(0, 0, 0), shape=(256, 4, 4), cache=tmp / "long", threads=1)
            payloads = []
            for row in REQUESTS[start:]:
                if row[1] != "long/c/0/0/0" or row[2] != "GET":
                    continue
                a, b = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", row[3]).groups())
                if b != len(OBJECTS[row[1]]) - 1:
                    payloads.append((a, b)); assert b - a + 1 <= 2 << 20
            assert 2 <= len(payloads) < 64, payloads
            print(f"HTTP reader passed: {counter} Zarr reads, sparse/raw/sharded/reference bytes, cache equivalence, failure paths, lifetime, concurrent processes; payload GETs {counts['0']} -> {counts['1']}")
    finally:
        server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    main()
