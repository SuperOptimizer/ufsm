"""AWS pagination/download, atomic publication, real TIFF union and CPU samples."""
import hashlib
import http.server
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.parse
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("surface_store", REPO / "tools/build_surface_store.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def float_tiff(values, w, h):
    tags = [(256, 4, 1, w), (257, 4, 1, h), (258, 3, 1, 32), (259, 3, 1, 1),
            (273, 4, 1, 8 + 2 + 9 * 12 + 4), (277, 3, 1, 1), (278, 4, 1, h),
            (279, 4, 1, w * h * 4), (339, 3, 1, 3)]
    return (b"II" + struct.pack("<HIH", 42, 8, len(tags)) +
            b"".join(struct.pack("<HHII", *t) for t in tags) + struct.pack("<I", 0) +
            struct.pack("<" + "f" * len(values), *values))


class SurfaceStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.objects, self.gets = {}, []
        for segment, xpos in [("s1", 40), ("s2", 80)]:
            prefix = f"PHercParis4/segments/{segment}/mesh/{segment}-on-20260411134726-1um.tifxyz/"
            grids = [[xpos] * 64, [20 + 12 * r for r in range(8) for c in range(8)],
                     [20 + 12 * c for r in range(8) for c in range(8)]]
            for name, values in zip(["x.tif", "y.tif", "z.tif"], grids):
                self.objects[prefix + name] = float_tiff(values, 8, 8)
            self.objects[prefix + "meta.json"] = b'{"format":"tifxyz"}'
            self.objects[prefix.replace("20260411134726", "other-scan") + "meta.json"] = b"{}"
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urllib.parse.urlparse(self.path); query = urllib.parse.parse_qs(url.query)
                if "list-type" in query:
                    prefix = query["prefix"][0]; delimiter = query.get("delimiter", [None])[0]
                    rows = set()
                    for key in outer.objects:
                        if not key.startswith(prefix):
                            continue
                        tail = key[len(prefix):]
                        rows.add(("prefix", prefix + tail.split(delimiter)[0] + delimiter)
                                 if delimiter and delimiter in tail else ("object", key))
                    rows = sorted(rows); start = int(query.get("continuation-token", [0])[0])
                    end = min(start + 2, len(rows))
                    root = ET.Element("ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/")
                    ET.SubElement(root, "IsTruncated").text = "true" if end < len(rows) else "false"
                    if end < len(rows):
                        ET.SubElement(root, "NextContinuationToken").text = str(end)
                    for kind, key in rows[start:end]:
                        if kind == "prefix":
                            ET.SubElement(ET.SubElement(root, "CommonPrefixes"), "Prefix").text = key
                        else:
                            row = ET.SubElement(root, "Contents")
                            for tag, val in [("Key", key), ("Size", len(outer.objects[key])),
                                             ("ETag", '"' + hashlib.md5(outer.objects[key]).hexdigest() + '"'),
                                             ("LastModified", "2026-10-02T00:00:00Z")]:
                                ET.SubElement(row, tag).text = str(val)
                    body = ET.tostring(root)
                else:
                    key = urllib.parse.unquote(url.path.lstrip("/"))
                    if key not in outer.objects:
                        self.send_error(404); return
                    body = outer.objects[key]
                    if self.headers.get("If-Match") != '"' + hashlib.md5(body).hexdigest() + '"':
                        self.send_error(412); return
                    outer.gets.append(key)
                self.send_response(200); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        patch = mock.patch.object(builder, "AWS", f"http://127.0.0.1:{server.server_port}")
        patch.start(); self.addCleanup(patch.stop)
        self.work = self.root / "work"; self.out = self.root / "labels.zarr"; self.cfg = self.root / "merged.json"

    def args(self, binary):
        subprocess.run([str(REPO / "build/make_pipeline_fixture"), str(self.root)], check=True)
        ct = self.root / "ct"; (ct / "1").rename(ct / "0")
        group = json.loads((ct / "zarr.json").read_text())
        group["attributes"]["ome"]["multiscales"][0]["datasets"][0]["path"] = "0"
        (ct / "zarr.json").write_text(json.dumps(group)); ct.rename(self.root / "20260411134726-ct")
        template = self.root / "template.json"
        template.write_text(json.dumps({"sources": [{"name": "PHercParis4-hf", "ct": "20260411134726-ct",
                                                     "trust_band": 8, "weight": 1}]}))
        return ["builder", "--binary", str(binary), "--source-template", str(template), "--ct-root", str(self.root),
                "--ct", "20260411134726-ct", "--um", "1", "--work", str(self.work), "--out", str(self.out),
                "--sources-out", str(self.cfg), "--shard", "128", "--levels", "4", "--threads", "2"]

    def test_registered_inventory_download_and_resume(self):
        surfaces = builder.inventory("PHercParis4", "20260411134726", 1, 2)
        self.assertEqual([s["segment"] for s in surfaces], ["s1", "s2"])
        self.assertEqual(sum(len(s["objects"]) for s in surfaces), 8)
        obj = surfaces[0]["objects"][0]; first = builder.download(obj, self.work)
        self.assertEqual(first["sha256"], hashlib.sha256(self.objects[obj["key"]]).hexdigest())
        self.assertEqual(builder.download(obj, self.work), first); self.assertEqual(self.gets, [obj["key"]])
        obj = dict(obj, etag='"' + "0" * 32 + '"')
        with mock.patch.object(builder.time, "sleep"), self.assertRaises(Exception):
            builder.download(obj, self.work)

    def test_real_merged_store_and_native_training_sample(self):
        with mock.patch.object(sys, "argv", self.args(REPO / "build/ufsm")):
            builder.main()
        self.assertFalse(self.out.with_name(self.out.name + ".building").exists())
        self.assertEqual(json.loads((self.work / "build.json").read_text())["status"], "complete")
        self.assertEqual(len(json.loads(self.cfg.read_text())["sources"]), 1); self.assertEqual(len(self.gets), 8)
        source = json.loads(self.cfg.read_text())["sources"][0]
        self.assertNotIn("trust_band", source)
        self.assertEqual(source["targets"]["recto"]["encoding"], "binary")
        self.assertEqual(source["targets"]["recto"]["min_level"], 1)
        self.assertFalse((self.out / "1").exists())
        meta = json.loads((self.out / "2/zarr.json").read_text())
        self.assertEqual(meta["shape"], [64, 64, 64]); self.assertEqual(meta["fill_value"], 0)
        self.assertEqual(meta["attributes"]["ufsm"]["encoding"], "binary")
        # Binary labels must be scored as surfaces, including native requests that upsample.
        result = subprocess.run([str(REPO / "build/ufsm"), "eval", str(self.root), "20260411134726-ct",
                                 str(self.out), ".", "--um", "1", "--level", "0", "--thr", "0.5"],
                                check=True, capture_output=True, text=True)
        self.assertIn("2097152 valid voxels (100.0%)", result.stdout)
        self.assertNotIn(", 0 labelled surface", result.stdout)
        check = self.root / "check"; check.mkdir()
        subprocess.run([sys.executable, str(REPO / "tools/verify_surface_store.py"), "--work", str(self.work),
                        "--sources", str(self.cfg), "--out", str(check), "--binary", str(REPO / "build/check_surface_samples"),
                        "--P", "128", "--batches", "2"], check=True)
        samples = json.loads((check / "samples.json").read_text())
        self.assertTrue(samples["passed"]); self.assertEqual(len(samples["samples"]), 2)
        self.assertTrue(all(s["valid"] == 128**3 for s in samples["samples"]))
        self.assertEqual(json.loads((check / "qualification.json").read_text())["status"], "complete")
        pgm = (check / "sample-00-axis-0.pgm").read_bytes().split(b"\n", 3)[3]
        # Both distinct surfaces must be visible in the same target slice.
        for xpos in [40, 80]:
            self.assertEqual(pgm[60 * 3 * 128 + 128 + xpos], 255)
        # Native training can start from the finished finest level while its builder
        # continues. The snapshot must survive the staging directory's eventual rename.
        view = self.root / "training.zarr"; view_cfg = self.root / "training.json"
        subprocess.run([sys.executable, str(REPO / "tools/prepare_surface_training.py"),
                        "--work", str(self.work), "--out", str(view), "--sources-out", str(view_cfg),
                        "--source-template", str(self.cfg)], check=True)
        original = self.out / "2/c/0/0/0"
        self.assertEqual(original.stat().st_ino, (view / "2/c/0/0/0").stat().st_ino)
        self.out.rename(self.root / "renamed-labels.zarr")
        native_check = self.root / "view-check"; native_check.mkdir()
        subprocess.run([str(REPO / "build/check_surface_samples"), str(view_cfg), str(native_check), "128", "1"], check=True)
        self.assertTrue(json.loads((native_check / "samples.json").read_text())["passed"])
        raw = self.root / "occupancy.raw"
        subprocess.run([str(REPO / "build/ufsm"), "read", str(view), "256", "0", "0", "0", "1", "1", "1", str(raw)], check=True)
        self.assertEqual(raw.read_bytes(), b'\xff')

    def test_mismatched_tiff_axes_fail_the_entire_build(self):
        key = next(k for k in self.objects if k.endswith("y.tif"))
        self.objects[key] = float_tiff([40] * 16, 4, 4)
        with mock.patch.object(sys, "argv", self.args(REPO / "build/ufsm")), self.assertRaises(subprocess.CalledProcessError):
            builder.main()
        self.assertFalse(self.out.exists())
        self.assertEqual(json.loads((self.work / "build.json").read_text())["status"], "failed")

    def test_failed_raster_is_not_published(self):
        binary = self.root / "fail-raster"
        binary.write_text("#!/usr/bin/env python3\nimport pathlib,sys\npathlib.Path(sys.argv[2]).mkdir()\nraise SystemExit(1)\n")
        binary.chmod(0o755)
        with mock.patch.object(sys, "argv", self.args(binary)), self.assertRaises(subprocess.CalledProcessError):
            builder.main()
        self.assertFalse(self.out.exists()); self.assertFalse(self.cfg.exists())
        self.assertEqual(json.loads((self.work / "build.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
