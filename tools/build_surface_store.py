#!/usr/bin/env python3
"""Build one lossless label pyramid from every AWS surface registered to a CT scan.

The final store is published only after rasterization succeeds. Raw AWS files,
ETags and SHA256 hashes remain in --work for restart and provenance.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

AWS = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
REPO = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def list_objects(prefix, delimiter=None):
    params = {"list-type": "2", "prefix": prefix}
    if delimiter:
        params["delimiter"] = delimiter
    objects, prefixes = [], []
    while True:
        url = AWS + "/?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for row in root.findall("s:Contents", NS):
            objects.append({"key": row.find("s:Key", NS).text,
                            "bytes": int(row.find("s:Size", NS).text),
                            "etag": row.find("s:ETag", NS).text,
                            "modified": row.find("s:LastModified", NS).text})
        prefixes += [r.find("s:Prefix", NS).text for r in root.findall("s:CommonPrefixes", NS)]
        if root.find("s:IsTruncated", NS).text != "true":
            return objects, prefixes
        params["continuation-token"] = root.find("s:NextContinuationToken", NS).text


def inventory(scroll, volume, um, workers):
    _, prefixes = list_objects(f"{scroll}/segments/", "/")
    def get(prefix):
        segment = prefix.rstrip("/").split("/")[-1]
        _, meshes = list_objects(prefix + "mesh/", "/")
        suffix = f"-on-{volume}-{um:g}um.tifxyz/"
        candidates = [p for p in meshes if p.endswith(suffix)]
        if len(candidates) != 1:
            raise ValueError(f"{segment}: expected one registered mesh for {volume}; got {candidates}")
        objects, _ = list_objects(candidates[0], "/")
        required = {"x.tif", "y.tif", "z.tif", "meta.json"}
        available = {r["key"].split("/")[-1] for r in objects}
        if not required <= available:
            raise ValueError(f"{segment}: missing {required - available}")
        selected = [r for r in objects if r["key"].split("/")[-1] in required | {"mask.tif"}]
        return {"segment": segment, "mesh_prefix": candidates[0], "objects": selected}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(get, sorted(prefixes)))


def download(obj, work):
    path = work / "aws" / obj["key"]
    receipt = path.with_name(path.name + ".receipt.json")
    if path.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if path.stat().st_size == obj["bytes"] and saved.get("etag") == obj["etag"]:
            return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    url = AWS + "/" + urllib.parse.quote(obj["key"], safe="/")
    for attempt in range(3):
        try:
            sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
            request = urllib.request.Request(url, headers={"If-Match": obj["etag"]})
            with urllib.request.urlopen(request, timeout=90) as response, tmp.open("wb") as dest:
                while chunk := response.read(1024 * 1024):
                    dest.write(chunk)
                    sha.update(chunk)
                    md5.update(chunk)
                    size += len(chunk)
            if size != obj["bytes"]:
                raise IOError(f"{obj['key']}: {size} bytes, expected {obj['bytes']}")
            etag = obj["etag"].strip('"')
            if "-" not in etag and md5.hexdigest() != etag:
                raise IOError(f"{obj['key']}: ETag checksum mismatch")
            tmp.replace(path)
            saved = dict(obj, sha256=sha.hexdigest(), url=url, local=str(path))
            atomic_json(receipt, saved)
            return saved
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scroll", default="PHercParis4")
    parser.add_argument("--volume", default="20260411134726")
    parser.add_argument("--um", type=float, default=2.4)
    parser.add_argument("--ct-root", default="/vesuvius/usrm/volcomp")
    parser.add_argument("--ct", default="PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr")
    parser.add_argument("--work", type=Path, default=Path("/vesuvius/ufsm/gt/paris4-all-surfaces"))
    parser.add_argument("--out", type=Path, default=Path("/vesuvius/ufsm/gt/paris4-all-surfaces/labels.zarr"))
    parser.add_argument("--sources-out", type=Path, default=REPO / "configs/paris4-all-surfaces.json")
    parser.add_argument("--source-template", type=Path, default=REPO / "configs/all2.json")
    parser.add_argument("--binary", type=Path, default=REPO / "build/ufsm")
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shard", type=int, default=512)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.out = args.out.resolve()
    if (min(args.download_workers, args.threads) <= 0 or args.threads > 256 or
            args.shard < 128 or args.shard % 128 or not 1 <= args.levels <= 12 or
            not 1 <= args.T <= 84 or not 0 <= args.level <= 20 or not math.isfinite(args.um) or args.um <= 0):
        parser.error("invalid worker count, geometry or label width")
    args.work.mkdir(parents=True, exist_ok=True)
    manifest_path = args.work / "aws-registered-surfaces.json"
    if manifest_path.exists():
        surfaces = json.loads(manifest_path.read_text())
        expected_prefix = f"{args.scroll}/segments/"
        expected_suffix = f"-on-{args.volume}-{args.um:g}um.tifxyz/"
        if not surfaces or any(not s["mesh_prefix"].startswith(expected_prefix) or
                               not s["mesh_prefix"].endswith(expected_suffix) for s in surfaces):
            raise ValueError("cached inventory belongs to another scan")
    else:
        surfaces = inventory(args.scroll, args.volume, args.um, args.download_workers)
        if not surfaces:
            raise ValueError("no registered surfaces found")
        atomic_json(manifest_path, surfaces)
    objects = [o for s in surfaces for o in s["objects"]]
    print(f"AWS inventory: {len(surfaces)} surfaces, {len(objects)} files, "
          f"{sum(o['bytes'] for o in objects)/2**30:.2f} GiB", flush=True)
    receipts, downloaded = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        jobs = {pool.submit(download, o, args.work): o for o in objects}
        for future in concurrent.futures.as_completed(jobs):
            receipt = future.result()
            receipts.append(receipt)
            downloaded += receipt["bytes"]
            if len(receipts) % 12 == 0 or len(receipts) == len(objects):
                print(f"AWS download: {len(receipts)}/{len(objects)} files, {downloaded/2**30:.2f} GiB", flush=True)
    atomic_json(args.work / "download-receipts.json", sorted(receipts, key=lambda o: o["key"]))
    if args.fetch_only:
        return
    if args.out.exists():
        raise FileExistsError(f"store already exists: {args.out}; choose a new output")
    staging = args.out.with_name(args.out.name + ".building")
    if staging.exists():
        raise FileExistsError(f"incomplete staging store exists: {staging}")
    template = json.loads(args.source_template.read_text())
    matching = [s for s in template["sources"] if s["name"].startswith(args.scroll + "-hf") and args.volume in s["ct"]]
    if len(matching) != 1:
        raise ValueError("expected one matching template source for axis and held-out box")
    ct_meta = json.loads((Path(args.ct_root) / args.ct / "0/zarr.json").read_text())
    shape = ct_meta["shape"]
    mesh_paths = [args.work / "aws" / s["mesh_prefix"] for s in surfaces]
    command = [str(args.binary.resolve()), "raster", str(staging), "--shape", ",".join(map(str, shape)),
               "--um", str(args.um), "--level", str(args.level), "--T", str(args.T),
               "--levels", str(args.levels), "--threads", str(args.threads), "--shard", str(args.shard),
               *map(str, mesh_paths)]
    provenance = {"scroll": args.scroll, "volume": args.volume, "ct_root": args.ct_root, "ct": args.ct,
                  "native_shape_zyx": shape, "native_um": args.um, "surface_count": len(surfaces),
                  "surface_segments": [s["segment"] for s in surfaces], "encoding": "0=background,254=surface,255=ignore",
                  "label_level": args.level, "T": args.T, "command": command,
                  "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  "inventory_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    atomic_json(args.work / "build.json", dict(provenance, status="rasterizing"))
    print("Rasterizing every surface into one store", flush=True)
    try:
        with (args.work / "raster.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    except Exception as error:
        atomic_json(args.work / "build.json", dict(provenance, status="failed", error=str(error)))
        raise
    atomic_json(staging / "provenance.json", provenance)
    staging.rename(args.out)
    source = matching[0].copy()
    source.update(name=args.scroll + "-all-surfaces", root=args.ct_root, ct=args.ct, um=args.um,
                  targets={"recto": {"root": str(args.out), "group": "."}}, weight=1.0)
    source.pop("trust_band", None)  # validity is already encoded around the union of meshes
    atomic_json(args.sources_out, {"cache": template.get("cache"), "sources": [source]})
    atomic_json(args.work / "build.json", dict(provenance, status="complete", store=str(args.out), sources=str(args.sources_out)))
    print(f"SURFACE_STORE_READY {args.out}\nSOURCES_READY {args.sources_out}", flush=True)


if __name__ == "__main__":
    main()
