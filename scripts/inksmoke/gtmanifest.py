#!/usr/bin/env python3
"""scripts/inksmoke/gtmanifest.py OUT.tsv [SCROLL ...]: the human ink GT files the segment lists (/vesuvius/ufsm/inkgt/segs-SCROLL.json)
reference, each mapped to its public upstream URL (open-data S3 for inkgt/od via od/index.json; the scrollprize/datasets HF bucket for
inkgt/hf), with size and md5: rows "relpath url size md5" (relpath under inkgt/). A pod fetches from upstream with gtfetch.sh and checks
size + md5, so no label data travels desk -> pod (CLAUDE.md), only this list."""
import hashlib, json, os, sys
D = "/vesuvius/ufsm/inkgt"; S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink"
out = sys.argv[1]; scrolls = sys.argv[2:] or ["PHercParis4", "PHerc1667", "PHerc0139", "PHerc0814", "PHerc0009B", "PHercMAN5"]
dirs = set()
for s in scrolls:
    for e in json.load(open(f"{D}/segs-{s}.json"))["segments"]:
        for k in ("tifxyz", "labels", "mask", "validation"):
            p = e.get(k)
            if not p: continue
            r = os.path.relpath(p, D).split("/")
            dirs.add("/".join(r[:3]))   # od/<SCROLL>/<seg> or hf/<scroll>/<seg>
idx = {os.path.relpath(x["dir"], D): x for x in json.load(open(f"{D}/od/index.json"))}
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
rows = []
for d in sorted(dirs):
    for root, _, files in os.walk(os.path.join(D, d)):
        for f in files:
            p = os.path.join(root, f); rel = os.path.relpath(p, D); sub = os.path.relpath(p, os.path.join(D, d))
            if d.startswith("od/"):
                x = idx[d]
                if sub.startswith("tifxyz/"): url = f"{S3}/{x['tifxyz']}{sub[len('tifxyz/'):]}"
                else: url = f"{S3}/{x['labels']}{sub}"
            else:
                _, sc, seg = d.split("/"); url = f"{HF}/{sc}/{seg}/{sub}"
            rows.append((rel, url, os.path.getsize(p), md5(p)))
with open(out, "w") as f:
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
print(f"{len(dirs)} segment dirs, {len(rows)} files, {sum(r[2] for r in rows) / 1e9:.2f} GB -> {out}")
