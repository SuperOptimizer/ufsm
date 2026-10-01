#!/usr/bin/env python3
"""Predict a checkpoint on every held-out box of a sources file and score it against the labels.

usage: eval_holdouts.py <ckpt> [--sources configs/all.json] [--out /vesuvius/ufsm/eval/<run>] [--gpu 1] [--level 0]
Prints one table per source; the per-source band-tolerant F1 at threshold 0.5 is the headline number.
"""
import json, os, subprocess, sys

B = "/home/forrest/ufsm/build/ufsm"


def arg(name, dflt):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else dflt


def main():
    ckpt = sys.argv[1]
    src = arg("--sources", "configs/all.json")
    out = arg("--out", "/vesuvius/ufsm/eval/" + os.path.basename(os.path.dirname(os.path.abspath(ckpt))))
    gpu, level = arg("--gpu", "1"), arg("--level", "0")
    cfg = json.load(open(src))
    cache = cfg.get("cache")
    for s in cfg["sources"]:
        h = s.get("holdout")
        if not h:
            continue
        name = s["name"]
        pdir = os.path.join(out, name)
        box = ",".join(str(v >> int(level)) for v in h)
        cmd = [B, "predict", ckpt, s["root"], s["ct"], pdir, "--um", str(s["um"]), "--level", level, "--box", box,
               "--window", "160", "--halo", "16", "--shard", "256", "--gpu", gpu, "--levels", "1"]
        if cache:
            cmd += ["--cache", cache]
        if "axis" in s:
            cmd += ["--axis", s["axis"]]
        print("==", name, "box", box, flush=True)
        if not os.path.exists(os.path.join(pdir, "zarr.json")):
            r = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True)
            print(r.stdout[-400:])
            if r.returncode:
                continue
        t = s["targets"]["recto"]
        lroot, lgroup = (t["root"], t["group"]) if isinstance(t, dict) and "group" in t else (s["root"], t)
        ev = [B, "eval", pdir, ".", lroot, lgroup, "--um", str(s["um"]), "--level", level, "--tol", "2",
              "--pred-origin", ",".join(str(v) for v in h[:3])]
        r = subprocess.run(ev, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True)
        print(r.stdout)


if __name__ == "__main__":
    main()
