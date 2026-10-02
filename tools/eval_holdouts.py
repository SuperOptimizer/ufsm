#!/usr/bin/env python3
"""Predict a checkpoint on every held-out box of a sources file and score it against the labels.

usage: eval_holdouts.py <ckpt> [--sources configs/all.json] [--out /vesuvius/ufsm/eval/<run>] [--gpu 1] [--level 0]
Prints one table per source. --scores writes structured metrics; any failed box makes the command fail.
"""
import hashlib, json, os, shlex, shutil, subprocess, sys, tempfile

# the ufsm binary of this checkout (env UFSM_BIN overrides), so checkpoints with newer config fields load
B = os.environ.get("UFSM_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "ufsm"))


def arg(name, dflt):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else dflt


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def completed_prediction(pdir, signature):
    if not os.path.isfile(os.path.join(pdir, "zarr.json")):
        return False
    try:
        with open(os.path.join(pdir, ".ufsm-eval-prediction.json")) as f:
            return json.load(f) == signature
    except (OSError, ValueError):
        return False


def predict_atomic(cmd, pdir, signature):
    """Replace a previous prediction only after the new one has completed."""
    parent = os.path.dirname(os.path.abspath(pdir))
    os.makedirs(parent, exist_ok=True)
    prefix = "." + os.path.basename(pdir) + "-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as staging:
        run = cmd.copy()
        run[5] = staging
        r = subprocess.run(run, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True)
        print(r.stdout[-400:], flush=True)
        if r.returncode or not os.path.isfile(os.path.join(staging, "zarr.json")):
            return False
        with open(os.path.join(staging, ".ufsm-eval-prediction.json"), "w") as f:
            json.dump(signature, f, indent=2)
        previous = None
        if os.path.exists(pdir):
            previous = tempfile.mkdtemp(prefix=prefix + "previous-", dir=parent)
            os.rmdir(previous)
            os.replace(pdir, previous)
        try:
            os.replace(staging, pdir)
        except BaseException:
            if previous:
                os.replace(previous, pdir)
            raise
        if previous:
            shutil.rmtree(previous)
    return True


def main():
    ckpt = sys.argv[1]
    src = arg("--sources", "configs/all.json")
    out = arg("--out", "/vesuvius/ufsm/eval/" + os.path.basename(os.path.dirname(os.path.abspath(ckpt))))
    gpu, level = arg("--gpu", "1"), arg("--level", "0")
    thresholds = arg("--thresholds", "0.3,0.5,0.7")
    score_only = "--score-only" in sys.argv
    scores_path = arg("--scores", None)
    scores, failures = {}, []
    if scores_path and os.path.exists(scores_path):
        os.unlink(scores_path)  # a failed rerun must not leave an old success report
    cfg = json.load(open(src))
    cache = cfg.get("cache")
    identity = None
    if not score_only:
        manifest = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "precision.txt")
        identity = dict(version=1, checkpoint_sha256=digest(ckpt), binary_sha256=digest(shutil.which(B) or B),
                        precision_manifest_sha256=digest(manifest) if os.path.isfile(manifest) else None,
                        environment={k: v for k, v in os.environ.items() if k.startswith("UFSM_")})
    for s in cfg["sources"]:
        h = s.get("holdout")
        if not h:
            continue
        name = s["name"]
        source_level = str(max(int(level), s.get("min_level", 0)))
        # Some rasters start below CT resolution. Keep their predictions separate from earlier, finer outputs.
        suffix = ".level" + source_level if source_level != level else ""
        pdir = os.path.join(out, name + suffix)
        box = ",".join(str(v >> int(source_level)) for v in h)
        cmd = [B, "predict", ckpt, s["root"], s["ct"], pdir, "--um", str(s["um"]), "--level", source_level, "--box", box,
               "--window", "528", "--halo", "8", "--shard", "512", "--gpu", gpu, "--levels", "1"]
        if cache:
            cmd += ["--cache", cache]
        if os.environ.get("UFSM_PREDICT_ARGS"):   # e.g. "--prec 3" for fp4 inference; overrides the defaults above (ufsm takes the first occurrence of a flag)
            extra = shlex.split(os.environ["UFSM_PREDICT_ARGS"])
            keys = {a for a in extra if a.startswith("--")}
            base, i = [], 0
            while i < len(cmd):
                if cmd[i] in keys and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"): i += 2; continue
                base.append(cmd[i]); i += 1
            cmd = base + extra
        if "axis" in s:
            cmd += ["--axis", s["axis"]]
        print("==", name, "box", box, "level", source_level, flush=True)
        if score_only:
            if not os.path.isfile(os.path.join(pdir, "zarr.json")):
                failures.append(name)
                print("ERROR: missing completed prediction", flush=True)
                continue
        else:
            signature = dict(identity, predict_command=cmd,
                             axis_sha256=digest(s["axis"]) if "axis" in s else None)
            if not completed_prediction(pdir, signature) and not predict_atomic(cmd, pdir, signature):
                failures.append(name)
                continue
        t = s["targets"]["recto"]
        lroot, lgroup = (t["root"], t["group"]) if isinstance(t, dict) and "group" in t else (s["root"], t)
        ev = [B, "eval", pdir, ".", lroot, lgroup, "--um", str(s["um"]), "--level", source_level, "--tol", "2",
              "--pred-origin", ",".join(str(v) for v in h[:3]), "--thr", thresholds]
        r = subprocess.run(ev, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True)
        print(r.stdout, flush=True)
        rows = []
        for line in r.stdout.splitlines():
            fields = line.replace("|", " ").split()
            if len(fields) != 7:
                continue
            try:
                t, p, recall, f1, dice, bp, br = map(float, fields)
            except ValueError:
                continue
            rows.append(dict(threshold=t, precision=p, recall=recall, f1=f1, dice=dice,
                             band_precision=bp, band_recall=br, band_f1=2*bp*br/(bp+br) if bp+br else 0))
        if r.returncode or len(rows) != len(thresholds.split(",")):
            failures.append(name)
            print("ERROR: incomplete evaluation", flush=True)
        else:
            scores[name] = dict(level=int(source_level), rows=rows, peak=max(rows, key=lambda row: row["f1"]),
                                peak_band=max(rows, key=lambda row: row["band_f1"]))
    if failures:
        raise SystemExit("Failed held-out boxes: " + ", ".join(failures))
    if not scores:
        raise SystemExit("No held-out boxes were scored")
    if scores_path:
        os.makedirs(os.path.dirname(os.path.abspath(scores_path)), exist_ok=True)
        with open(scores_path + ".tmp", "w") as f:
            json.dump(dict(checkpoint=ckpt, sources=src, level=int(level), scores=scores), f, indent=2)
        os.replace(scores_path + ".tmp", scores_path)


if __name__ == "__main__":
    main()
