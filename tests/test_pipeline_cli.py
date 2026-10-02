"""Exercise the real trainer's resume branch and portable prediction settings on local data."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"


def run(*args, ok=0):
    r = subprocess.run([str(BIN), *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert r.returncode == ok, r.stdout
    return r.stdout


def weights(path):
    with path.open("rb") as f:
        meta = json.loads(f.readline()[4:])
        return meta, f.read(meta["nparams"] * 4)


with tempfile.TemporaryDirectory(prefix="ufsm-pipeline-") as tmp:
    t = pathlib.Path(tmp)
    subprocess.run([ROOT / "build/make_pipeline_fixture", t], check=True)
    cfg = t / "sources.json"
    cfg.write_text(json.dumps({"sources": [{"name": "fixture", "root": str(t), "ct": "ct", "um": 1,
                                          "targets": {"recto": "labels"}, "holdout": [96, 96, 96, 32, 32, 32]}]}))
    opts = ["--P", "16", "--B", "1", "--levels", "1", "--gpus", "0", "--workers", "1", "--val-batches", "1",
            "--log-every", "1", "--val-every", "1", "--ckpt-every", "1", "--fp4", "2", "--down-norm", "1"]
    run("train", cfg, "--out", t / "first", "--steps", "2", "--warmup", "0", *opts)
    old, wp = weights(t / "first/last.ckpt")
    assert old["extra"]["runtime"]["act_mx4"] == 1
    assert old["extra"]["runtime"]["f16"] == 1
    run("train", cfg, "--out", t / "resume", "--resume", t / "first/last.ckpt", "--steps", "3", "--lr", "0", *opts)
    new, wr = weights(t / "resume/last.ckpt")
    assert new["step"] == 3 and wp == wr, "normal resume reinitialized or changed zero-LR weights"
    assert (t / "resume/log.csv").read_text().startswith("step,"), "resumed CSV has no header"
    # Only the model file is moved; prediction must not depend on precision.txt.
    shutil.copyfile(t / "first/last.ckpt", t / "moved.ckpt")
    pred = ["predict", t / "moved.ckpt", t, "ct"]
    geometry = ["--um", "1", "--window", "24", "--halo", "4", "--shard", "128", "--levels", "1", "--box", "0,0,0,16,16,16"]
    out = run(*pred, t / "default", *geometry)
    assert "embedded checkpoint, prec=1 f16=1 act_mx4=1" in out, out
    run(*pred, t / "explicit", *geometry, "--prec", "1", "--f16", "1", "--fp4", "1", "--policy", "all=fp4:fp4:fp4,enc0.c1=fp16")
    def chunks(path):
        return {p.relative_to(path): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file() and p.name != "zarr.json"}
    assert chunks(t / "default") == chunks(t / "explicit"), "embedded and explicit precisions differ"
    run('eval', t / 'default', '.', t / 'labels', '.', '--um', '1', '--level', '0', '--thr', '0.333,0.6',
        '--seam-core', '8', '--seam-band', '2', '--scores', t / 'regions.json')
    regions = json.loads((t / 'regions.json').read_text())['thresholds']
    assert regions[0]['threshold'] == 0.333
    for row in regions:
        for key in ('valid_voxels', 'positive_voxels', 'tp', 'fp', 'fn'):
            assert row['all'][key] == row['seam'][key] + row['interior'][key], (key, row)
    for args in (["--window", "0"], ["--halo", "12"], ["--box", "bad"], ["--shard", "129"], ["--um", "nan"], ["--prec", "999"], ["--input-mx", "2"]):
        # First occurrence wins, so put invalid overrides before the valid geometry.
        run(*pred, t / "invalid", *args, *geometry, ok=2)
    impossible = json.loads(cfg.read_text())
    impossible['sources'][0]['holdout'] = [0, 0, 0, 128, 128, 128]
    cfg.write_text(json.dumps(impossible))
    msg = run('train', cfg, '--out', t / 'impossible', '--steps', '1', *opts, ok=1)
    assert 'no eligible sources' in msg, msg
    print("real CLI resume, portable precision and geometry validation: ok")
