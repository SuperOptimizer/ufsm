"""Exercise the real trainer's resume branch and portable prediction settings on local data."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile


def gpu_count():
    """GPUs nvidia-smi lists (the two-worker grid check needs two)"""
    try:
        lines = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.splitlines()
        return sum(line.startswith('GPU ') for line in lines)
    except OSError:
        return 0

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"


def run(*args, ok=0, extra_env=None):
    env = dict(os.environ)
    env.update(extra_env or {})
    r = subprocess.run([str(BIN), *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60, env=env)
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
            "--log-every", "1", "--val-every", "1", "--ckpt-every", "1", "--fp4", "2", "--down-norm", "1", "--input-prec", "8"]
    train_env = {"UFSM_GRAD_MX8": "1"}
    run("train", cfg, "--out", t / "first", "--steps", "2", "--warmup", "0", *opts, extra_env=train_env)
    old, wp = weights(t / "first/last.ckpt")
    assert old["extra"]["runtime"]["act_mx4"] == 1
    assert old["extra"]["runtime"]["f16"] == 1
    assert old['extra']['runtime']['input_prec'] == 8
    assert old['extra']['runtime']['grad_mx8'] == 1
    assert old['extra']['runtime']['gn_stored'] == 1
    assert "input_prec 8" in (t / "first/precision.txt").read_text()
    # --heads: a second source trains its own recto head (output channel 2); channel 1 stays the sheet channel.
    hcfg = t / "heads.json"
    src = json.loads(cfg.read_text())["sources"][0]
    hcfg.write_text(json.dumps({"sources": [src, dict(src, name="teacher2", head=1)]}))
    msg = run("train", hcfg, "--out", t / "nohead", "--steps", "1", "--warmup", "0", *opts, ok=2, extra_env=train_env)
    assert "trains head 1; train --heads 2" in msg, msg
    out = run("train", hcfg, "--out", t / "heads", "--steps", "4", "--warmup", "0", "--heads", "2", *opts, extra_env=train_env)
    assert "head 1 (channel 2): teacher2" in out and "val heads: 0 " in out, out
    hm, _ = weights(t / "heads/last.ckpt")
    assert hm["extra"]["heads"] == {"version": 1, "recto_channels": [0, 2], "sources": [["fixture"], ["teacher2"]]}, hm["extra"]
    run("predict", t / "heads/last.ckpt", t, "ct", t / "heads-pred", "--channel", "2", "--um", "1", "--window", "24", "--halo", "4", "--shard", "128",
        "--levels", "1", "--box", "0,0,0,16,16,16")
    run("train", cfg, "--out", t / "resume", "--resume", t / "first/last.ckpt", "--steps", "3", "--lr", "0", *opts[:-2], extra_env=train_env)
    new, wr = weights(t / "resume/last.ckpt")
    assert new["step"] == 3 and wp == wr, "normal resume reinitialized or changed zero-LR weights"
    assert new['extra']['runtime']['gn_stored'] == 1
    assert new['extra']['runtime']['input_prec'] == 8
    assert (t / "resume/log.csv").read_text().startswith("step,"), "resumed CSV has no header"
    # Only the model file is moved; prediction must not depend on precision.txt.
    shutil.copyfile(t / "first/last.ckpt", t / "moved.ckpt")
    pred = ["predict", t / "moved.ckpt", t, "ct"]
    geometry = ["--um", "1", "--window", "24", "--halo", "4", "--shard", "128", "--levels", "1", "--box", "0,0,0,16,16,16"]
    out = run(*pred, t / "default", *geometry)
    assert "embedded checkpoint, prec=1 f16=1 act_mx4=1" in out, out
    run(*pred, t / "explicit", *geometry, "--prec", "1", "--f16", "1", "--fp4", "1", "--input-prec", "8", "--gn-stats", "stored", "--policy", "all=fp4:fp4:fp4,enc0.c1=fp16")
    def chunks(path):
        return {p.relative_to(path): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file() and p.name != "zarr.json"}
    assert chunks(t / "default") == chunks(t / "explicit"), "embedded and explicit precisions differ"
    run(*pred, t / 'aligned-grid', *geometry, '--grid-origin', '0,0,0', '--q', '0')
    run(*pred, t / 'aligned-legacy', *geometry, '--q', '0')
    assert chunks(t / 'aligned-grid') == chunks(t / 'aligned-legacy')
    attrs = json.loads((t / 'aligned-grid/1/zarr.json').read_text())['attributes']['ufsm']
    assert attrs['grid_origin_zyx'] == [0, 0, 0]
    assert 'grid_origin_zyx' not in json.loads((t / 'aligned-legacy/1/zarr.json').read_text())['attributes']['ufsm']

    def read_bytes(path, origin, shape):
        raw = t / 'crop.raw'
        run('read', path, '1', *origin, *shape, raw, '--threads', '1')
        data = raw.read_bytes()
        assert len(data) == shape[0] * shape[1] * shape[2]
        return data

    # Complete windows stay anchored to CT coordinates even when a request starts
    # between tiles, includes CT boundaries, or changes output shards / workers.
    # Core 24 deliberately does not divide the 128-byte output shard dimension.
    grid_opts = ['--um', '1', '--window', '32', '--halo', '4', '--levels', '1',
                 '--grid-origin', '3,5,7', '--q', '0', '--threads', '1']
    grid_shape = [144, 40, 40]
    run(*pred, t / 'grid-base', *grid_opts, '--shard', '128', '--box', '0,0,0,144,40,40')
    base = read_bytes(t / 'grid-base', [0, 0, 0], grid_shape)
    assert any(base), 'grid fixture has no nonzero predictions'
    for name, placement in [('grid-shard256', ['--shard', '256']),
                            *([('grid-workers', ['--shard', '128', '--gpus', '0,1'])] if gpu_count() >= 2 else [])]:
        run(*pred, t / name, *grid_opts, *placement, '--box', '0,0,0,144,40,40')
        assert read_bytes(t / name, [0, 0, 0], grid_shape) == base, name
        assert json.loads((t / name / 'zarr.json').read_text())['attributes']['ufsm']['grid_origin_zyx'] == [3, 5, 7]
    for i, (origin, shape) in enumerate([([1, 2, 3], [19, 17, 15]), ([119, 9, 11], [17, 19, 21])]):
        expected = read_bytes(t / 'grid-base', origin, shape)
        for host in (0, 1):
            destination = t / f'grid-crop-{i}-{host}'
            run(*pred, destination, *grid_opts, '--shard', '128', '--box', ','.join(map(str, origin + shape)),
                extra_env={'UFSM_PRED_HOSTPATH': str(host)})
            assert read_bytes(destination, [0, 0, 0], shape) == expected, (origin, shape, host)
    # --full-grid: one output on the CT's whole grid, a box selecting (shard-rounded) shards; --resume skips written shards
    # (predict-done-<level>.txt); --occupancy L skips shards that are all zero on CT level L. z < 128 of ctair is air.
    air = ['predict', t / 'moved.ckpt', t, 'ctair']
    air_opts = ['--um', '1', '--window', '32', '--halo', '0', '--shard', '128', '--levels', '1', '--q', '0', '--threads', '1']
    run(*air, t / 'air-box', *air_opts, '--box', '128,0,0,128,128,128')
    out = run(*air, t / 'air-full', *air_opts, '--full-grid', '1', '--resume', '1', '--box', '130,5,5,10,10,10')
    assert 'box rounded to shards: origin 128,0,0 size 128,128,128' in out, out
    assert json.loads((t / 'air-full/1/zarr.json').read_text())['shape'] == [256, 128, 128]
    assert json.loads((t / 'air-full/1/zarr.json').read_text())['attributes']['ufsm']['origin_zyx'] == [0, 0, 0]
    assert (t / 'air-full/predict-done-1.txt').read_text() == '1 0 0\n'
    full = read_bytes(t / 'air-full', [128, 0, 0], [128, 128, 128])
    assert any(full) and full == read_bytes(t / 'air-box', [0, 0, 0], [128, 128, 128]), 'full-grid shard differs from the box run'
    before = chunks(t / 'air-full')
    out = run(*air, t / 'air-full', *air_opts, '--full-grid', '1', '--resume', '1', '--occupancy', '1')
    assert '2 shards in the box: 1 already written (--resume), 1 air on CT level 1 (--occupancy), 0 to compute' in out, out
    assert chunks(t / 'air-full') == before and (t / 'air-full/predict-done-1.txt').read_text() == '1 0 0\n'
    out = run(*air, t / 'air-occ', *air_opts, '--full-grid', '1', '--resume', '1', '--occupancy', '1')
    assert '0 already written (--resume), 1 air on CT level 1 (--occupancy), 1 to compute' in out, out
    assert read_bytes(t / 'air-occ', [128, 0, 0], [128, 128, 128]) == full
    assert (t / 'air-occ/predict-done-1.txt').read_text() == '1 0 0\n'
    if gpu_count() >= 2:
        run(*air, t / 'air-gpus', *air_opts, '--full-grid', '1', '--resume', '1', '--gpus', '0,1', '--box', '0,0,0,256,128,128')
        assert read_bytes(t / 'air-gpus', [128, 0, 0], [128, 128, 128]) == full
        assert sorted((t / 'air-gpus/predict-done-1.txt').read_text().splitlines()) == ['0 0 0', '1 0 0']
    # --occupancy also skips windows whose interior (in its shard) is air on the coarse level (one coarse voxel of margin): stride 4
    # windows over z 128..143, the first (interior z 128..131, coarse 63..66) skipped; the same bytes as without the tile test
    tile_opts = ['--um', '1', '--window', '8', '--halo', '2', '--shard', '128', '--levels', '1', '--q', '0', '--threads', '1', '--box', '128,0,0,16,8,8']
    run(*air, t / 'occ-t0', *tile_opts, '--occupancy', '1', extra_env={'UFSM_OCC_TILES': '0'})
    out = run(*air, t / 'occ-t1', *tile_opts, '--occupancy', '1')
    assert '12 windows to run, 4 more skipped (interior air on CT level 1, --occupancy)' in out, out
    assert read_bytes(t / 'occ-t1', [0, 0, 0], [16, 8, 8]) == read_bytes(t / 'occ-t0', [0, 0, 0], [16, 8, 8])
    assert any(read_bytes(t / 'occ-t1', [0, 0, 0], [16, 8, 8]))
    run(*air, t / 'invalid', *air_opts, '--occupancy', '0', ok=2)
    # --gpus with fit fields: every worker would write the whole field set -> rejected
    out = run(*air, t / 'invalid', *air_opts, '--box', '0,0,0,128,128,128', '--gpus', '0,0', '--fit-zarr', t / 'ffz-gpus', '--no-out', '1', ok=2)
    assert '--gpus with --fit-zarr / --fit-fields is not supported' in out, out
    assert not (t / 'ffz-gpus').exists()
    run(*air, t / 'invalid', *air_opts, '--full-grid', '1', '--box', '512,0,0,8,8,8', ok=2)
    # Version-1 models without a normalization field keep the historical graph.
    header, payload = (t / 'moved.ckpt').read_bytes().split(b'\n', 1)
    legacy = json.loads(header[4:]); del legacy['extra']['runtime']['gn_stored']
    (t / 'legacy.ckpt').write_bytes(b'UFSM' + json.dumps(legacy, separators=(',', ':')).encode() + b'\n' + payload)
    msg = run('predict', t / 'legacy.ckpt', t, 'ct', t / 'legacy-default', *geometry)
    assert 'gn_stats=legacy' in msg
    run('predict', t / 'legacy.ckpt', t, 'ct', t / 'legacy-explicit', *geometry, '--gn-stats', 'legacy')
    assert chunks(t / 'legacy-default') == chunks(t / 'legacy-explicit')
    run('train', cfg, '--out', t / 'legacy-resume', '--resume', t / 'legacy.ckpt', '--steps', '3', '--lr', '0', *opts[:-2], extra_env=train_env)
    assert weights(t / 'legacy-resume/last.ckpt')[0]['extra']['runtime']['gn_stored'] == 0
    run('eval', t / 'default', '.', t / 'labels', '.', '--um', '1', '--level', '0', '--thr', '0.333,0.6',
        '--seam-core', '8', '--seam-band', '2', '--scores', t / 'regions.json')
    regions = json.loads((t / 'regions.json').read_text())['thresholds']
    assert regions[0]['threshold'] == 0.333
    for row in regions:
        for key in ('valid_voxels', 'positive_voxels', 'tp', 'fp', 'fn'):
            assert row['all'][key] == row['seam'][key] + row['interior'][key], (key, row)
    # Histogram grids preserve CLI threshold conversion, order, duplicate rows,
    # endpoints, tolerance and seam support relative to the direct one-cutoff path.
    thresholds = ['1', '0', '0.333', '0.6', '0.333', '0.6']
    for tol in range(4):
        for core in (0, 8):
            geometry_eval = ['--um', '1', '--level', '0', '--box', '0,0,0,16,16,16',
                             '--tol', str(tol), '--seam-core', str(core), '--seam-band', '2']
            run('eval', t / 'default', '.', t / 'labels', '.', *geometry_eval,
                '--thr', ','.join(thresholds), '--scores', t / 'grid.json')
            grid = json.loads((t / 'grid.json').read_text())['thresholds']
            direct = {}
            for threshold in dict.fromkeys(thresholds):
                run('eval', t / 'default', '.', t / 'labels', '.', *geometry_eval,
                    '--thr', threshold, '--scores', t / 'direct.json')
                direct[threshold] = json.loads((t / 'direct.json').read_text())['thresholds'][0]
            assert grid == [direct[threshold] for threshold in thresholds], (tol, core, grid, direct)
    for args in (["--window", "0"], ["--halo", "12"], ["--box", "bad"], ["--shard", "129"], ["--um", "nan"], ["--prec", "999"], ["--input-mx", "2"], ["--input-prec", "7"], ["--input-prec", "7", "--input-mx", "0"], ["--gn-stats", "invalid"],
                 ['--grid-origin', '1,2'], ['--grid-origin', '0,0,-1'], ['--grid-origin', '0,0,0,0'],
                 ['--grid-origin', '0,0,9223372036854775808'], ['--box', '9223372036854775807,0,0,16,16,16']):
        # First occurrence wins, so put invalid overrides before the valid geometry.
        run(*pred, t / "invalid", *args, *geometry, ok=2)
    impossible = json.loads(cfg.read_text())
    impossible['sources'][0]['holdout'] = [0, 0, 0, 128, 128, 128]
    cfg.write_text(json.dumps(impossible))
    msg = run('train', cfg, '--out', t / 'impossible', '--steps', '1', *opts, ok=1)
    assert 'no eligible sources' in msg, msg
    print("real CLI resume, portable precision, globally anchored crop/shard/worker predictions and geometry validation: ok")
