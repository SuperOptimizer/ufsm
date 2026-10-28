"""Validation compares the same voxels and weights across training window sizes."""
import csv
import json
import os
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / 'build/ufsm'

with tempfile.TemporaryDirectory(prefix='ufsm-val-window-') as tmp:
    root = pathlib.Path(tmp)
    subprocess.run([ROOT / 'build/make_pipeline_fixture', root], check=True)
    sources = root / 'sources.json'
    sources.write_text(json.dumps({'sources': [{
        'name': 'fixture', 'root': str(root), 'ct': 'ct', 'um': 1,
        'targets': {'recto': 'labels'}, 'holdout': [96, 96, 96, 32, 32, 32],
    }]}))
    common = ['--B', '1', '--widths', '16,16,32', '--levels', '1', '--gpus', '0',
              '--workers', '1', '--sampler-slots', '3', '--val-batches', '2',
              '--steps', '1', '--lr', '0', '--wd', '0', '--warmup', '0',
              '--val-every', '1', '--log-every', '1', '--ckpt-every', '1',
              '--seed', '17', '--val-seed', '29', '--fp4', '2', '--input-prec', '8',
              '--down-norm', '1', '--val-live', '1']

    def run(name, *args, force=False, ok=0):
        env = dict(os.environ)
        env.pop('UFSM_VALP_FORCE', None)
        if force:
            env['UFSM_VALP_FORCE'] = '1'
        out = root / name
        proc = subprocess.run([BIN, 'train', sources, '--out', out, *args, *common],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=180)
        assert proc.returncode == ok, proc.stdout
        if ok:
            assert '--val-P:' in proc.stdout, proc.stdout
            return
        match = re.search(r'windows hash ([0-9a-f]{16})', proc.stdout)
        assert match, proc.stdout
        rows = list(csv.DictReader((out / 'log.csv').open()))
        return match[1], float(rows[-1]['val_loss']), proc.stdout

    ordinary = run('ordinary', '--P', '16')
    copy = run('copy', '--P', '16', '--val-P', '16', force=True)
    wide = run('wide', '--P', '32', '--val-P', '16')
    assert ordinary[0] == copy[0] == wide[0], 'validation windows changed with training P'
    for other in (copy, wide):
        assert abs(ordinary[1] - other[1]) <= 2e-4, (ordinary, other)
        assert 'validation network' in other[2], other[2]
    for name, opts in [('too-large', ['--P', '16', '--val-P', '32']),
                       ('unaligned', ['--P', '32', '--val-P', '15']),
                       ('slab', ['--P', '32', '--Pz', '16', '--val-P', '16'])]:
        run(name, *opts, ok=2)
    print('validation voxels and zero-LR scores agree across training windows and validation copies: ok')
