"""An exported bundle must run end to end, reuse matching predictions, and reject changed artifacts."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/production.py'


def run(*args, fail=False):
    # Untracked precision settings must not leak into frozen production runs.
    env = dict(os.environ, UFSM_ACT_MX4='0', UFSM_F16='0')
    gpu_args = ['--gpu', os.environ.get('UFSM_TEST_GPU', '0')] if args[0] in ('train', 'evaluate', 'predict') else []
    p = subprocess.run([sys.executable, str(RUNNER), *map(str, args), *gpu_args], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert (p.returncode != 0) if fail else (p.returncode == 0), p.stdout
    return p.stdout


with tempfile.TemporaryDirectory(prefix='ufsm-production-') as tmp:
    t = Path(tmp)
    subprocess.run([ROOT / 'build/make_pipeline_fixture', t], check=True)
    source = t / 'sources.json'
    source.write_text(json.dumps({'sources': [{'name': 'fixture', 'root': str(t), 'ct': 'ct', 'um': 1,
        'targets': {'recto': 'labels'}, 'holdout': [96, 96, 96, 32, 32, 32]}]}))
    r = json.loads((ROOT / 'configs/production-candidate.json').read_text())
    r['sources'] = str(source)
    r['train'].update({'workers': 1, 'val-batches': 1, 'levels': '1', 'val-every': 1, 'log-every': 1, 'ckpt-every': 1})
    r['stages'] = [{'name': name, 'P': 16, 'B': 1, 'steps': 2, 'warmup': 1} for name in ('warmup', 'large')]
    r['stages'][1]['mem'] = 'wide'
    r['predict'].update(window=24, halo=4, shard=128, q=0, threads=1)
    r['evaluation'].update(level=0, thresholds=[0.333, 0.6])
    recipe = t / 'recipe.json'; recipe.write_text(json.dumps(r))
    run('train', '--recipe', recipe, '--out', t / 'run')
    manifest = json.loads((t / 'run/run.json').read_text())
    assert [s['end_step'] for s in manifest['stages']] == [2, 4]
    assert 'wide finest-level up gradient' in (t / 'run/large.log').read_text()
    with (t / 'run/large/log.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]['lr']) == 0.001, 'second stage did not restart its LR schedule'
    # Validate all storage/compute profiles and require complete numeric scores before export.
    run('evaluate', t / 'run', '--predictions', t / 'eval')
    report = json.loads((t / 'run/evaluation.json').read_text())
    assert report['profiles'].keys() == r['profiles'].keys()
    assert all(v['prediction_seconds'] > 0 and v['boxes'] == 1 for v in report['profiles'].values())
    for profile in r['profiles']:
        scores = json.loads((t / 'run' / f'scores-{profile}.json').read_text())['scores']['fixture']
        assert scores['rows'][0]['threshold'] == 0.333
        assert 0 < scores['positive_voxels'] < scores['valid_voxels']
        assert scores['constant_foreground_f1'] == 2*scores['positive_voxels']/(scores['valid_voxels']+scores['positive_voxels'])
    run('export', t / 'run', '--profile', 'matched', '--out', t / 'bundle')
    # Move the complete bundle: its binary, checkpoint and settings must be self-contained.
    (t / 'bundle').rename(t / 'moved')
    args = ['predict', t / 'moved', '--root', t, '--ct', 'ct', '--um', '1', '--out', t / 'pred',
            '--box', '0,0,0,16,16,16', '--levels', '1']
    run(*args)
    group = t / 'pred/zarr.json'; unchanged = group.stat().st_mtime_ns
    run(*args); assert group.stat().st_mtime_ns == unchanged, 'matching prediction was regenerated'
    # A failed redo leaves the previous successful store intact.
    bad = args.copy(); bad[bad.index('--ct') + 1] = 'missing'
    run(*bad, fail=True); assert group.stat().st_mtime_ns == unchanged
    with (t / 'moved/model.ckpt').open('ab') as f: f.write(b'changed')
    run(*args, fail=True); assert group.stat().st_mtime_ns == unchanged
    print('production train/evaluate/export/moved-bundle/predict/reuse/failure integrity: ok')
