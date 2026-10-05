"""Real CLI regression: finite EOF must apply the last update, and resume must
preserve committed tile order and the original LR schedule. Requires an idle GPU."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('UFSM_TEST_BINARY', ROOT / 'build/ufsm'))

def header(path):
    with path.open('rb') as f:
        return json.loads(f.readline()[4:])

with tempfile.TemporaryDirectory(prefix='ufsm-cover-cli-') as tmp:
    t = Path(tmp)
    subprocess.run([ROOT / 'build/make_pipeline_fixture', t], check=True)
    source = t / 'sources.json'
    source.write_text(json.dumps({'sources': [{'name': 'fixture', 'root': str(t), 'ct': 'ct', 'um': 1,
        'targets': {'recto': 'labels'}, 'holdout': [96, 96, 96, 32, 32, 32]}]}))
    cover = t / 'cover.json'
    cover.write_text(json.dumps({'version': 1, 'P': 32, 'level': 0, 'count': 3,
        'source_names': ['fixture'], 'tiles': [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]]}))
    env = {k: v for k, v in os.environ.items() if not k.startswith('UFSM_')}
    base = [BINARY, 'train', source, '--P', '32', '--B', '1', '--cover', cover,
            '--steps', '3', '--warmup', '10', '--seed', '2', '--workers', '2', '--val-batches', '1',
            '--levels', '1', '--log-every', '1', '--ckpt-every', '1', '--val-every', '10',
            '--down-norm', '1', '--zfix', '1', '--symmetry-p', '.5', '--ct-aug', '1', '--soft', '3',
            '--opt', 'muon', '--input-prec', '8', '--gn-stats', 'stored']
    for mode, gpu in [('single', os.environ.get('UFSM_TEST_GPU', '0')), ('split', '0,1')]:
        if mode == 'split' and os.environ.get('UFSM_TEST_SINGLE_ONLY'):
            continue
        command = base + ['--gpus', gpu] + (['--split', 'z'] if mode == 'split' else [])
        def train(name, extra=()):
            out = t / (mode + '-' + name)
            p = subprocess.run(list(map(str, command + ['--out', out, *extra])), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert p.returncode == 0, p.stdout
            return out, header(out / 'last.ckpt')
        full, h = train('full')
        assert h['step'] == 3 and h['extra']['cover']['cursor'] == 3
        assert h['extra']['cover']['base_step'] == 0 and h['nparams'] == 1172050
        partial, hp = train('partial', ['--limit-seconds', '.000001'])
        assert hp['step'] == 1 and hp['extra']['cover']['cursor'] == 1
        resumed, hr = train('resumed', ['--resume', partial / 'last.ckpt'])
        assert hr['step'] == 3 and hr['extra']['cover'] == h['extra']['cover']
        with (full / 'log.csv').open() as f: full_rows = list(csv.DictReader(f))
        with (resumed / 'log.csv').open() as f: resume_rows = list(csv.DictReader(f))
        assert [r['step'] for r in full_rows] == ['1', '2', '3']
        assert [r['step'] for r in resume_rows] == ['2', '3']
        assert [r['lr'] for r in resume_rows] == [r['lr'] for r in full_rows[1:]]
        # Cursor3 must include a real optimizer update, not merely an EOF header.
        def weights(path):
            with path.open('rb') as f: f.readline(); return f.read(1172050 * 4)
        assert weights(full / 'last.ckpt') != weights(partial / 'last.ckpt')
        print(f'{mode}: all3 tiles applied; partial cursor1 resumes at2 with original LR: ok')
        # Repeating a pass retains the global optimizer step, and only a fully
        # committed pass may transition to a different shuffled plan.
        bad = subprocess.run(list(map(str,command + ['--out',t/'bad','--resume',partial/'last.ckpt',
            '--cover-next-pass','1'])),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        assert bad.returncode and 'requires a completed' in bad.stdout,bad.stdout
        second=t/'second.json'; plan=json.loads(cover.read_text());plan['seed']=3;plan['tiles'].reverse()
        second.write_text(json.dumps(plan))
        command[command.index('--cover')+1]=second
        command += ['--sched','wsd','--schedule-seconds','10000','--schedule-elapsed','6000','--warmup-seconds','0']
        next_partial, hn = train('next-partial',['--resume',full/'last.ckpt','--cover-next-pass','1','--limit-seconds','.000001'])
        assert hn['step']==4 and hn['extra']['cover']['cursor']==1 and hn['extra']['cover']['base_step']==3
        assert hn['extra']['cover']['sha256']!=h['extra']['cover']['sha256'] and hn['muon_mom']==1
        next_full, hf = train('next-resumed',['--resume',next_partial/'last.ckpt'])
        assert hf['step']==6 and hf['extra']['cover']['cursor']==3 and hf['extra']['cover']['base_step']==3
        assert weights(next_partial/'last.ckpt')!=weights(full/'last.ckpt')
        for out in (next_partial,next_full):
            rows=list(csv.DictReader((out/'log.csv').open()))
            assert all(float(r['lr'])==.001 for r in rows),rows
        print(f'{mode}: completed pass transitions and resumes with optimizer step and wall-time LR intact: ok')
