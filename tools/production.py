#!/usr/bin/env python3
"""Freeze, train, evaluate and deploy a staged ufsm pipeline. Full training is explicitly invoked."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_holdouts import completed_prediction, predict_atomic


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def header(path):
    with Path(path).open('rb') as f:
        line = f.readline(16384)
    if not line.startswith(b'UFSM') or not line.endswith(b'\n'):
        raise ValueError(f'invalid checkpoint header: {path}')
    return json.loads(line[4:])


def flags(opts):
    return [part for key, value in opts.items() for part in ('--' + key, str(value))]


def recipe(path):
    r = json.loads(Path(path).read_text())
    if r['version'] != 1 or not r['stages']:
        raise ValueError('unsupported or empty recipe')
    names = set()
    for stage in r['stages']:
        name = stage['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name) or name in names:
            raise ValueError('stage names must be unique simple directory names')
        names.add(name)
        if stage['P'] <= 0 or stage['B'] <= 0 or stage['steps'] <= 0 or stage.get('seconds', 0) < 0:
            raise ValueError('invalid stage dimensions or budget')
    p = r['predict']
    if p['window'] <= 0 or p['halo'] < 0 or 2 * p['halo'] >= p['window'] or p['shard'] % 128 or p['shard'] <= 0:
        raise ValueError('invalid inference tiling')
    for opts in (r['train'], *(s for s in r['stages']), *r['profiles'].values()):
        if set(opts) & {'out', 'resume', 'gpus', 'gpu'}:
            raise ValueError('output, resume and GPU selection belong to the runner')
    if r['evaluation']['report_threshold'] not in r['evaluation']['thresholds']:
        raise ValueError('report threshold must be included in the evaluation grid')
    return r


def frozen_sources(source_path, dest):
    """Keep source names and holdouts, and snapshot axis files."""
    cfg = json.loads(Path(source_path).read_text())
    if not cfg.get('sources'):
        raise ValueError('no sources')
    seen = set()
    for i, s in enumerate(cfg['sources']):
        if not s['name'] or s['name'] in seen or '/' in s['name'] or s['name'] in ('.', '..'):
            raise ValueError('source names must be unique directory names')
        seen.add(s['name'])
        if not math.isfinite(s['um']) or s['um'] <= 0 or not s['targets'].get('recto'):
            raise ValueError(f"missing CT resolution or recto labels: {s['name']}")
        h = s.get('holdout')
        if h and (len(h) != 6 or any(v < 0 for v in h[:3]) or any(v <= 0 for v in h[3:])):
            raise ValueError(f"invalid holdout: {s['name']}")
        if 'axis' in s:
            axis = dest / f'axis-{i}.json'
            shutil.copyfile(s['axis'], axis)
            s['axis'] = str(axis)
        for obj in (s, *(v for v in s['targets'].values() if isinstance(v, dict))):
            if 'root' in obj and '://' not in obj['root']:
                obj['root'] = str(Path(obj['root']).resolve())
    if 'cache' in cfg:
        cfg['cache'] = str(Path(cfg['cache']).resolve())
    atomic_json(dest / 'sources.json', cfg)
    return cfg


def environment(r, binary):
    env = {k: v for k, v in os.environ.items() if not k.startswith('UFSM_')}
    env.update({k: str(v) for k, v in r.get('environment', {}).items()})
    env['UFSM_BIN'] = str(binary)
    return env


@contextlib.contextmanager
def gpu_lock(gpu):
    # Cooperating production runners hold the lease through the whole command.
    lockdir = ROOT / 'runs/.gpu-locks'
    lockdir.mkdir(parents=True, exist_ok=True)
    with (lockdir / f'gpu-{gpu}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'GPU {gpu} already leased by a production runner')
        yield


def execute(cmd, log, env):
    print(shlex.join(map(str, cmd)), flush=True)
    started = time.monotonic()
    with Path(log).open('w') as f:
        proc = subprocess.Popen(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=env)
        try:
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            raise
    if code:
        raise RuntimeError(f'command failed ({code}); see {log}')
    return time.monotonic() - started


def train(a):
    r = recipe(a.recipe)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'; inputs.mkdir()
    binary = inputs / 'ufsm'; shutil.copy2(Path(a.binary).resolve(), binary)
    shutil.copyfile(ROOT / 'tools/eval_holdouts.py', inputs / 'eval_holdouts.py')
    frozen_sources(a.sources or r['sources'], inputs)
    atomic_json(inputs / 'recipe.json', r)
    previous = None
    if a.resume:
        previous = inputs / 'resume.ckpt'; shutil.copyfile(a.resume, previous)
        sidecar = Path(a.resume).resolve().parent / 'precision.txt'
        if sidecar.is_file(): shutil.copyfile(sidecar, inputs / 'precision.txt')
    state = dict(version=1, status='prepared', trial_seconds=a.trial_seconds, gpu=a.gpu,
                 inputs={p.name: digest(p) for p in inputs.iterdir() if p.is_file()}, stages=[])
    try:
        with gpu_lock(a.gpu):
            for stage in r['stages']:
                opts = dict(r['train'], **{k: v for k, v in stage.items() if k != 'name'})
                start_step = header(previous)['step'] if previous else 0
                opts['steps'] = start_step + opts['steps']
                opts['schedule-start'] = start_step
                if a.trial_seconds:
                    opts.update(seconds=a.trial_seconds, **{'warmup-seconds': min(40, a.trial_seconds * 0.05)})
                cmd = [binary, 'train', inputs / 'sources.json', '--out', out / stage['name'], '--gpus', a.gpu, *flags(opts)]
                if previous: cmd += ['--resume', previous]
                row = dict(name=stage['name'], command=list(map(str, cmd)), start_step=start_step, status='running')
                state['stages'].append(row); state['status'] = 'training'; atomic_json(out / 'run.json', state)
                row['seconds'] = execute(cmd, out / (stage['name'] + '.log'), environment(r, binary))
                previous = out / stage['name'] / 'last.ckpt'
                saved = header(previous)
                if saved['step'] <= start_step or not saved.get('extra', {}).get('runtime'):
                    raise RuntimeError('stage did not save a new, portable checkpoint')
                row.update(status='complete', end_step=saved['step'], checkpoint_sha256=digest(previous))
            state.update(status='trained', checkpoint=str(previous), checkpoint_sha256=digest(previous))
            atomic_json(out / 'run.json', state)
    except BaseException as e:
        state.update(status='failed', error=str(e)); atomic_json(out / 'run.json', state); raise
    print(f'trained checkpoint: {previous}', flush=True)


def run_inputs(path):
    out = Path(path).resolve(); state = json.loads((out / 'run.json').read_text())
    if state['status'] != 'trained': raise ValueError('run has not completed training')
    for name, sha in state['inputs'].items():
        if digest(out / 'inputs' / name) != sha: raise ValueError(f'frozen input changed: {name}')
    if digest(state['checkpoint']) != state['checkpoint_sha256']: raise ValueError('checkpoint changed')
    return out, state, recipe(out / 'inputs/recipe.json')


def evaluate(a):
    out, state, r = run_inputs(a.run)
    cfg = json.loads((out / 'inputs/sources.json').read_text())
    expected = {s['name'] for s in cfg['sources'] if s.get('holdout')}
    if not expected: raise ValueError('no held-out boxes')
    result = dict(version=1, checkpoint_sha256=state['checkpoint_sha256'], report_threshold=r['evaluation']['report_threshold'],
                  scope='Configured holdouts at a fixed threshold; independent calibration and global optimality are not established.', profiles={})
    with gpu_lock(a.gpu):
        for name in a.profiles.split(','):
            p = dict(r['predict'], **r['profiles'][name])
            env = environment(r, out / 'inputs/ufsm'); env['UFSM_PREDICT_ARGS'] = shlex.join(flags(p))
            scores = out / f'scores-{name}.json'
            cmd = [sys.executable, out / 'inputs/eval_holdouts.py', state['checkpoint'], '--sources', out / 'inputs/sources.json',
                   '--out', Path(a.predictions).resolve() / out.name / name, '--gpu', a.gpu,
                   '--level', r['evaluation']['level'], '--thresholds', ','.join(map(str, r['evaluation']['thresholds'])), '--scores', scores]
            elapsed = execute(cmd, out / f'eval-{name}.log', env)
            data = json.loads(scores.read_text())['scores']
            if data.keys() != expected: raise RuntimeError('evaluation omitted or added held-out boxes')
            rows = [next(row for row in s['rows'] if row['threshold'] == result['report_threshold']) for s in data.values()]
            times = [s['prediction_seconds'] for s in data.values()]
            seconds = sum(times) if all(t is not None for t in times) else None
            result['profiles'][name] = dict(arguments=p, mean_f1=sum(v['f1'] for v in rows) / len(rows),
                mean_band_f1=sum(v['band_f1'] for v in rows) / len(rows), boxes=len(rows), total_seconds=elapsed,
                prediction_seconds=seconds, output_mvox_per_second=sum(s['output_voxels'] for s in data.values()) / seconds / 1e6 if seconds else None,
                scores_sha256=digest(scores))
            atomic_json(out / 'evaluation.json', result)
            print(json.dumps({name: result['profiles'][name]}), flush=True)


def export(a):
    out, state, r = run_inputs(a.run)
    report = json.loads((out / 'evaluation.json').read_text())
    if report['checkpoint_sha256'] != state['checkpoint_sha256']: raise ValueError('evaluation is for another checkpoint')
    selected = report['profiles'][a.profile]
    if digest(out / f'scores-{a.profile}.json') != selected['scores_sha256']: raise ValueError('evaluation scores changed')
    bundle = Path(a.out).resolve(); bundle.mkdir(parents=True, exist_ok=False)
    shutil.copy2(out / 'inputs/ufsm', bundle / 'ufsm')
    shutil.copyfile(state['checkpoint'], bundle / 'model.ckpt')
    sidecar = Path(state['checkpoint']).parent / 'precision.txt'
    if sidecar.is_file(): shutil.copyfile(sidecar, bundle / 'precision.txt')
    shutil.copyfile(out / 'evaluation.json', bundle / 'evaluation.json')
    atomic_json(bundle / 'model.json', dict(version=1, status='evaluated candidate', profile=a.profile,
        prediction=selected['arguments'], threshold=report['report_threshold'], environment=r.get('environment', {}),
        artifacts={p.name: digest(p) for p in bundle.iterdir() if p.is_file()}))
    print(f"exported {bundle}; {selected['boxes']} held-outs scored", flush=True)


def predict(a):
    bundle = Path(a.bundle).resolve(); model = json.loads((bundle / 'model.json').read_text())
    for name, sha in model['artifacts'].items():
        if digest(bundle / name) != sha: raise ValueError(f'bundle artifact changed: {name}')
    out = Path(a.out).resolve()
    cmd = [str(bundle / 'ufsm'), 'predict', str(bundle / 'model.ckpt'), a.root, a.ct, str(out),
           '--um', a.um, '--gpu', a.gpu, '--level', a.level, '--levels', a.levels, *flags(model['prediction'])]
    if a.box: cmd += ['--box', a.box]
    if a.axis: cmd += ['--axis', str(Path(a.axis).resolve())]
    if a.cache: cmd += ['--cache', str(Path(a.cache).resolve())]
    signature = dict(version=1, artifacts=model['artifacts'], command=cmd, environment=model['environment'],
                     axis_sha256=digest(a.axis) if a.axis else None)
    env = environment(model, bundle / 'ufsm')
    with gpu_lock(a.gpu):
        if not completed_prediction(out, signature):
            if not predict_atomic(cmd, str(out), signature, env=env): raise RuntimeError('prediction failed; previous output preserved')
    print(f'completed prediction: {out}', flush=True)


def main():
    def interrupted(sig, _frame):
        raise SystemExit(128 + sig)
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('--recipe', default='configs/production-candidate.json')
    p = sub.add_parser('train'); p.add_argument('--recipe', default='configs/production-candidate.json')
    p.add_argument('--sources'); p.add_argument('--binary', default='build/ufsm'); p.add_argument('--out', required=True)
    p.add_argument('--gpu', default='0'); p.add_argument('--resume'); p.add_argument('--trial-seconds', type=float, default=0)
    p = sub.add_parser('evaluate'); p.add_argument('run'); p.add_argument('--gpu', default='0')
    p.add_argument('--profiles', default='matched,fp4,fp8,fp16'); p.add_argument('--predictions', required=True)
    p = sub.add_parser('export'); p.add_argument('run'); p.add_argument('--profile', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('predict'); p.add_argument('bundle'); p.add_argument('--root', required=True); p.add_argument('--ct', required=True)
    p.add_argument('--um', required=True); p.add_argument('--out', required=True); p.add_argument('--box'); p.add_argument('--axis')
    p.add_argument('--cache'); p.add_argument('--gpu', default='0'); p.add_argument('--level', default='0'); p.add_argument('--levels', default='4')
    a = parser.parse_args()
    if hasattr(a, 'gpu') and not a.gpu.isdigit(): parser.error('--gpu must select one nonnegative device')
    if hasattr(a, 'trial_seconds') and (not math.isfinite(a.trial_seconds) or a.trial_seconds < 0): parser.error('invalid trial budget')
    if a.command == 'plan': print(json.dumps(recipe(a.recipe), indent=2))
    else: {'train': train, 'evaluate': evaluate, 'export': export, 'predict': predict}[a.command](a)


if __name__ == '__main__':
    main()
