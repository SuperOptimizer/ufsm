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
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_holdouts import completed_prediction, predict_atomic
import build_training_cover as training_cover


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
    thresholds = r['evaluation']['thresholds']
    if not 1 <= len(thresholds) <= 16 or len(set(thresholds)) != len(thresholds) or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError('threshold grid must contain 1..16 distinct finite values in [0,1]')
    return r


def evaluation_plan(e, expected):
    """Validate explicit label groups and a split before examining model scores."""
    groups = e.get('groups', {})
    assigned = []
    for name, members in groups.items():
        if not name or not isinstance(members, list) or not members:
            raise ValueError('evaluation groups must have names and source lists')
        assigned.extend(members)
    if groups and (len(set(assigned)) != len(assigned) or set(assigned) != expected):
        raise ValueError('evaluation groups must cover each held-out source exactly once')
    levels = e.get('source_levels', {})
    if set(levels) - expected or any(type(v) is not int or not 0 <= v <= 20 for v in levels.values()):
        raise ValueError('source evaluation levels must name held-out sources and use integers 0..20')
    calibration = e.get('calibration')
    if calibration:
        if calibration.get('aggregation', 'arithmetic') not in ('arithmetic', 'geometric'):
            raise ValueError('calibration aggregation must be arithmetic or geometric')
        names = calibration.get('sources', [])
        acceptance = e.get('acceptance_sources', [])
        group = groups.get(calibration.get('group'), [])
        if (not names or not acceptance or len(set(names)) != len(names) or len(set(acceptance)) != len(acceptance)
                or set(names) & set(acceptance) or set(names + acceptance) != set(group)):
            raise ValueError('calibration and acceptance must partition the configured calibration group')
    elif e.get('acceptance_sources'):
        raise ValueError('acceptance sources require a calibration split')
    return groups


def summarize_scores(data, e):
    """Fit one cutoff per serving profile using calibration sources only."""
    groups = evaluation_plan(e, set(data))
    grid = e['thresholds']
    by_threshold = {}
    for name, score in data.items():
        rows = score['rows']
        if len(rows) != len(grid) or {r['threshold'] for r in rows} != set(grid):
            raise ValueError(f'incomplete threshold grid: {name}')
        by_threshold[name] = {r['threshold']: r for r in rows}
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for row in rows for k in ('f1', 'band_f1')):
            raise ValueError(f'invalid metrics: {name}')
    threshold = e['report_threshold']
    calibration = None
    if e.get('calibration'):
        names = e['calibration']['sources']
        curve = [dict(threshold=t, mean_f1=sum(by_threshold[n][t]['f1'] for n in names)/len(names)) for t in grid]
        geometric = e['calibration'].get('aggregation', 'arithmetic') == 'geometric'
        if geometric:
            # Relative gains across sources have equal weight. A collapsed source
            # cannot be offset by an absolute gain on an easier source. Preserve
            # exact zeros rather than introducing a score floor or pseudocount.
            for row in curve:
                values = [by_threshold[n][row['threshold']]['f1'] for n in names]
                row['geometric_mean_f1'] = math.exp(math.fsum(math.log(v) for v in values)/len(values)) if all(values) else 0.
        # Resolve ties by the lower cutoff, independently of JSON grid ordering.
        criterion = 'geometric_mean_f1' if geometric else 'mean_f1'
        threshold = max(curve, key=lambda v: (v[criterion], -v['threshold']))['threshold']
        calibration = dict(sources=names, metric='geometric mean box F1' if geometric else 'macro box F1', curve=curve, threshold=threshold)
        if geometric:
            calibration['aggregation'] = 'geometric'
    def aggregate(names):
        selected = [by_threshold[n][threshold] for n in names]
        constants = [data[n].get('constant_foreground_f1') for n in names]
        return dict(sources=names, boxes=len(names), mean_f1=sum(r['f1'] for r in selected)/len(selected),
                    mean_band_f1=sum(r['band_f1'] for r in selected)/len(selected),
                    constant_foreground_mean_f1=sum(constants)/len(constants) if all(v is not None for v in constants) else None)
    result = dict(threshold=threshold, **aggregate(list(data)), groups={name:aggregate(members) for name,members in groups.items()})
    if calibration:
        result.update(calibration=calibration, acceptance=aggregate(e['acceptance_sources']))
    result['per_box_support'] = {n:dict(f1=by_threshold[n][threshold]['f1'], level=s.get('level'),
        positive_fraction=s.get('positive_fraction'), constant_foreground_f1=s.get('constant_foreground_f1')) for n,s in data.items()}
    return result


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


def validate_cover(plan, cfg):
    """Check finite-plan structure and the frozen source's volume identity.

    The original source JSON checksum is provenance only: freezing changes axis
    filenames. CT/label metadata, label provenance and holdout identities remain
    stable and are checked explicitly instead.
    """
    names = [s['name'] for s in cfg['sources']]
    if (len(names) != 1 or plan.get('version') != 1 or plan.get('level') != 0 or
            plan.get('source_names') != names):
        raise ValueError('cover must name the single merged native source')
    P, count = plan.get('P'), plan.get('count')
    tiles, bounds = plan.get('tiles'), plan.get('bounds')
    if (type(P) is not int or P <= 0 or type(count) is not int or count <= 0 or
            not isinstance(tiles, list) or len(tiles) != count or
            not isinstance(bounds, list) or len(bounds) != 6 or
            any(type(v) is not int or v < 0 for v in bounds) or
            any(bounds[d + 3] - bounds[d] < P for d in range(3))):
        raise ValueError('invalid cover dimensions, bounds or tile count')
    source = cfg['sources'][0]
    target = source['targets']['recto']
    if not isinstance(target, dict) or target.get('encoding') != 'binary':
        raise ValueError('cover requires a merged binary target')
    binding, coverage = plan.get('binding', {}), plan.get('coverage', {})
    ct_root = training_cover.local_root(source['root'])
    label_root = training_cover.local_root(target.get('root', source['root']))
    label_group = str(Path(target.get('group', '.')))
    if (binding.get('ct_root') != ct_root or binding.get('ct') != source['ct'] or
            binding.get('label_root') != label_root or
            str(Path(binding.get('label_group', ''))) != label_group or
            not math.isclose(binding.get('native_um', 0), source['um'], rel_tol=1e-8)):
        raise ValueError('cover source CT or label identity does not match')
    ct_group, sha = training_cover.read_metadata(ct_root, source['ct'])
    if binding.get('ct_group_metadata_sha256') != sha:
        raise ValueError('cover CT group metadata changed')
    levels = training_cover.datasets(ct_group)
    relative = all(math.isclose(v, 1, rel_tol=1e-8) for v in levels[0][1])
    scale = 1 if relative else source['um']
    native = next((k for k, s in levels if all(math.isclose(v, scale, rel_tol=1e-8) for v in s)), None)
    if native is None:
        raise ValueError('cover source has no native CT array')
    meta, sha = training_cover.read_metadata(ct_root, source['ct'].rstrip('/') + '/' + native)
    shape = training_cover.shape3(meta)
    if binding.get('ct_array_metadata_sha256') != sha or binding.get('native_shape_zyx') != list(shape):
        raise ValueError('cover CT shape or array metadata changed')
    if any(bounds[d + 3] > shape[d] for d in range(3)):
        raise ValueError('cover bounds exceed the native CT shape')
    _, sha = training_cover.read_metadata(label_root, label_group)
    if binding.get('label_group_metadata_sha256') != sha:
        raise ValueError('cover label group metadata changed')
    if '://' in label_root:
        with urllib.request.urlopen(label_root.rstrip('/') + '/' + label_group + '/provenance.json', timeout=60) as f:
            provenance_sha = hashlib.sha256(f.read()).hexdigest()
    else:
        provenance_sha = digest(Path(label_root) / label_group / 'provenance.json')
    if binding.get('label_provenance_sha256') != provenance_sha:
        raise ValueError('cover label provenance changed')
    occupancy_key = binding.get('occupancy_array')
    if not isinstance(occupancy_key, str) or not occupancy_key or '..' in Path(occupancy_key).parts:
        raise ValueError('invalid cover occupancy array')
    occupancy_meta, sha = training_cover.read_metadata(label_root, label_group + '/' + occupancy_key)
    if (binding.get('occupancy_metadata_sha256') != sha or
            binding.get('occupancy_shape_zyx') != list(training_cover.shape3(occupancy_meta))):
        raise ValueError('cover occupancy metadata changed')
    occupancy_sha = binding.get('occupancy_sha256', '')
    if len(occupancy_sha) != 64 or any(c not in '0123456789abcdef' for c in occupancy_sha):
        raise ValueError('invalid cover occupancy checksum')
    holdout = source.get('holdout')
    holes = [holdout[:3] + [holdout[d] + holdout[d + 3] for d in range(3)]] if holdout else []
    if coverage.get('holdouts') != holes or coverage.get('snap') is not False or coverage.get('position_jitter') is not False:
        raise ValueError('cover holdout, snapping or position-jitter contract changed')
    seen = set()
    for row in tiles:
        if not isinstance(row, list) or len(row) != 4 or any(type(v) is not int for v in row) or row[0] != 0:
            raise ValueError('invalid cover tile source or coordinates')
        origin = row[1:]
        if any(origin[d] < bounds[d] or origin[d] + P > bounds[d + 3] for d in range(3)):
            raise ValueError('cover tile escapes its bounds')
        if any(training_cover.intersects(tuple(origin) + tuple(v + P for v in origin), h) for h in holes):
            raise ValueError('cover tile intersects a holdout')
        if tuple(row) in seen:
            raise ValueError('cover contains duplicate tiles')
        seen.add(tuple(row))


def cover_checkpoint(saved, sha, count):
    value = saved.get('extra', {}).get('cover')
    if (not isinstance(value, dict) or value.get('sha256') != sha or value.get('count') != count or
            any(type(value.get(k)) is not int for k in ('count', 'cursor', 'base_step')) or
            not 0 <= value['cursor'] <= count or value['base_step'] < 0 or
            saved.get('step') != value['base_step'] + value['cursor']):
        raise ValueError('checkpoint cover hash, count or committed cursor does not match')
    return value


def environment(r, binary):
    env = {k: v for k, v in os.environ.items() if not k.startswith('UFSM_')}
    env.update({k: str(v) for k, v in r.get('environment', {}).items()})
    env['UFSM_BIN'] = str(binary)
    return env


def gpu_devices(selection):
    parts = str(selection).split(',')
    if not 1 <= len(parts) <= 8 or any(not p.strip() or any(c not in '0123456789' for c in p.strip()) for p in parts):
        raise ValueError('GPU selection must contain 1..8 nonnegative device IDs')
    devices = [int(p) for p in parts]
    if len(set(devices)) != len(devices):
        raise ValueError('GPU selection must not repeat a device')
    return devices


@contextlib.contextmanager
def gpu_lock(gpu):
    # Cooperating production runners hold the lease through the whole command.
    lockdir = ROOT / 'runs/.gpu-locks'
    lockdir.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as locks:
        for device in sorted(gpu_devices(gpu)):
            lock = locks.enter_context((lockdir / f'gpu-{device}.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'GPU {device} already leased by a production runner')
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
    if a.split is not None:
        r['train']['split'] = a.split
    if a.mem is not None:
        for stage in r['stages']:
            stage['mem'] = a.mem
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'; inputs.mkdir()
    binary = inputs / 'ufsm'; shutil.copy2(Path(a.binary).resolve(), binary)
    shutil.copyfile(ROOT / 'tools/eval_holdouts.py', inputs / 'eval_holdouts.py')
    cfg = frozen_sources(a.sources or r['sources'], inputs)
    evaluation_plan(r['evaluation'], {s['name'] for s in cfg['sources'] if s.get('holdout')})
    cover_path = (getattr(a, 'cover', None) or r['train'].get('cover') or
                  next((s['cover'] for s in r['stages'] if s.get('cover')), None))
    cover, cover_sha = None, None
    if cover_path:
        if len(r['stages']) != 1:
            raise ValueError('a finite cover must be one complete-pass stage')
        frozen_cover = inputs / 'cover.json'
        shutil.copyfile(cover_path, frozen_cover)
        cover = json.loads(frozen_cover.read_text()); cover_sha = digest(frozen_cover)
        validate_cover(cover, cfg)
        opts = dict(r['train'], **{k: v for k, v in r['stages'][0].items() if k != 'name'})
        if opts['P'] != cover['P'] or opts['B'] != 1 or opts['steps'] != cover['count']:
            raise ValueError('cover stage must use its P, batch 1 and exactly count steps')
        if (int(opts.get('finetune', 0)) or int(opts.get('overfit', 0)) or
                float(opts.get('seconds', 0)) or float(opts.get('limit-seconds', 0))):
            raise ValueError('cover forbids finetune, overfit and recipe time budgets; use --trial-seconds for a trial')
        if len(gpu_devices(a.gpu)) > 1 and opts.get('split') != 'z':
            raise ValueError('cover with multiple GPUs requires spatial splitting')
        if opts.get('split') == 'z' and len(gpu_devices(a.gpu)) != 2:
            raise ValueError('cover spatial splitting requires exactly two GPUs')
        r['train'].update(cover=str(frozen_cover), **{'cover-sha256': cover_sha})
        r['stages'][0].pop('cover', None); r['stages'][0].pop('cover-sha256', None)
    elif r['train'].get('cover-sha256') or any(s.get('cover-sha256') for s in r['stages']):
        raise ValueError('cover-sha256 requires a cover plan')
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
                resumed = header(previous) if previous else {'step': 0}
                start_step = resumed['step']
                base_step, start_cursor = start_step, 0
                if cover:
                    if 'cover' in resumed.get('extra', {}):
                        continuation = cover_checkpoint(resumed, cover_sha, cover['count'])
                        base_step, start_cursor = continuation['base_step'], continuation['cursor']
                        if start_cursor == cover['count']:
                            raise ValueError('resumed cover already completed its pass')
                    opts['steps'] = base_step + cover['count']
                    opts['schedule-start'] = base_step
                    opts.pop('seconds', None); opts.pop('warmup-seconds', None)
                else:
                    opts['steps'] = start_step + opts['steps']
                    opts['schedule-start'] = start_step
                if a.trial_seconds:
                    if cover:
                        opts['limit-seconds'] = a.trial_seconds
                    else:
                        opts.update(seconds=a.trial_seconds, **{'warmup-seconds': min(40, a.trial_seconds * 0.05)})
                cmd = [binary, 'train', inputs / 'sources.json', '--out', out / stage['name'], '--gpus', a.gpu, *flags(opts)]
                if previous: cmd += ['--resume', previous]
                spatial = opts.get('split') == 'z'
                row = dict(name=stage['name'], command=list(map(str, cmd)), start_step=start_step, status='running',
                           parallelism='spatial' if spatial else 'data',
                           effective_batch=opts['B'] * (1 if spatial else len(gpu_devices(a.gpu))))
                if cover:
                    row['cover'] = dict(sha256=cover_sha, count=cover['count'], cursor=start_cursor,
                                        base_step=base_step, remaining=cover['count'] - start_cursor)
                state['stages'].append(row); state['status'] = 'training'; atomic_json(out / 'run.json', state)
                row['seconds'] = execute(cmd, out / (stage['name'] + '.log'), environment(r, binary))
                previous = out / stage['name'] / 'last.ckpt'
                saved = header(previous)
                if saved['step'] <= start_step or not saved.get('extra', {}).get('runtime'):
                    raise RuntimeError('stage did not save a new, portable checkpoint')
                status = 'complete'
                if cover:
                    committed = cover_checkpoint(saved, cover_sha, cover['count'])
                    if committed['base_step'] != base_step or committed['cursor'] <= start_cursor:
                        raise RuntimeError('cover checkpoint changed schedule base or made no progress')
                    remaining = cover['count'] - committed['cursor']
                    row['cover'] = dict(committed, remaining=remaining)
                    status = 'partial' if remaining else 'complete'
                row.update(status=status, end_step=saved['step'], checkpoint_sha256=digest(previous))
                if status == 'partial':
                    break
            partial = any(s['status'] == 'partial' for s in state['stages'])
            state.update(status='partial' if partial else 'trained', checkpoint=str(previous), checkpoint_sha256=digest(previous))
            atomic_json(out / 'run.json', state)
    except BaseException as e:
        state.update(status='failed', error=str(e)); atomic_json(out / 'run.json', state); raise
    print(f"{'partial cover' if state['status'] == 'partial' else 'trained'} checkpoint: {previous}", flush=True)


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
    evaluation_plan(r['evaluation'], expected)
    calibrated = bool(r['evaluation'].get('calibration'))
    result = dict(version=2, checkpoint_sha256=state['checkpoint_sha256'], report_threshold=None if calibrated else r['evaluation']['report_threshold'],
                  scope=('One global cutoff per profile fitted on declared calibration sources; acceptance sources and label groups reported separately. Historical holdouts may have been inspected previously; this does not certify production quality.' if calibrated else
                         'Configured holdouts at a fixed threshold; independent calibration and global optimality are not established.'), profiles={})
    with gpu_lock(a.gpu):
        for name in a.profiles.split(','):
            p = dict(r['predict'], **r['profiles'][name])
            env = environment(r, out / 'inputs/ufsm'); env['UFSM_PREDICT_ARGS'] = shlex.join(flags(p))
            scores = out / f'scores-{name}.json'
            cmd = [sys.executable, out / 'inputs/eval_holdouts.py', state['checkpoint'], '--sources', out / 'inputs/sources.json',
                   '--out', Path(a.predictions).resolve() / out.name / name, '--gpu', a.gpu,
                   '--level', r['evaluation']['level'], '--thresholds', ','.join(map(str, r['evaluation']['thresholds'])), '--scores', scores]
            if r['evaluation'].get('source_levels'):
                cmd += ['--source-levels', json.dumps(r['evaluation']['source_levels'])]
            elapsed = execute(cmd, out / f'eval-{name}.log', env)
            data = json.loads(scores.read_text())['scores']
            if data.keys() != expected: raise RuntimeError('evaluation omitted or added held-out boxes')
            metrics = summarize_scores(data, r['evaluation'])
            times = [s['prediction_seconds'] for s in data.values()]
            seconds = sum(times) if all(t is not None for t in times) else None
            result['profiles'][name] = dict(metrics, arguments=p, total_seconds=elapsed,
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
        prediction=selected['arguments'], threshold=selected.get('threshold', report['report_threshold']), environment=r.get('environment', {}),
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
    devices = p.add_mutually_exclusive_group()
    devices.add_argument('--gpu', default='0')
    devices.add_argument('--gpus', help='comma-separated GPU IDs; batch size is per GPU unless --split z is selected')
    p.add_argument('--split', choices=['0', 'z'], help='split each window along z across exactly two GPUs')
    p.add_argument('--mem', choices=['auto', 'auto16', 'default', 'wide'], help='override memory mode for every training stage')
    p.add_argument('--resume'); p.add_argument('--trial-seconds', type=float, default=0)
    p.add_argument('--cover', help='immutable native-volume plan; one batch-1 stage whose steps equal its tile count')
    p = sub.add_parser('evaluate'); p.add_argument('run'); p.add_argument('--gpu', default='0')
    p.add_argument('--profiles', default='matched,fp4,fp8,fp16'); p.add_argument('--predictions', required=True)
    p = sub.add_parser('export'); p.add_argument('run'); p.add_argument('--profile', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('predict'); p.add_argument('bundle'); p.add_argument('--root', required=True); p.add_argument('--ct', required=True)
    p.add_argument('--um', required=True); p.add_argument('--out', required=True); p.add_argument('--box'); p.add_argument('--axis')
    p.add_argument('--cache'); p.add_argument('--gpu', default='0'); p.add_argument('--level', default='0'); p.add_argument('--levels', default='4')
    a = parser.parse_args()
    if a.command == 'train' and a.gpus is not None:
        try:
            a.gpu = ','.join(map(str, gpu_devices(a.gpus)))
        except ValueError as e:
            parser.error(str(e))
    if hasattr(a, 'gpu') and not (a.command == 'train' and a.gpus is not None):
        try:
            devices = gpu_devices(a.gpu)
        except ValueError as e:
            parser.error(str(e))
        if len(devices) != 1:
            parser.error('--gpu must select one nonnegative device; use train --gpus for multiple GPUs')
        a.gpu = str(devices[0])
    if a.command == 'train':
        # Reject an invalid spatial split before creating the run directory or
        # starting any stage; recipes may select it instead of the CLI override.
        split = a.split if a.split is not None else recipe(a.recipe)['train'].get('split', '0')
        if split == 'z' and len(gpu_devices(a.gpu)) != 2:
            parser.error('--split z requires exactly two GPUs')
    if hasattr(a, 'trial_seconds') and (not math.isfinite(a.trial_seconds) or a.trial_seconds < 0): parser.error('invalid trial budget')
    if a.command == 'plan': print(json.dumps(recipe(a.recipe), indent=2))
    else: {'train': train, 'evaluate': evaluate, 'export': export, 'predict': predict}[a.command](a)


if __name__ == '__main__':
    main()
