"""Production finite-cover freezing and continuation; fake trainer, no GPU work."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('production_cover_runner', ROOT / 'tools/production.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
builder = runner.training_cover


def checkpoint(path, step, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('UFSM' + json.dumps({'step': step, 'extra': extra or {'runtime': {'version': 1}}}) + '\n')
    return path


def group(um, path):
    return {'attributes': {'ome': {'multiscales': [{'datasets': [
        {'path': path, 'coordinateTransformations': [{'type': 'scale', 'scale': [um] * 3}]}]}]}}}


class ProductionCover(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        for directory in ('ct', 'ct/0', 'labels', 'labels/coarse'):
            (self.path / directory).mkdir(parents=True, exist_ok=True)
        (self.path / 'ct/zarr.json').write_text(json.dumps(group(1, '0')))
        (self.path / 'ct/0/zarr.json').write_text(json.dumps({'shape': [16] * 3}))
        (self.path / 'labels/zarr.json').write_text(json.dumps(group(4.8, 'coarse')))
        (self.path / 'labels/coarse/zarr.json').write_text(json.dumps({
            'shape': [8] * 3, 'attributes': {'ufsm': {'content': 'sampling occupancy'}}}))
        (self.path / 'labels/provenance.json').write_text(json.dumps(
            {'ct': 'ct', 'native_um': 2.4, 'native_shape_zyx': [16] * 3}))
        axis = self.path / 'axis.json'; axis.write_text('{"control_points":[]}')
        self.source = {'name': 'fixture', 'root': str(self.path), 'ct': 'ct', 'um': 2.4,
                       'targets': {'recto': {'root': str(self.path / 'labels'), 'group': '.', 'encoding': 'binary'}},
                       'axis': str(axis), 'holdout': [4, 4, 4, 6, 6, 6]}
        self.sources = self.path / 'sources.json'
        self.sources.write_text(json.dumps({'sources': [self.source]}))
        occupancy = self.path / 'occupancy.raw'; occupancy.write_bytes(bytes([255] * (8 ** 3)))
        self.plan = builder.build(self.sources, 4, 2, occupancy)
        self.cover = self.path / 'cover.json'; self.cover.write_text(json.dumps(self.plan))
        self.cover_sha = runner.digest(self.cover)
        self.binary = self.path / 'fake-ufsm'; self.binary.write_text('not executed')
        self.r = json.loads((ROOT / 'configs/production-candidate.json').read_text())
        self.r['sources'] = str(self.sources)
        self.r['train'].update(cover=str(self.cover), split='z')
        self.r['stages'] = [{'name': 'cover', 'P': 4, 'B': 1, 'steps': self.plan['count'], 'warmup': 5}]
        self.r['evaluation'] = {'level': 0, 'thresholds': [0.5], 'report_threshold': 0.5,
                                'groups': {'dense': ['fixture']}}
        self.recipe = self.path / 'recipe.json'
        self.calls, self.updates, self.saved_override = [], None, None
        self.lease_active, self.leases = False, []

        @contextlib.contextmanager
        def lease(devices):
            self.leases.append(devices); self.lease_active = True
            try:
                yield
            finally:
                self.lease_active = False

        self.lease_mock = mock.patch.object(runner, 'gpu_lock', lease)
        self.execute_mock = mock.patch.object(runner, 'execute', self.execute)
        self.lease_mock.start(); self.addCleanup(self.lease_mock.stop)
        self.execute_mock.start(); self.addCleanup(self.execute_mock.stop)

    def args(self, name='run', resume=None, trial=0, gpu='0,1', cover=None):
        self.recipe.write_text(json.dumps(self.r))
        return SimpleNamespace(recipe=str(self.recipe), sources=None, binary=str(self.binary),
                               out=str(self.path / name), split=None, mem=None, resume=resume,
                               trial_seconds=trial, gpu=gpu, cover=cover)

    def execute(self, cmd, log, env):
        self.assertTrue(self.lease_active, 'GPU lease must cover the whole trainer execution')
        strings = list(map(str, cmd))
        opts = {part[2:]: strings[i + 1] for i, part in enumerate(strings) if part.startswith('--')}
        self.calls.append(opts)
        resumed = runner.header(opts['resume']) if 'resume' in opts else {'step': 0}
        final_step = int(opts['steps'])
        step = final_step if self.updates is None else min(final_step, resumed['step'] + self.updates)
        extra = {'runtime': {'version': 1}}
        if 'cover' in opts:
            plan = json.loads(Path(opts['cover']).read_text())
            base = int(opts['schedule-start'])
            extra['cover'] = dict(sha256=opts['cover-sha256'], count=plan['count'],
                                  cursor=step - base, base_step=base)
        if self.saved_override:
            self.saved_override(extra)
        checkpoint(Path(opts['out']) / 'last.ckpt', step, extra)
        return 1.25

    def state(self, name='run'):
        return json.loads((self.path / name / 'run.json').read_text())

    def test_warm_start_freezes_cover_preserves_optimizer_step_and_complete_budget(self):
        prior = checkpoint(self.path / 'prior.ckpt', 100)
        runner.train(self.args(resume=str(prior)))
        opts = self.calls[0]
        self.assertEqual(int(opts['steps']), 100 + self.plan['count'])
        self.assertEqual(int(opts['schedule-start']), 100)
        self.assertNotIn('finetune', opts)
        self.assertEqual(opts['cover-sha256'], self.cover_sha)
        self.assertEqual(Path(opts['cover']).read_bytes(), self.cover.read_bytes())
        state = self.state()
        self.assertEqual(state['inputs']['cover.json'], self.cover_sha)
        self.assertEqual(state['status'], 'trained')
        self.assertEqual(state['stages'][0]['status'], 'complete')
        self.assertEqual(state['stages'][0]['cover']['cursor'], self.plan['count'])
        self.assertEqual(state['stages'][0]['cover']['remaining'], 0)
        self.assertEqual(state['stages'][0]['effective_batch'], 1)
        self.assertEqual(self.leases, ['0,1'])
        self.assertFalse(self.lease_active)
        frozen = json.loads((self.path / 'run/inputs/sources.json').read_text())
        self.assertNotEqual(frozen['sources'][0]['axis'], self.source['axis'])
        self.assertNotEqual(runner.digest(self.path / 'run/inputs/sources.json'),
                            self.plan['binding']['source_config_sha256'])

    def test_trial_is_partial_and_resume_keeps_original_pass_schedule(self):
        prior = checkpoint(self.path / 'prior.ckpt', 100)
        self.updates = 2
        runner.train(self.args('trial', resume=str(prior), trial=10))
        opts = self.calls[0]
        self.assertEqual(opts['limit-seconds'], '10')
        self.assertNotIn('seconds', opts)
        self.assertNotIn('warmup-seconds', opts)
        trial = self.state('trial')
        self.assertEqual(trial['status'], 'partial')
        self.assertEqual(trial['stages'][0]['status'], 'partial')
        self.assertEqual(trial['stages'][0]['cover']['cursor'], 2)
        self.assertEqual(trial['stages'][0]['cover']['remaining'], self.plan['count'] - 2)
        with self.assertRaisesRegex(ValueError, 'not completed'):
            runner.run_inputs(self.path / 'trial')
        self.updates = None
        runner.train(self.args('continued', resume=trial['checkpoint']))
        opts = self.calls[-1]
        self.assertEqual(int(opts['schedule-start']), 100)
        self.assertEqual(int(opts['steps']), 100 + self.plan['count'])
        self.assertEqual(self.state('continued')['status'], 'trained')
        self.assertEqual(self.state('continued')['stages'][0]['start_step'], 102)

    def test_continuation_rejects_hash_count_cursor_and_already_complete(self):
        good = dict(sha256=self.cover_sha, count=self.plan['count'], cursor=2, base_step=100)
        invalid = [dict(good, sha256='0' * 64), dict(good, count=self.plan['count'] + 1),
                   dict(good, cursor=3), dict(good, cursor=-1), dict(good, base_step=-1)]
        for index, value in enumerate(invalid):
            prior = checkpoint(self.path / f'prior-{index}.ckpt', 102, {'runtime': {'version': 1}, 'cover': value})
            with self.assertRaisesRegex(ValueError, 'committed cursor'):
                runner.train(self.args(f'bad-{index}', resume=str(prior)))
            self.assertEqual(self.state(f'bad-{index}')['status'], 'failed')
        value = dict(good, cursor=self.plan['count'])
        prior = checkpoint(self.path / 'complete.ckpt', 100 + self.plan['count'],
                           {'runtime': {'version': 1}, 'cover': value})
        with self.assertRaisesRegex(ValueError, 'already completed'):
            runner.train(self.args('already', resume=str(prior)))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.lease_active)

    def test_changed_labels_and_bad_tile_structure_fail_before_execution(self):
        changed = dict(self.plan); changed['tiles'] = list(self.plan['tiles'])
        changed['tiles'][0] = changed['tiles'][1]
        self.cover.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            runner.train(self.args('duplicate'))
        self.cover.write_text(json.dumps(self.plan))
        path = self.path / 'labels/provenance.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            runner.train(self.args('changed'))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.leases, [])

    def test_incompatible_dimensions_schedule_and_parallelism_fail(self):
        cases = [('P', 8, 'stage'), ('B', 2, 'stage'), ('steps', self.plan['count'] - 1, 'stage'),
                 ('finetune', 1, 'forbids'), ('overfit', 1, 'forbids'), ('seconds', 10, 'forbids'),
                 ('split', '0', 'requires spatial')]
        original = dict(self.r['stages'][0]); train_original = dict(self.r['train'])
        for index, (key, value, message) in enumerate(cases):
            self.r['stages'][0] = dict(original); self.r['train'] = dict(train_original)
            if key in ('P', 'B', 'steps'):
                self.r['stages'][0][key] = value
            else:
                self.r['train'][key] = value
            with self.assertRaisesRegex(ValueError, message):
                runner.train(self.args(f'incompatible-{index}'))
        self.assertEqual(self.calls, [])

    def test_cli_cover_override_and_post_checkpoint_integrity(self):
        self.r['train']['cover'] = '/missing/recipe/cover.json'
        runner.train(self.args('override', cover=str(self.cover)))
        self.assertEqual(self.state('override')['status'], 'trained')
        self.saved_override = lambda extra: extra['cover'].update(sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'committed cursor'):
            runner.train(self.args('bad-checkpoint', cover=str(self.cover)))
        self.assertEqual(self.state('bad-checkpoint')['status'], 'failed')

    def test_noncover_stages_and_time_trials_keep_existing_behavior(self):
        self.r['train'].pop('cover'); self.r['train'].pop('split')
        self.r['stages'] = [{'name': 'first', 'P': 4, 'B': 1, 'steps': 3, 'warmup': 1},
                            {'name': 'second', 'P': 4, 'B': 1, 'steps': 5, 'warmup': 1}]
        prior = checkpoint(self.path / 'prior.ckpt', 100)
        runner.train(self.args('ordinary', resume=str(prior), trial=20, gpu='0'))
        self.assertEqual([int(o['steps']) for o in self.calls], [103, 108])
        self.assertEqual([int(o['schedule-start']) for o in self.calls], [100, 103])
        self.assertTrue(all(o['seconds'] == '20' and o['warmup-seconds'] == '1.0' for o in self.calls))
        self.assertTrue(all('cover' not in o and 'limit-seconds' not in o for o in self.calls))
        self.assertEqual(self.state('ordinary')['status'], 'trained')


if __name__ == '__main__':
    unittest.main()
