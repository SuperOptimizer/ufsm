"""Extension freezes unchanged state and queues repeat passes; no GPU work."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import extend_surface_training as runner
from production import atomic_json,digest


class ExtendTests(unittest.TestCase):
    def test_extension_preserves_checkpoint_payload_targets_history_and_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);(repo/'runs').mkdir();old=repo/'jobs/original';inputs=old/'full/inputs';inputs.mkdir(parents=True)
            atomic_json(inputs/'sources.json',dict(sources=[dict(axis=str(inputs/'axis-0.json'))]))
            atomic_json(inputs/'evaluation-sources.json',dict(sources=[dict(axis=str(inputs/'axis-0.json'))]))
            atomic_json(inputs/'axis-0.json',{})
            atomic_json(inputs/'recipe.json',dict(train=dict(soft=2,erode=1,cover=str(inputs/'cover.json'))))
            plan=dict(seed=2,count=2,tiles=[[0,0,0,0],[0,16,0,0]])
            atomic_json(inputs/'cover.json',plan);(inputs/'ufsm').write_bytes(b'old trainer')
            model=old/'full/model';model.mkdir();sha=digest(inputs/'cover.json')
            saved=dict(step=3,extra=dict(cover=dict(base_step=2,cursor=1,count=2,sha256=sha),
                target=dict(erode_native_voxels=1,soft_sigma=2)))
            payload=b'UFSM'+json.dumps(saved).encode()+b'\n'+bytes(range(256))*100
            (model/'last.ckpt').write_bytes(payload);(model/'precision.txt').write_text('frozen precision')
            (model/'log.csv').write_text('step,loss\n3,.8\n');(inputs/'resume.ckpt').write_bytes(payload)
            state=dict(task='surface',training_code_commit='old',inputs={p.name:digest(p) for p in inputs.iterdir()},
                command=[str(inputs/'ufsm'),'train',str(inputs/'sources.json'),'--out',str(model),
                    '--cover',str(inputs/'cover.json'),'--resume',str(inputs/'resume.ckpt')])
            atomic_json(old/'full/state.json',state);(old/'hours/01').mkdir(parents=True)
            (old/'evaluation-cache').mkdir();(old/'best-development.ckpt').write_bytes(b'best')
            tool=repo/'watch.py';tool.write_text('frozen tool')
            progress=dict(root=str(old),unit='old.service',deadline=200,started_at=100,reports={'1':{}},
                tool_sha256={str(tool):digest(tool)},training_soft_sigma=2,training_erosion=1)
            atomic_json(old/'testing.json',progress);atomic_json(repo/'runs/active-production.json',progress)
            binary=repo/'new-trainer';binary.write_bytes(b'new trainer')
            args=SimpleNamespace(root=None,hours=24,binary=str(binary),binary_source_patch=None,training_code_commit='new')
            calls=[]
            def call(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(stdout='--cover-next-pass --schedule-seconds',stderr='',returncode=2)
            with patch.object(runner,'REPO',repo),patch.object(runner,'validate_cover'),\
                 patch.object(runner.subprocess,'run',side_effect=call),patch.object(runner.time,'time',return_value=150):
                runner.extend(args)
            replacement=Path(json.loads((old/'testing.json').read_text())['replacement_root'])
            new=json.loads((replacement/'testing.json').read_text())
            self.assertEqual(new['deadline'],200+86400);self.assertEqual(new['budget_seconds'],100+86400)
            self.assertEqual(new['training_soft_sigma'],2);self.assertEqual(new['training_erosion'],1)
            self.assertEqual(new['lr_schedule'],'wall');self.assertEqual(new['first_interval'],2)
            self.assertEqual((replacement/'full/model/last.ckpt').read_bytes(),payload)
            self.assertEqual((replacement/'full/inputs/resume.ckpt').read_bytes(),payload)
            self.assertEqual((replacement/'best-development.ckpt').read_bytes(),b'best')
            self.assertEqual(len(new['cover_passes']),3)
            for p in new['cover_passes']:
                frozen=json.loads(Path(p['path']).read_text())
                self.assertEqual({tuple(t) for t in frozen['tiles']},{tuple(t) for t in plan['tiles']})
                self.assertEqual(digest(p['path']),p['sha256'])
            self.assertEqual(calls[1][:3],['systemctl','--user','stop'])
            self.assertEqual(calls[-1][0],'systemd-run')


if __name__=='__main__':unittest.main()
