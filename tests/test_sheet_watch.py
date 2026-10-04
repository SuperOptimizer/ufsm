#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest
import argparse
from contextlib import nullcontext
import json
import tempfile
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sheet_diagnostics import binary_scores
from sheet_watch import interval_deadline,trend,watch


class WatchTests(unittest.TestCase):
    def test_hourly_controller_resumes_state_and_survives_evaluation_failure(self):
        for fail_evaluation in (False,True):
            with self.subTest(fail_evaluation=fail_evaluation), tempfile.TemporaryDirectory() as tmp:
                repo=Path(tmp);root=repo/'job';run=root/'full';inputs=run/'inputs';inputs.mkdir(parents=True)
                (inputs/'ufsm').write_text('fixture'); (inputs/'sources.json').write_text('{}')
                original=dict(step=5,extra=dict(cover=dict(base_step=2,cursor=3,count=3,sha256='old')))
                def checkpoint(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text('UFSM'+json.dumps(obj)+'\n')
                checkpoint(inputs/'resume.ckpt',original)
                state=dict(status='prepared',updates=100,inputs={'cover.json':'new'},command=[str(inputs/'ufsm'),'train',str(inputs/'sources.json'),
                    '--out',str(run/'model'),'--resume',str(inputs/'resume.ckpt'),'--gpus','0,1','--cover-extend-from','previous'])
                (run/'state.json').write_text(json.dumps(state))
                progress=dict(budget_seconds=20,evaluation_seconds=10,development_box=[0,0,0,16,16,16],truth='truth',donor_step=5,tool_sha256={},preview_dir=str(root/'previews'))
                (root/'testing.json').write_text(json.dumps(progress))
                clock=[100.];commands=[]
                class Process:
                    pid=1234
                    def __init__(self,command,**kw):self.command=command;commands.append(command)
                    def wait(self):
                        command=self.command
                        if command[1]=='train':
                            saved=json.loads(Path(command[command.index('--resume')+1]).read_text()[4:])
                            saved['step']+=2;saved['extra']['cover'].update(cursor=saved['step']-2,count=100,sha256='new')
                            checkpoint(run/'model/last.ckpt',saved)
                            clock[0]+=float(command[command.index('--limit-seconds')+1])+.1
                        elif 'predict' in command:Path(command[command.index('--out')+1]).mkdir()
                        elif 'evaluate' in command:
                            path=Path(command[command.index('--out')+1])
                            if fail_evaluation and path.parent.name=='01':return 1
                            path.write_text(json.dumps(dict(supported_coverage=0)))
                        return 0
                def diagnostic(run,prediction,checkpoint,output,box,**kw):
                    image=output/'preview.png';image.write_bytes(b'fixture image');index=int(output.name)
                    return dict(best_binary_f1=.1+index*.01,roc_auc=.5,best_cutoff=.2,image=str(image))
                with patch('sheet_watch.REPO',repo),patch('sheet_watch.verify_inputs'),patch('sheet_watch.gpu_lock',side_effect=lambda *_:nullcontext()),\
                     patch('sheet_watch.subprocess.Popen',Process),patch('sheet_watch.time.time',side_effect=lambda:clock[0]),\
                     patch('sheet_watch.signal.signal'),patch('sheet_watch.diagnose',side_effect=diagnostic):
                    watch(argparse.Namespace(root=root))
                result=json.loads((root/'testing.json').read_text())
                self.assertEqual(result['status'],'24-hour run complete')
                self.assertEqual(result['deadline'],120)
                self.assertEqual(result['checkpoint_step'],9)
                training=[c for c in commands if c[1]=='train']
                self.assertEqual(len(training),2)
                self.assertIn('--cover-extend-from',training[0]);self.assertNotIn('--cover-extend-from',training[1])
                self.assertEqual(training[1][training[1].index('--resume')+1],str(run/'model/last.ckpt'))
                self.assertAlmostEqual(float(training[1][training[1].index('--schedule-elapsed')+1]),10.1)
                self.assertEqual(result['best_hour'],2)
                self.assertEqual(bool(result.get('evaluation_errors')),fail_evaluation)

    def test_uninformative_prediction_has_random_auc_and_foreground_floor(self):
        positive=np.zeros(256,np.int64);negative=positive.copy()
        positive[50]=5;negative[50]=95
        result=binary_scores(positive,negative)
        self.assertEqual(result['roc_auc'],.5)
        self.assertAlmostEqual(result['best_binary_f1'],10/105)
        self.assertEqual(result['best_binary_f1'],result['constant_foreground_binary_f1'])

    def test_true_discrimination_is_not_just_a_cutoff_shift(self):
        positive=np.zeros(256,np.int64);negative=positive.copy()
        positive[200]=5;negative[20]=95
        result=binary_scores(positive,negative)
        self.assertEqual(result['roc_auc'],1)
        self.assertEqual(result['best_binary_f1'],1)
        self.assertGreater(result['mean_probability_surface'],result['mean_probability_background'])

    def test_deadlines_do_not_drift_with_evaluation_time(self):
        self.assertEqual(interval_deadline(100,86400,3600,1),3700)
        self.assertEqual(interval_deadline(100,86400,3600,2),7300)
        self.assertEqual(interval_deadline(100,86400,3600,24),86500)
        self.assertEqual(interval_deadline(100,25,10,3),125)

    def test_stalls_and_regressions_are_reported(self):
        self.assertEqual(trend(.12,.1),'improved')
        self.assertEqual(trend(.10,.12),'regressed')
        self.assertEqual(trend(.1001,.1),'flat')


if __name__=='__main__':unittest.main()
