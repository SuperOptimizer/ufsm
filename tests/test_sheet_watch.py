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
from sheet_watch import cover_continuation,interval_deadline,trend,watch


class WatchTests(unittest.TestCase):
    def test_completed_pass_rolls_over_inside_the_hour_without_shortening_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);root=repo/'job';run=root/'full';inputs=run/'inputs';inputs.mkdir(parents=True)
            def checkpoint(path,step,base,cursor,count,sha):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('UFSM'+json.dumps(dict(step=step,extra=dict(cover=dict(base_step=base,cursor=cursor,count=count,sha256=sha))))+'\n')
            checkpoint(inputs/'resume.ckpt',3,2,1,2,'first')
            (inputs/'sources.json').write_text(json.dumps(dict(sources=[dict(root='ct',ct='ct',um=2.4)])))
            passes=[]
            for name,count,seed in [('first',2,2),('second',2,3)]:
                path=inputs/f'{name}.json';path.write_text(json.dumps(dict(count=count,seed=seed,tiles=[[0,0,0,0],[0,16,0,0]])))
                passes.append(dict(path=str(path),sha256=name,count=count,seed=seed))
            state=dict(task='surface',updates=2,inputs={'cover.json':'first','first.json':'first','second.json':'second'},
                command=[str(inputs/'ufsm'),'train',str(inputs/'sources.json'),'--out',str(run/'model'),'--resume',str(inputs/'resume.ckpt')])
            (run/'state.json').write_text(json.dumps(state))
            progress=dict(budget_seconds=10,evaluation_seconds=10,development_box=[0,0,0,16,16,16],donor_step=3,
                tool_sha256={},preview_dir=str(root/'previews'),lr_schedule='wall',monitor_predict=dict(window=528),cover_passes=passes)
            (root/'testing.json').write_text(json.dumps(progress));clock=[100.];commands=[]
            class Process:
                pid=1234
                def __init__(self,command,**kw):self.command=command;commands.append(command)
                def wait(self):
                    command=self.command
                    if command[1]=='train':
                        sha=command[command.index('--cover-sha256')+1]
                        if sha=='first':checkpoint(run/'model/last.ckpt',4,2,2,2,sha);clock[0]+=4
                        else:checkpoint(run/'model/last.ckpt',5,4,1,2,sha);clock[0]+=float(command[command.index('--limit-seconds')+1])+.1
                    else:
                        path=Path(command[5]);path.mkdir();(path/'zarr.json').write_text('{}')
                    return 0
            def diagnostic(run,prediction,checkpoint,output,box,**kw):
                path=output/'preview.png';path.write_bytes(b'preview')
                return dict(best_binary_f1=.1,roc_auc=.5,best_cutoff=.2,image=str(path))
            with patch('sheet_watch.REPO',repo),patch('sheet_watch.verify_inputs'),patch('production.validate_cover'),\
                 patch('sheet_watch.gpu_lock',side_effect=lambda *_:nullcontext()),patch('sheet_watch.subprocess.Popen',Process),\
                 patch('sheet_watch.time.time',side_effect=lambda:clock[0]),patch('sheet_watch.signal.signal'),patch('sheet_watch.diagnose',side_effect=diagnostic):
                watch(argparse.Namespace(root=root))
            result=json.loads((root/'testing.json').read_text());training=[c for c in commands if c[1]=='train']
            self.assertEqual(len(training),2);self.assertNotIn('--cover-next-pass',training[0]);self.assertIn('--cover-next-pass',training[1])
            self.assertEqual(result['checkpoint_step'],5);self.assertEqual(result['deadline'],110)
            self.assertEqual(result['cover_pass'],2);self.assertEqual(len(result['completed_cover_passes']),1)
            self.assertEqual(set(result['reports']),{'0','1'})
            self.assertEqual(float(training[1][training[1].index('--schedule-elapsed')+1]),4)
            self.assertEqual(float(training[1][training[1].index('--limit-seconds')+1]),6)

    def test_pass_transition_preserves_step_and_cannot_skip_unfinished_tiles(self):
        passes=[dict(path='first.json',sha256='first',count=3,seed=2),
                dict(path='second.json',sha256='second',count=3,seed=3)]
        command=['ufsm','train','sources','--resume','last.ckpt','--steps','103','--schedule-start','100']
        saved=dict(step=102,extra=dict(cover=dict(base_step=100,cursor=2,count=3,sha256='first')))
        cmd,sha,index=cover_continuation(command,saved,passes)
        self.assertEqual((sha,index),('first',1));self.assertNotIn('--cover-next-pass',cmd)
        saved['step']=103;saved['extra']['cover']['cursor']=3
        cmd,sha,index=cover_continuation(cmd,saved,passes)
        self.assertEqual((sha,index),('second',2));self.assertIn('--cover-next-pass',cmd)
        self.assertEqual(cmd[cmd.index('--steps')+1],'106');self.assertEqual(cmd[cmd.index('--schedule-start')+1],'103')
        saved=dict(step=104,extra=dict(cover=dict(base_step=103,cursor=1,count=3,sha256='second')))
        cmd,sha,index=cover_continuation(cmd,saved,passes)
        self.assertNotIn('--cover-next-pass',cmd);self.assertEqual(cmd[cmd.index('--schedule-start')+1],'103')
        saved['step']=106;saved['extra']['cover']['cursor']=3
        self.assertIsNone(cover_continuation(cmd,saved,passes))
        saved['extra']['cover']['sha256']='unfrozen'
        with self.assertRaises(RuntimeError):cover_continuation(cmd,saved,passes)

    def test_hourly_controller_resumes_state_and_survives_evaluation_failure(self):
        for surface_only,fail_evaluation in ((False,False),(False,True),(True,False),(True,True)):
            with self.subTest(surface_only=surface_only,fail_evaluation=fail_evaluation), tempfile.TemporaryDirectory() as tmp:
                repo=Path(tmp);root=repo/'job';run=root/'full';inputs=run/'inputs';inputs.mkdir(parents=True)
                (inputs/'ufsm').write_text('fixture'); (inputs/'sources.json').write_text(json.dumps(dict(sources=[dict(root='ct-store',ct='ct',um=2.4)])))
                original=dict(step=5,extra=dict(cover=dict(base_step=2,cursor=3,count=3,sha256='old')))
                def checkpoint(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text('UFSM'+json.dumps(obj)+'\n')
                checkpoint(inputs/'resume.ckpt',original)
                state=dict(status='prepared',updates=100,inputs={'cover.json':'new'},command=[str(inputs/'ufsm'),'train',str(inputs/'sources.json'),
                    '--out',str(run/'model'),'--resume',str(inputs/'resume.ckpt'),'--gpus','0,1','--cover-extend-from','previous',
                    '--P','704','--soft','3','--warmup','500','--schedule-start','2'])
                if surface_only:state['task']='surface'
                (run/'state.json').write_text(json.dumps(state))
                progress=dict(budget_seconds=20,evaluation_seconds=10,development_box=[0,0,0,16,16,16],truth='truth',donor_step=5,tool_sha256={},preview_dir=str(root/'previews'))
                if surface_only:progress.update(lr_schedule='steps',environment={'UFSM_RC_KEEP_COARSE':'1'},monitor_predict=dict(window=528,halo=8,shard=512))
                (root/'testing.json').write_text(json.dumps(progress))
                clock=[100.];commands=[]
                class Process:
                    pid=1234
                    def __init__(self,command,**kw):
                        self.command=command;commands.append(command)
                        if surface_only:assert kw['env']['UFSM_RC_KEEP_COARSE']=='1'
                    def wait(self):
                        command=self.command
                        if command[1]=='train':
                            saved=json.loads(Path(command[command.index('--resume')+1]).read_text()[4:])
                            saved['step']+=2;saved['extra']['cover'].update(cursor=saved['step']-2,count=100,sha256='new')
                            checkpoint(run/'model/last.ckpt',saved)
                            clock[0]+=float(command[command.index('--limit-seconds')+1])+.1
                        elif command[1]=='predict':
                            path=Path(command[5]);path.mkdir()
                            if fail_evaluation and path.parent.name=='01':return 1
                            (path/'zarr.json').write_text('{}')
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
                if surface_only:
                    for command in training:
                        self.assertNotIn('--schedule-seconds',command);self.assertNotIn('--schedule-elapsed',command)
                        self.assertEqual(command[command.index('--schedule-start')+1],'2')
                        self.assertEqual(command[command.index('--warmup')+1],'500')
                        self.assertEqual(command[command.index('--P')+1],'704')
                    self.assertFalse(any('extract' in c or 'evaluate' in c for c in commands))
                    self.assertIsNone(result['reports']['2']['geometry'])
                    self.assertTrue((repo/'runs/active-production.json').exists())
                else:self.assertAlmostEqual(float(training[1][training[1].index('--schedule-elapsed')+1]),10.1)
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
