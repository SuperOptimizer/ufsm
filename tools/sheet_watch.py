#!/usr/bin/env python3
"""Continue a frozen winding run for a wall-time budget with hourly evaluation."""
import argparse
import csv
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from production import gpu_lock,header
from sheet_geometry import atomic_json,digest
from sheet_pipeline import verify_inputs
from sheet_diagnostics import diagnose

REPO=Path(__file__).resolve().parents[1]


def trend(current,previous,tolerance=.001):
    delta=current-previous
    return 'improved' if delta>tolerance else 'regressed' if delta<-tolerance else 'flat'


def interval_deadline(start,seconds,interval,index):
    return min(start+seconds,start+interval*index)


def watch(args):
    root=Path(args.root).resolve(); run=root/'full'; pointer=REPO/'runs/active-sheet-testing.json'
    state=json.loads((run/'state.json').read_text()); progress=json.loads((root/'testing.json').read_text())
    seconds=progress['budget_seconds']; interval=progress['evaluation_seconds']
    box=progress['development_box']; binary=run/'inputs/ufsm'; tool=REPO/'tools/sheet_pipeline.py'
    worker=None; stopping=False
    env={k:v for k,v in os.environ.items() if not k.startswith('UFSM_')}
    def publish(**kw):
        progress.update(kw); atomic_json(root/'testing.json',progress); atomic_json(pointer,progress)
    def interrupted(sig,frame):
        nonlocal stopping
        stopping=True
        if worker is not None and worker.poll() is None:
            try: os.killpg(worker.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    def execute(command,log):
        nonlocal worker
        if stopping: raise InterruptedError('stopped by signal')
        for path,sha in progress['tool_sha256'].items():
            if digest(path)!=sha: raise RuntimeError('frozen experiment tool changed: '+path)
        with Path(log).open('a') as stream:
            worker=subprocess.Popen(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            publish(worker_pid=worker.pid)
            code=worker.wait()
        worker=None
        if stopping: raise InterruptedError('stopped by signal')
        if code: raise subprocess.CalledProcessError(code,command)
    def setflag(command,flag,value):
        if flag in command: command[command.index(flag)+1]=str(value)
        else: command.extend([flag,str(value)])
    def snapshot(index,checkpoint):
        target=root/'hours'/f'{index:02d}'; target.mkdir(parents=True,exist_ok=True)
        saved=header(checkpoint); model=target/'model'
        if not model.exists():
            model.mkdir(); (target/'inputs').symlink_to(run/'inputs',target_is_directory=True)
            shutil.copyfile(checkpoint,model/'last.ckpt')
            precision=run/'model/precision.txt'
            if precision.exists(): shutil.copyfile(precision,model/'precision.txt')
            frozen=dict(state,status='trained',checkpoint_sha256=digest(model/'last.ckpt'),
                updates=saved['step']-progress['donor_step'],planned_cover_count=state['updates'],evaluation_snapshot_step=saved['step'])
            atomic_json(target/'state.json',frozen)
        return target
    def evaluate(index,checkpoint):
        target=snapshot(index,checkpoint); saved=header(target/'model/last.ckpt'); prediction=target/'prediction'
        pipeline=[sys.executable,str(tool)]
        publish(status='evaluating',phase='predict',hour=index,checkpoint_step=saved['step'])
        if not prediction.exists(): execute(pipeline+['predict','--run',target,'--box',','.join(map(str,box)),'--out',prediction,'--gpu','0'],root/'evaluation.log')
        evidence=target/'evidence.npz'
        publish(phase='geometry')
        execute(pipeline+['extract','--prediction',prediction,'--out',evidence,'--binary',binary,'--threshold','0.3'],root/'evaluation.log')
        execute(pipeline+['evaluate','--truth',progress['truth'],'--evidence',evidence,'--split','development','--out',target/'geometry.json'],root/'evaluation.log')
        baseline_raw=root/'hours/00/probability.raw'
        measured=diagnose(run,prediction,target/'model/last.ckpt',target,box,
            baseline_raw=baseline_raw if index else None,baseline_step=progress.get('baseline_step',progress['donor_step']))
        preview=Path(progress.get('preview_dir','/tmp'))/f'paris4-sheet24-hour-{index:02d}.png'
        preview.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(measured['image'],preview)
        measured.update(step=saved['step'],hour=index,image=str(preview),measured_at=time.time(),
            training_soft_sigma=saved.get('extra',{}).get('augmentation',{}).get('soft_sigma'))
        log=run/'model/log.csv'; validation={}
        if log.exists():
            rows=[r for r in csv.DictReader(log.open()) if r.get('val_loss')]
            if rows: validation={k:float(rows[-1][k]) for k in ('val_loss','val_bce','val_dice')}
        prior=progress.get('reports',{}); previous=max((int(k) for k in prior),default=None)
        measured['trend']=trend(measured['best_binary_f1'],prior[str(previous)]['diagnostic']['best_binary_f1']) if previous is not None else 'baseline'
        report=dict(diagnostic=measured,geometry=json.loads((target/'geometry.json').read_text()),validation=validation)
        atomic_json(target/'report.json',report); prior[str(index)]=report
        best=max(prior,key=lambda k:prior[k]['diagnostic']['best_binary_f1']); best_data=prior[best]['diagnostic']
        best_path=root/'hours'/f'{int(best):02d}'/'model/last.ckpt'
        shutil.copyfile(best_path,root/'best-development.ckpt')
        with (root/'hourly.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['hour','step','training_soft_sigma','best_binary_f1','roc_auc','best_cutoff','trend','val_bce','val_dice','supported_coverage'])
            writer.writeheader()
            for key in sorted(prior,key=int):
                d=prior[key]['diagnostic']; r={k:d.get(k) for k in ('hour','step','training_soft_sigma','best_binary_f1','roc_auc','best_cutoff','trend')}
                r.update({k:prior[key]['validation'].get(k) for k in ('val_bce','val_dice')});r['supported_coverage']=prior[key]['geometry']['supported_coverage'];writer.writerow(r)
        summary=f'Hour {index}: step {saved["step"]:,}, best-cutoff development F1 {measured["best_binary_f1"]:.4f} ({measured["trend"]}).\nBest checkpoint: hour {best}, F1 {best_data["best_binary_f1"]:.4f}.\nFixed-cutoff supported coverage: {report["geometry"]["supported_coverage"]:.4%}.\nPreview: {preview}\n'
        (root/'hourly-status.txt').write_text(summary)
        publish(reports=prior,last_evaluation_hour=index,last_evaluation_step=saved['step'],last_best_binary_f1=measured['best_binary_f1'],
            trend=measured['trend'],best_hour=int(best),best_binary_f1=best_data['best_binary_f1'],image=str(preview))
        print(summary,flush=True)
    try:
        verify_inputs(run,state)
        donor=run/'inputs/resume.ckpt'
        if '0' not in progress.get('reports',{}): evaluate(0,donor)
        if 'started_at' not in progress:
            start=time.time(); publish(started_at=start,deadline=start+seconds,finish_utc=datetime.fromtimestamp(start+seconds,timezone.utc).isoformat())
        start=progress['started_at']; end=progress['deadline']
        checks=math.ceil(seconds/interval)
        for index in range(1,checks+1):
            if str(index) in progress.get('reports',{}): continue
            checkpoint=run/'model/last.ckpt'; current=header(checkpoint if checkpoint.exists() else donor)
            remaining=interval_deadline(start,seconds,interval,index)-time.time()
            if remaining>0:
                command=list(state['command'])
                if checkpoint.exists():
                    setflag(command,'--resume',checkpoint)
                    if '--cover-extend-from' in command:
                        i=command.index('--cover-extend-from');del command[i:i+2]
                for flag in ('--stop-at','--seconds'):
                    if flag in command:
                        i=command.index(flag);del command[i:i+2]
                elapsed=max(0,time.time()-start)
                setflag(command,'--schedule-seconds',seconds);setflag(command,'--schedule-elapsed',elapsed)
                setflag(command,'--warmup-seconds',0);setflag(command,'--limit-seconds',remaining)
                state['status']='training';atomic_json(run/'state.json',state)
                publish(status='training',phase='train',hour=index,checkpoint_step=current['step'],next_evaluation_at=interval_deadline(start,seconds,interval,index),remaining_seconds=max(0,end-time.time()))
                with gpu_lock('0,1'): execute(command,run/'train.log')
            if not checkpoint.exists(): raise RuntimeError('no committed continuation checkpoint')
            saved=header(checkpoint)
            if saved['extra']['cover']['sha256']!=state['inputs']['cover.json'] or saved['step']!=saved['extra']['cover']['base_step']+saved['extra']['cover']['cursor']:
                raise RuntimeError('checkpoint coverage identity/cursor changed')
            if saved['extra']['cover']['cursor']==saved['extra']['cover']['count'] and time.time()<end-60:
                raise RuntimeError('training cover exhausted before requested wall-time budget')
            state.update(status='interrupted',checkpoint_sha256=digest(checkpoint),checkpoint_step=saved['step']);atomic_json(run/'state.json',state)
            try: evaluate(index,checkpoint)
            except subprocess.CalledProcessError as error:
                errors=progress.get('evaluation_errors',[]);errors.append(dict(hour=index,step=saved['step'],error=str(error)))
                publish(evaluation_errors=errors,last_evaluation_error=str(error));print('evaluation failed; retaining checkpoint and continuing training',error,flush=True)
            if time.time()>=end: break
        state.update(status='budget_complete',checkpoint_sha256=digest(run/'model/last.ckpt'));atomic_json(run/'state.json',state)
        publish(status='24-hour run complete',phase='complete',completed_at=time.time(),remaining_seconds=0,checkpoint_step=header(run/'model/last.ckpt')['step'])
    except InterruptedError as error:
        state['status']='interrupted';atomic_json(run/'state.json',state);publish(status='interrupted',error=str(error));raise SystemExit(143)
    except Exception as error:
        publish(status='failed',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);watch(parser.parse_args())
