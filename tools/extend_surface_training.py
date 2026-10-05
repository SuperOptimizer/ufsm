#!/usr/bin/env python3
"""Extend a detached surface run, freezing shuffled repeat passes and its donor."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from production import atomic_json, digest, header, validate_cover
from sheet_pipeline import verify_inputs

REPO=Path(__file__).resolve().parents[1]


def extend(args):
    before=json.loads((REPO/'runs/active-production.json').read_text())
    old=Path(args.root or before['root']).resolve()
    before=json.loads((old/'testing.json').read_text())
    state=json.loads((old/'full/state.json').read_text())
    if state.get('task')!='surface' or not math.isfinite(args.hours) or args.hours<=0:
        raise ValueError('extension requires a surface-only run and a positive number of hours')
    verify_inputs(old/'full',state)
    binary=Path(args.binary).resolve()
    help_result=subprocess.run([binary,'train'],capture_output=True,text=True)
    help_text=help_result.stdout+help_result.stderr
    if '--cover-next-pass' not in help_text or '--schedule-seconds' not in help_text:
        raise ValueError('the frozen trainer needs --cover-next-pass and --schedule-seconds support')
    cfg=json.loads((old/'full/inputs/sources.json').read_text())
    plan=json.loads((old/'full/inputs/cover.json').read_text());validate_cover(plan,cfg)
    patch=Path(args.binary_source_patch).read_bytes() if args.binary_source_patch else None
    deadline=before['deadline']+args.hours*3600
    if deadline<=time.time(): raise ValueError('extended deadline has already passed')

    # SIGTERM saves the last complete model/optimizer update and coverage cursor.
    subprocess.run(['systemctl','--user','stop',before['unit']],check=True)
    progress=json.loads((old/'testing.json').read_text())
    donor=old/'full/model/last.ckpt';saved=header(donor);donor_sha=digest(donor)
    if saved.get('extra',{}).get('sheet'): raise ValueError('cannot extend a winding model as surface-only')
    current=saved['extra']['cover']
    old_passes=progress.get('cover_passes',[dict(path=str(old/'full/inputs/cover.json'),
        sha256=state['inputs']['cover.json'],count=plan['count'],seed=plan['seed'])])
    if not any(p['sha256']==current['sha256'] and p['count']==current['count'] for p in old_passes):
        raise ValueError('donor is not from a frozen coverage pass')
    if saved['step']!=current['base_step']+current['cursor']: raise ValueError('donor coverage cursor changed')

    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=old.parent/f'paris4-surface-extended-{stamp}';run=root/'full';inputs=run/'inputs';model=run/'model'
    inputs.mkdir(parents=True);model.mkdir()
    for path in (old/'full/inputs').iterdir():
        if path.name not in ('resume.ckpt','ufsm'): shutil.copy2(path,inputs/path.name)
    shutil.copy2(binary,inputs/'ufsm');shutil.copyfile(donor,inputs/'resume.ckpt');shutil.copyfile(donor,model/'last.ckpt')
    if patch is not None: (inputs/'training-source.patch').write_bytes(patch)
    for name in ('log.csv','precision.txt'):
        if (old/'full/model'/name).exists(): shutil.copyfile(old/'full/model'/name,model/name)
    for name in ('sources.json','evaluation-sources.json'):
        source_cfg=json.loads((inputs/name).read_text())
        for source in source_cfg['sources']:
            if source.get('axis'): source['axis']=str(inputs/Path(source['axis']).name)
        atomic_json(inputs/name,source_cfg)
    recipe=json.loads((inputs/'recipe.json').read_text());recipe['train']['cover']=str(inputs/'cover.json')
    atomic_json(inputs/'recipe.json',recipe)
    passes=[dict(p,path=str(inputs/Path(p['path']).name)) for p in old_passes]
    # More immutable passes can be queued than consumed. Only committed updates
    # advance a cursor, and the absolute deadline stops partway through a pass.
    for i in range(len(passes),len(passes)+max(2,math.ceil(args.hours/24)*2)):
        repeated=copy.deepcopy(plan);repeated['seed']=plan['seed']+i
        random.Random(repeated['seed']).shuffle(repeated['tiles'])
        path=inputs/f'cover-pass-{i+1:02d}.json';atomic_json(path,repeated)
        validate_cover(repeated,json.loads((inputs/'sources.json').read_text()))
        passes.append(dict(path=str(path),sha256=digest(path),count=repeated['count'],seed=repeated['seed']))
    command=[str(inputs/Path(v).name) if str(v).startswith(str(old/'full/inputs')+'/') else str(v) for v in state['command']]
    command[command.index('--out')+1]=str(model)
    unit=f'ufsm-paris4-surface-extended-{stamp.lower()}.service'
    extension=dict(previous_root=str(old),previous_deadline=before['deadline'],deadline=deadline,
        added_seconds=args.hours*3600,resume_step=saved['step'],donor_sha256=donor_sha,
        binary_sha256=digest(binary),extended_at=time.time(),preserved_optimizer=True,preserved_ema=True)
    new_state=copy.deepcopy(state);new_state.update(status='prepared',command=command,
        donor_sha256=donor_sha,checkpoint_sha256=donor_sha,checkpoint_step=saved['step'],
        inputs={p.name:digest(p) for p in inputs.iterdir()},extension=extension)
    # A replacement binary has its own patch/hash provenance; old commit tags
    # describe the donor, not this newly frozen executable.
    new_state['donor_training_code_commit']=state.get('training_code_commit')
    new_state['training_code_commit']=args.training_code_commit
    atomic_json(run/'state.json',new_state);verify_inputs(run,new_state)
    (root/'hours').mkdir();(root/'evaluation-cache').symlink_to(old/'evaluation-cache',target_is_directory=True)
    for key in progress['reports']:
        (root/'hours'/f'{int(key):02d}').symlink_to(old/'hours'/f'{int(key):02d}',target_is_directory=True)
    for name in ('hourly.csv','hourly-status.txt','best-development.ckpt'):
        if (old/name).exists(): shutil.copyfile(old/name,root/name)
    for key in ('error','worker_pid','completed_at','last_evaluation_error'): progress.pop(key,None)
    seconds=deadline-progress['started_at']
    progress.update(status='prepared',phase='resume',root=str(root),run=str(run),unit=unit,
        train_log=str(run/'train.log'),checkpoint=str(model/'last.ckpt'),checkpoint_step=saved['step'],
        budget_seconds=seconds,deadline=deadline,finish_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        first_interval=max(map(int,progress['reports']))+1,cover_passes=passes,
        completion_status=f'{seconds/3600:g}-hour run complete',remaining_seconds=deadline-time.time(),
        lr_schedule='wall',learning_rate='WSD over the extended wall-time budget; no new warmup; cooldown in the final 20%',
        extensions=[*progress.get('extensions',[]),extension],
        tool_sha256={path:digest(path) for path in progress['tool_sha256']})
    atomic_json(root/'testing.json',progress);atomic_json(root/'extension.json',extension)
    (REPO/'runs'/root.name).symlink_to(root,target_is_directory=True)
    progress_old=json.loads((old/'testing.json').read_text())
    progress_old.update(status='superseded by extended surface continuation',phase='stopped',
        checkpoint_step=saved['step'],replacement_root=str(root),replacement_unit=unit,replaced_at=time.time())
    atomic_json(old/'testing.json',progress_old)
    for name in ('ufsm-surface-rollback-root','ufsm-surface-erosion1-root'):
        Path('/tmp',name).write_text(str(root)+'\n')
    subprocess.run(['systemd-run','--user',f'--unit={unit}',f'--property=WorkingDirectory={REPO}',
        '--property=KillMode=mixed','--property=TimeoutStopSec=180',sys.executable,
        str(REPO/'tools/sheet_watch.py'),'--root',str(root)],check=True)
    print(json.dumps(dict(root=str(root),unit=unit,finish_utc=progress['finish_utc'],resume_step=saved['step'],
        added_hours=args.hours,queued_passes=len(passes)),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',help='defaults to the active production root')
    p.add_argument('--hours',type=float,default=24,help='hours added to the existing deadline')
    p.add_argument('--binary',required=True,help='trainer with next-pass and wall-time schedule support')
    p.add_argument('--binary-source-patch',help='complete source diff for the frozen trainer')
    p.add_argument('--training-code-commit',help='source commit that built the frozen trainer')
    extend(p.parse_args())
