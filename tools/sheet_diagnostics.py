#!/usr/bin/env python3
"""Development-only surface diagnostics for a frozen Paris4 winding run."""
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.ndimage import map_coordinates

from sheet_geometry import atomic_json, digest


def binary_scores(pos, neg):
    pos=np.asarray(pos,np.int64); neg=np.asarray(neg,np.int64)
    tp=np.cumsum(pos[::-1])[::-1]; fp=np.cumsum(neg[::-1])[::-1]; fn=pos.sum()-tp
    f1=2*tp/np.maximum(1,2*tp+fp+fn); best=int(np.argmax(f1))
    below=np.cumsum(neg)-neg
    auc=float(np.sum(pos.astype(float)*(below+.5*neg))/(float(pos.sum())*float(neg.sum()))) if pos.sum() and neg.sum() else None
    return dict(best_binary_f1=float(f1[best]),best_cutoff=best/255,
        precision_at_best=float(tp[best]/max(1,tp[best]+fp[best])),recall_at_best=float(tp[best]/max(1,pos.sum())),
        fixed_cutoffs={str(i/255):float(f1[i]) for i in (60,115)},roc_auc=auc,
        mean_probability_surface=float(np.dot(pos,np.arange(256))/pos.sum()/255) if pos.sum() else None,
        mean_probability_background=float(np.dot(neg,np.arange(256))/neg.sum()/255) if neg.sum() else None,
        constant_foreground_binary_f1=float(2*pos.sum()/max(1,2*pos.sum()+neg.sum())))


def diagnose(run, prediction, checkpoint, output, box, baseline_raw=None, baseline_step=None):
    run=Path(run); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    inputs=run/'inputs'; binary=inputs/'ufsm'
    eval_sources=inputs/'evaluation-sources.json'
    cfg=json.loads((eval_sources if eval_sources.exists() else inputs/'sources.json').read_text())['sources'][0]
    origin=np.array(box[:3]); shape=tuple(box[3:])
    if cfg['um']!=2.4 or cfg['targets']['recto'].get('min_level')!=1:
        raise ValueError('diagnostics require native Paris4 CT and the registered 4.8um binary target')
    common=run.parent/'evaluation-cache'; common.mkdir(exist_ok=True)
    def read(store,key,lo,size,path):
        if not path.exists():
            result=subprocess.run([str(binary),'read',str(store),key,*map(str,lo),*map(str,size),str(path),'--threads','4'],capture_output=True,text=True)
            if result.returncode: raise RuntimeError(result.stderr)
        if path.stat().st_size!=int(np.prod(size)): raise ValueError('unexpected raw volume length: '+str(path))
        return np.memmap(path,np.uint8,mode='r',shape=tuple(size))
    ct=read(cfg['root'],cfg['ct']+'/0',origin,shape,common/'ct.raw')
    coarse=read(cfg['targets']['recto']['root'],'4.8',(origin+1)//2,tuple(n//2+1 for n in shape),common/'labels.raw')
    raw=output/'probability.raw'; volume=read(prediction,'2.4',(0,0,0),shape,raw)
    indices=[(np.arange(n)+1)//2 for n in shape]
    pos=np.zeros(256,np.int64); neg=pos.copy()
    for z in range(0,shape[0],16):
        truth=coarse[np.ix_(indices[0][z:z+16],indices[1],indices[2])]>0
        valid=ct[z:z+16]>0; p=volume[z:z+16]
        pos+=np.bincount(p[truth&valid],minlength=256); neg+=np.bincount(p[~truth&valid],minlength=256)
    result=binary_scores(pos,neg)
    if not eval_sources.exists():
        truth=dict(np.load(inputs/'geometry/evaluation-mesh.npz')) if (inputs/'geometry/evaluation-mesh.npz').exists() else dict(np.load('/vesuvius/ufsm/gt/paris4-sheet-20261003/geometry/evaluation-mesh.npz'))
        points=truth['xyz'][truth['region']==0]-origin
        surface=map_coordinates(volume,points.T,order=1,mode='constant',cval=0,output=np.float32)/255
        result.update(surface_mean_probability=float(surface.mean()),fraction_gt_points_above_cutoff={str(t):float((surface>=t).mean()) for t in (.2,.25,.3,.4,.5)})
    result.update(checkpoint_sha256=digest(checkpoint),
        original_42408_best_binary_f1=.1825675723,
        metric_note='Development hard thin-label F1 with CT air excluded; best cutoff fitted on this same development box. Not final-test validation or topology certification.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plane=ct[shape[0]//2]; lo,hi=np.percentile(plane[plane>0],[1,99])
    fig,axes=plt.subplots(1,4,figsize=(16,4.8),layout='constrained')
    axes[0].imshow(plane,cmap='gray',vmin=lo,vmax=hi); axes[0].set_title('CT scan')
    mask=coarse[np.ix_([indices[0][shape[0]//2]],indices[1],indices[2])][0]>0
    axes[1].imshow(plane,cmap='gray',vmin=lo,vmax=hi)
    overlay=np.zeros((*plane.shape,4)); overlay[mask]=[0,.9,1,.8]; axes[1].imshow(overlay); axes[1].set_title('Thin labels (cyan)')
    before=np.memmap(baseline_raw,np.uint8,mode='r',shape=shape) if baseline_raw else volume
    axes[2].imshow(np.asarray(before[shape[0]//2],np.float32)/255,cmap='magma',vmin=0,vmax=.6)
    axes[2].set_title(f'Starting checkpoint\nStep {baseline_step:,}' if baseline_step is not None else 'Starting checkpoint')
    im=axes[3].imshow(np.asarray(volume[shape[0]//2],np.float32)/255,cmap='magma',vmin=0,vmax=.6)
    axes[3].set_title(f'Current prediction\nBest development F1: {result["best_binary_f1"]:.3f}')
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im,ax=list(axes[2:]),label='Surface probability',shrink=.8)
    fig.suptitle('Paris 4 development slice — EMA, identical 528³ windows and probability scale')
    image=output/'preview.png'; fig.savefig(image,dpi=150,facecolor='white'); plt.close(fig)
    result.update(image=str(image),probability_raw=str(raw)); atomic_json(output/'diagnostics.json',result)
    return result
