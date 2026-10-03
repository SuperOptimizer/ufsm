#!/usr/bin/env python3
"""Experimental surface_winding runner: freeze/train, predict/extract, score/export.

GPU work uses the existing production leases. No full training run starts unless
the train subcommand is explicitly invoked. Existing production is untouched.
"""
import argparse
import ctypes
import ctypes.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from production import flags, gpu_lock, header
from sheet_geometry import atomic_json, digest, reference_value

ROOT=Path(__file__).resolve().parents[1]
BINARY=str(Path(__file__).resolve().parent/"ufsm") if (Path(__file__).resolve().parent/"ufsm").exists() else str(ROOT/"build/ufsm")
AFFINITY_DISTANCE=32  # ~22-voxel GT edges plus two 4-voxel endpoint tolerances


def decompress_zstd(path, expected):
    lib=ctypes.CDLL(ctypes.util.find_library("zstd"))
    lib.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
    lib.ZSTD_decompress.restype=ctypes.c_size_t
    encoded=Path(path).read_bytes(); source=ctypes.create_string_buffer(encoded)
    output=ctypes.create_string_buffer(expected)
    size=lib.ZSTD_decompress(output,expected,source,len(encoded))
    if size!=expected: raise ValueError(f"invalid winding shard: {path}")
    return output.raw


def run(command,log):
    with Path(log).open("w") as f:
        env={k:v for k,v in os.environ.items() if not k.startswith("UFSM_")}
        subprocess.run(list(map(str,command)),check=True,stdout=f,stderr=subprocess.STDOUT,env=env)


def freeze_geometry(source,destination):
    source=Path(source).resolve(); destination.mkdir()
    manifest=json.loads(source.read_text())
    for name,sha in manifest["files"].items():
        if Path(name).name!=name or digest(source.parent/name)!=sha:
            raise ValueError("geometry data changed or invalid filename")
        shutil.copyfile(source.parent/name,destination/name)
    shutil.copyfile(source,destination/"geometry.json")
    return destination/"geometry.json"


def audit(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(a.geometry).resolve().parent
    data=dict(np.load(root/"training-mesh.npz")); reference=json.loads((root/"reference.json").read_text())
    xyz,q=data["xyz"],data["q"]; z=a.z if a.z is not None else float(np.median(xyz[:,0]))
    ids=np.where(abs(xyz[:,0]-z)<=512)[0]
    if not len(ids): raise ValueError("no geometry near the selected axial slice")
    ids=ids[::max(1,len(ids)//20000)]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    plot=axes[0].scatter(xyz[ids,2],xyz[ids,1],c=q[ids],s=3,cmap="turbo")
    axes[0].set(xlabel="native X voxel",ylabel="native Y voxel",title=f"Mesh winding near Z={z:.0f}")
    axes[0].set_aspect("equal"); fig.colorbar(plot,ax=axes[0],label="continuous turns (arbitrary gauge)")
    prior=reference_value(reference,xyz[ids])
    axes[1].scatter(prior,q[ids],s=3,c=q[ids],cmap="turbo")
    lo,hi=float(min(prior.min(),q[ids].min())),float(max(prior.max(),q[ids].max()))
    axes[1].plot([lo,hi],[lo,hi],color="black",lw=1)
    axes[1].set(xlabel="coarse radial reference (turns)",ylabel="mesh winding (turns)",title="Reference versus audited geometry")
    fig.savefig(a.out,dpi=160); plt.close(fig); print(a.out)


def train(a):
    out=Path(a.out).resolve()
    if out.exists(): raise ValueError("choose a new experimental run directory")
    recipe=json.loads(Path(a.recipe).read_text()); out.mkdir(parents=True)
    inputs=out/"inputs"; inputs.mkdir()
    geometry=freeze_geometry(a.geometry,inputs/"geometry")
    shutil.copy2(a.binary,inputs/"ufsm"); shutil.copyfile(a.resume,inputs/"resume.ckpt")
    cfg=json.loads(Path(a.sources).read_text())
    if len(cfg["sources"])!=1: raise ValueError("Paris4 experiment requires one source")
    source=cfg["sources"][0]
    audit=json.loads((geometry.parent/"audit.json").read_text())
    scan=audit.get("scan")
    if scan and (scan["volume"] not in source["ct"] or float(source["um"])!=scan["native_um"]):
        raise ValueError("geometry and CT refer to different registered scans")
    if audit.get("axis_sha256") and (not source.get("axis") or digest(source["axis"])!=audit["axis_sha256"]):
        raise ValueError("geometry and CT use different axes")
    # Frozen recipes bind the immutable external stores through their metadata
    # and provenance, without copying CT/labels or publishing them to Git.
    bindings={}
    for target in source["targets"].values():
        if isinstance(target,dict) and "root" in target:
            root=Path(target["root"]).resolve()
            if not a.existing_targets:
                provenance=root/"provenance.json"
                if not provenance.exists() or json.loads(provenance.read_text()).get("surface_band_chamfer")!=0:
                    raise ValueError("thin/winding candidates require the unexpanded --band-chamfer 0 label store")
            for name in ("zarr.json","provenance.json"):
                if (root/name).exists(): bindings[str(root/name)]=digest(root/name)
    for target in source["targets"].values():
        if isinstance(target,dict) and "root" in target: target["root"]=str(Path(target["root"]).resolve())
    source["root"]=str(Path(source["root"]).resolve()) if "://" not in source["root"] else source["root"]
    if source.get("axis"):
        shutil.copyfile(source["axis"],inputs/"axis.json"); source["axis"]=str(inputs/"axis.json")
    atomic_json(inputs/"sources.json",cfg)
    plan=json.loads(Path(a.cover).read_text())
    excluded=audit["splits"]["development"]+audit["splits"]["test"]
    guard=int(audit["guard"])
    P=int(plan["P"])
    def safe(tile):
        origin=np.array(tile[1:])
        return not any(((origin<np.array(b[:3])+np.array(b[3:])+guard)&(origin+P>np.array(b[:3])-guard)).all() for b in excluded)
    safe_tiles=[t for t in plan["tiles"] if safe(t)]
    if len(safe_tiles)<a.updates: raise ValueError("not enough held-out-safe coverage tiles")
    plan["tiles"]=safe_tiles[:a.updates]; plan["count"]=a.updates
    # Finite native loader validates tile coordinates/source identity. The full
    # source plan remains provenance; this is intentionally a bounded ablation.
    atomic_json(inputs/"cover.json",plan)
    opts=dict(recipe["train"],P=P,B=1,steps=a.updates,cover=str(inputs/"cover.json"),
              task="surface_winding",geometry=str(geometry),**{"sheet-init":1,"sheet-variant":a.variant})
    opts.pop("input-prec",None); opts["mem"]="auto16"
    if a.baseline:
        for key in ("task","geometry","sheet-init","sheet-variant"): opts.pop(key,None)
        opts["warm-start"]=1; opts["input-prec"]=0
        if a.existing_targets: opts["soft"]=3
    atomic_json(inputs/"recipe.json",recipe)
    command=[inputs/"ufsm","train",inputs/"sources.json","--out",out/"model","--gpus",a.gpus,
             "--resume",inputs/"resume.ckpt",*flags(opts)]
    if "," in a.gpus: command += ["--split","z"]
    state=dict(version=1,task="surface_winding",status="prepared",command=list(map(str,command)),
               variant=a.variant,updates=a.updates,geometry_sha256=digest(geometry),
               external_metadata=bindings,donor_sha256=digest(inputs/"resume.ckpt"),
               inputs={str(p.relative_to(inputs)):digest(p) for p in inputs.rglob("*") if p.is_file()})
    if a.baseline: state["task"]="surface"
    atomic_json(out/"state.json",state)
    if a.prepare_only:
        print(f"prepared {out}; no GPU work started")
        return
    execute_candidate(out)


def verify_inputs(out,state):
    for name,sha in state["inputs"].items():
        if digest(out/"inputs"/name)!=sha: raise ValueError(f"changed run input: {name}")
    for name,sha in state.get("external_metadata",{}).items():
        if digest(name)!=sha: raise ValueError(f"external store metadata changed: {name}")


def execute_candidate(out):
    out=Path(out).resolve(); state=json.loads((out/"state.json").read_text())
    verify_inputs(out,state)
    if state["status"] not in ("prepared","interrupted","training"):
        raise ValueError("candidate is already complete")
    command=list(state["command"])
    checkpoint=out/"model/last.ckpt"
    if checkpoint.exists():
        saved=header(checkpoint)
        if saved["extra"]["cover"]["sha256"]!=digest(out/"inputs/cover.json"):
            raise ValueError("resume checkpoint belongs to another coverage plan")
        if state["task"]=="surface_winding" and saved["extra"]["sheet"]["geometry_sha256"]!=state["geometry_sha256"]:
            raise ValueError("resume checkpoint belongs to another geometry dataset")
        command[command.index("--resume")+1]=str(checkpoint)
        for flag in ("--sheet-init","--warm-start"):
            if flag in command:
                pos=command.index(flag); del command[pos:pos+2]
        if saved["extra"]["cover"]["cursor"]==state["updates"]:
            state.update(status="trained",checkpoint_sha256=digest(checkpoint))
            atomic_json(out/"state.json",state); return
    gpus=command[command.index("--gpus")+1]
    with gpu_lock(gpus):
        state["status"]="training"; atomic_json(out/"state.json",state)
        start=time.monotonic()
        try: run(command,out/"train.log")
        except BaseException:
            state["status"]="interrupted"; atomic_json(out/"state.json",state); raise
        saved=header(out/"model/last.ckpt")
        if saved["extra"]["cover"]["cursor"]!=state["updates"]: raise RuntimeError("bounded cover did not finish")
        state.update(status="trained",wall_seconds=state.get("wall_seconds",0)+time.monotonic()-start,
                     checkpoint_sha256=digest(out/"model/last.ckpt"))
        atomic_json(out/"state.json",state)
    print(f"trained experimental candidate: {out}")


def resume(a): execute_candidate(a.run)


def reconfigure(a):
    """Explicitly change augmentation in an interrupted finite-cover run.

    A new immutable run records the boundary; the checkpoint preserves the
    optimizer, cover cursor and original winding ramp/schedule. This is not a
    fresh matched ablation and must not be compared to the original cohort.
    """
    previous=Path(a.run).resolve(); old=json.loads((previous/"state.json").read_text())
    verify_inputs(previous,old)
    out=Path(a.out).resolve()
    if out.exists(): raise ValueError("choose a new reconfigured run directory")
    recipe=json.loads(Path(a.recipe).read_text())
    allowed={"noaug","rotonly","zfix","intonly","ct-aug","symmetry-p","axis-jitter",
             "geometry-aug","rotate-deg","rotate-p","elastic","elastic-p","label-morph","label-morph-p","soft"}
    command=list(old["command"])
    for key,value in recipe["train"].items():
        flag="--"+key
        if key not in allowed and (flag not in command or str(value)!=command[command.index(flag)+1]):
            raise ValueError("reconfigure changes augmentation only: "+key)
    # The trainer atomically publishes checkpoints. One copied snapshot binds
    # both the header and payload even if this is prepared before shutdown.
    out.mkdir(parents=True); inputs=out/"inputs"
    shutil.copytree(previous/"inputs",inputs)
    shutil.copyfile(previous/"model/last.ckpt",inputs/"resume.ckpt")
    saved=header(inputs/"resume.ckpt"); cover=saved.get("extra",{}).get("cover",{})
    if cover.get("sha256")!=digest(inputs/"cover.json") or cover.get("count")!=old["updates"]:
        raise ValueError("checkpoint/coverage contract mismatch")
    if cover["cursor"]>=cover["count"]: raise ValueError("finite cover is complete")
    if old["task"]=="surface_winding" and saved.get("extra",{}).get("sheet",{}).get("geometry_sha256")!=old["geometry_sha256"]:
        raise ValueError("checkpoint/geometry contract mismatch")
    shutil.copy2(a.binary,inputs/"ufsm")
    for i,value in enumerate(command):
        if value.startswith(str(previous/"inputs")+"/"):
            command[i]=str(inputs/Path(value).relative_to(previous/"inputs"))
    for flag,value in (("--out",str(out/"model")),("--resume",str(inputs/"resume.ckpt"))):
        command[command.index(flag)+1]=value
    for flag in ("--sheet-init","--warm-start"):
        if flag in command:
            pos=command.index(flag); del command[pos:pos+2]
    for key,value in recipe["train"].items():
        if key not in allowed: continue
        flag="--"+key
        if flag in command: command[command.index(flag)+1]=str(value)
        else: command.extend([flag,str(value)])
    cfg=json.loads((inputs/"sources.json").read_text())
    for source in cfg["sources"]:
        if source.get("axis","").startswith(str(previous/"inputs")+"/"):
            source["axis"]=str(inputs/Path(source["axis"]).relative_to(previous/"inputs"))
    atomic_json(inputs/"sources.json",cfg); atomic_json(inputs/"recipe.json",recipe)
    state=dict(old,status="prepared",command=command,donor_sha256=digest(inputs/"resume.ckpt"),
        inputs={str(p.relative_to(inputs)):digest(p) for p in inputs.rglob("*") if p.is_file()},
        augmentation_change=dict(previous_run=str(previous),step=saved["step"],cursor=cover["cursor"],
            previous_recipe_sha256=old["inputs"]["recipe.json"],previous_checkpoint_augmentation=saved.get("extra",{}).get("augmentation"),
            comparison="mixed augmentation history; a new matched cohort is required for promotion"))
    for key in ("checkpoint_sha256","wall_seconds"): state.pop(key,None)
    atomic_json(out/"state.json",state)
    if a.prepare_only: print(f"prepared augmentation continuation {out} at step {saved['step']}")
    else: execute_candidate(out)


def sweep(a):
    """Prepare equal donor/order/budget ablations. GPU jobs stay explicit."""
    out=Path(a.out)
    if out.exists(): raise ValueError("choose a new sweep directory")
    out.mkdir(parents=True)
    # The production checkpoint may be atomically replaced while candidates
    # are prepared. Take one donor snapshot for the entire comparison.
    donor=out/"donor.ckpt"
    shutil.copyfile(a.resume,donor)
    donor_sha=digest(donor)
    for name,baseline,variant,sources,existing in (("expanded",True,0,a.original_sources,True),
            ("thin",True,0,a.sources,False),("winding",False,1,a.sources,False),("full",False,2,a.sources,False)):
        args=argparse.Namespace(**vars(a)); args.out=str(out/name)
        args.resume=str(donor)
        args.baseline=baseline; args.variant=variant; args.sources=sources
        args.existing_targets=existing; args.prepare_only=True
        train(args)
    atomic_json(out/"sweep.json",dict(version=1,updates=a.updates,donor_sha256=donor_sha,order="same filtered cover order",
        candidates=[str((out/name).resolve()) for name in ("expanded","thin","winding","full")]))


def checked_run(path):
    out=Path(path).resolve(); state=json.loads((out/"state.json").read_text())
    verify_inputs(out,state)
    if state["status"]!="trained" or digest(out/"model/last.ckpt")!=state["checkpoint_sha256"]:
        raise ValueError("candidate is incomplete or checkpoint changed")
    return out,state


def predict(a):
    out,state=checked_run(a.run); cfg=json.loads((out/"inputs/sources.json").read_text())["sources"][0]
    target=Path(a.out).resolve()
    if target.exists(): raise ValueError("choose a new prediction directory")
    staging=target.with_name(target.name+".building")
    if staging.exists(): raise ValueError("incomplete prediction staging exists")
    recipe=json.loads((out/"inputs/recipe.json").read_text())
    opts=dict(recipe["predict"]); opts.update(box=a.box)
    if state["task"]=="surface_winding": opts["reference"]=str(out/"inputs/geometry/reference.json")
    command=[out/"inputs/ufsm","predict",out/"model/last.ckpt",cfg["root"],cfg["ct"],staging,
             "--um",cfg["um"],"--gpu",a.gpu,*flags(opts)]
    if cfg.get("axis"): command += ["--axis",cfg["axis"]]
    if a.window:
        opts["window"]=a.window
        command=[out/"inputs/ufsm","predict",out/"model/last.ckpt",cfg["root"],cfg["ct"],staging,
                 "--um",cfg["um"],"--gpu",a.gpu,*flags(opts)]
        if cfg.get("axis"): command += ["--axis",cfg["axis"]]
    with gpu_lock(a.gpu):
        run(command,target.with_name(target.name+".log"))
    if state["task"]=="surface_winding": shutil.copyfile(out/"inputs/geometry/reference.json",staging/"winding/reference.json")
    else: shutil.copyfile(out/"inputs/geometry/reference.json",staging/"reference.json")
    atomic_json(staging/"candidate.json",dict(checkpoint_sha256=state["checkpoint_sha256"],
                                               comparison=dict(updates=state["updates"],donor_sha256=state["donor_sha256"],
                                                   cover_sha256=digest(out/"inputs/cover.json"),geometry_sha256=state["geometry_sha256"]),
                                               run=str(out),command=list(map(str,command))))
    staging.rename(target)
    print(f"prediction: {target}")


def extract(a):
    pred=Path(a.prediction)
    meta=json.loads((pred/"zarr.json").read_text())
    attrs=meta["attributes"]; scales=attrs.get("ome",attrs)["multiscales"]
    key=scales[0]["datasets"][0]["path"]
    baseline=not (pred/"winding/manifest.json").exists()
    if baseline:
        array=json.loads((pred/key/"zarr.json").read_text())
        details=array["attributes"]["ufsm"]
        reference=json.loads((pred/"reference.json").read_text())
        manifest=dict(shard=array["chunk_grid"]["configuration"]["chunk_shape"][0],
                      shape=array["shape"],origin_zyx=details["origin_zyx"])
    else: manifest=json.loads((pred/"winding/manifest.json").read_text())
    shard=int(manifest["shard"]); origin=np.array(manifest["origin_zyx"]); shape=np.array(manifest["shape"])
    point_parts,q_parts,p_parts=[],[],[]
    cells=[(np.array([int(v) for v in path.name.split(".")[:3]]),path) for path in sorted((pred/"winding").glob("*.q.zst"))]
    if baseline:
        import itertools
        cells=[(np.array(cell),None) for cell in itertools.product(*(range(math.ceil(n/shard)) for n in shape))]
    for cell,path in cells:
        offset=cell*shard; size=np.minimum(shard,shape-offset)
        q=None
        if not baseline:
            q=np.frombuffer(decompress_zstd(path,shard**3*4),"<f4").reshape((shard,)*3)
            q=q[:size[0],:size[1],:size[2]]
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/"probability.raw"
            subprocess.run([a.binary,"read",str(pred),key,*map(str,offset),*map(str,size),str(raw),"--threads","2"],check=True,stdout=subprocess.DEVNULL)
            probability=np.fromfile(raw,np.uint8).reshape(tuple(size))/255
        # Sample within every spatial cell AND winding-coordinate bin; a close
        # second sheet is not discarded merely for sharing an XYZ cell.
        pv=probability[::a.stride,::a.stride,::a.stride]
        if baseline:
            positions=np.moveaxis(np.indices(pv.shape),0,-1)*a.stride+origin+offset
            view=reference_value(reference,positions)
        else: view=q[::a.stride,::a.stride,::a.stride]
        ij=np.array(np.where(np.isfinite(view)&(pv>=a.threshold))).T
        if not len(ij): continue
        coords=ij*a.stride; qq=view[tuple(ij.T)]; pp=pv[tuple(ij.T)]
        groups=np.column_stack([coords//a.cell,np.floor(qq).astype(np.int64)])
        order=np.lexsort((-pp,groups[:,3],groups[:,2],groups[:,1],groups[:,0]))
        gs=groups[order]; first=np.r_[True,(gs[1:]!=gs[:-1]).any(axis=1)]
        keep=order[first]
        point_parts.append(coords[keep]+origin+offset); q_parts.append(qq[keep]); p_parts.append(pp[keep])
    if not point_parts: raise ValueError("prediction has no supported surface evidence")
    xyz,q,probability=[np.concatenate(v) for v in (point_parts,q_parts,p_parts)]
    # Local planes provide unoriented normals only when the neighbourhood is
    # genuinely sheet-like. Unsupported normals stay zero and carry no claim.
    tree=cKDTree(xyz); distances,neighbours=tree.query(xyz,k=min(12,len(xyz)),workers=1)
    normal=np.zeros_like(xyz,dtype=float)
    if distances.ndim==2:
        for i in range(len(xyz)):
            ids=neighbours[i][(distances[i]<=32)&(abs(q[neighbours[i]]-q[i])<.2)]
            if len(ids)<4: continue
            local=xyz[ids]-xyz[ids].mean(axis=0)
            eigen,vectors=np.linalg.eigh(local.T@local)
            if eigen[0]<.1*max(eigen[1],1e-8): normal[i]=vectors[:,0]
    # A surface-only baseline has no winding head. Its bridges must be measured
    # from actual predicted probability across the gap, not from the radial
    # reference (which would hide its connectivity errors).
    pairs=tree.query_pairs(AFFINITY_DISTANCE,output_type="ndarray")
    affinity=np.zeros(len(pairs),np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        raw=Path(tmp)/"probability.raw"
        subprocess.run([a.binary,"read",str(pred),key,"0","0","0",*map(str,shape),str(raw),"--threads","4"],check=True,stdout=subprocess.DEVNULL)
        volume=np.memmap(raw,np.uint8,mode="r",shape=tuple(shape))
        t=np.linspace(0,1,13)
        for start in range(0,len(pairs),8192):
            ids=pairs[start:start+8192]
            lines=(1-t[None,:,None])*xyz[ids[:,0],None,:]+t[None,:,None]*xyz[ids[:,1],None,:]-origin
            sampled=map_coordinates(volume,lines.reshape(-1,3).T,order=1,mode="constant",cval=0,output=np.float32)
            affinity[start:start+len(ids)]=sampled.reshape(-1,13).min(axis=1)/255
        del volume
    candidate=json.loads((pred/"candidate.json").read_text())
    np.savez_compressed(a.out,xyz=xyz.astype(np.float32),q=q.astype(np.float32),
                        probability=probability.astype(np.float32),normal=normal.astype(np.float32),
                        checkpoint_sha256=candidate["checkpoint_sha256"],threshold=a.threshold,
                        comparison=json.dumps(candidate["comparison"],sort_keys=True),
                        uses_winding=not baseline,affinity_edges=pairs,affinity_probability=affinity,
                        reference_sha256=digest(pred/("reference.json" if baseline else "winding/reference.json")))
    print(f"extracted {len(q)} surface evidence points: {a.out}")


def score_geometry(truth,evidence,region):
    gt={k:v for k,v in truth.items()}; use=gt["region"]==region
    xyz=gt["xyz"][use]; q=gt["q"][use]
    if len(xyz)<16 or len(evidence["xyz"])<16: raise ValueError("too few evaluation points")
    tree=cKDTree(evidence["xyz"]); distance,nearest=tree.query(xyz,workers=1)
    observed=evidence["q"][nearest]; error=observed-q
    valid=distance<=4
    mapping=np.full(len(use),-1,np.int64); mapping[np.where(use)[0]]=np.arange(use.sum())
    edges=np.asarray(gt["edges"]); edges=edges[(mapping[edges]>=0).all(axis=1)]
    edges=mapping[edges]
    # Track along GT adjacency and stop at unsupported or half-turn jumps.
    uses_winding=bool(evidence.get("uses_winding",True))
    jumps=(abs(np.diff(error[edges],axis=1).ravel())>.5) if len(edges) and uses_winding else np.zeros(len(edges),bool)
    lengths=np.linalg.norm(xyz[edges[:,1]]-xyz[edges[:,0]],axis=1) if len(edges) else np.array([])
    good_edges=valid[edges].all(axis=1)&~jumps
    pred=np.asarray(evidence["xyz"]); pq=np.asarray(evidence["q"])
    pairs=np.asarray(evidence["affinity_edges"] if "affinity_edges" in evidence else tree.query_pairs(AFFINITY_DISTANCE,output_type="ndarray"),np.int64)
    affinity=np.asarray(evidence.get("affinity_probability",np.ones(len(pairs))))
    accepted=affinity>=float(evidence.get("threshold",.3))
    if len(pairs) and uses_winding: accepted &= abs(pq[pairs[:,1]]-pq[pairs[:,0]])<.25
    pairs=pairs[accepted]
    if "affinity_edges" in evidence and len(edges):
        # Supported endpoints alone cannot certify a continuous track: isolated
        # predicted dots on the GT grid would otherwise score as a long sheet.
        witness=np.sort(pairs,axis=1)
        keys=np.unique(witness[:,0]*len(pred)+witness[:,1])
        endpoints=np.sort(nearest[edges],axis=1)
        connected=(endpoints[:,0]==endpoints[:,1]) | np.isin(endpoints[:,0]*len(pred)+endpoints[:,1],keys)
        good_edges &= connected
    gt_distance,gt_nearest=cKDTree(xyz).query(pred,workers=1)
    bridges=np.zeros(len(pairs),bool); affinity_switches=np.zeros(len(pairs),bool)
    if len(pairs):
        a,b=pairs.T
        near=(gt_distance[a]<=8)&(gt_distance[b]<=8)
        true_delta=abs(q[gt_nearest[a]]-q[gt_nearest[b]])
        gap=np.linalg.norm(xyz[gt_nearest[a]]-xyz[gt_nearest[b]],axis=1)>=8
        bridges=near&(true_delta>.75)&gap
        affinity_switches=near&(true_delta>.5)
        blocked=np.zeros(len(pred),bool); blocked[pairs[bridges].ravel()]=True
        good_edges &= ~blocked[nearest[edges]].any(axis=1)
    # A sum of every edge in a 2D patch exaggerates track length. Measure a
    # two-sweep geodesic diameter instead (a lower bound on cyclic grid graphs).
    tracks=[]
    if good_edges.any():
        a,b=edges[good_edges].T; length=lengths[good_edges]
        graph=coo_matrix((np.r_[length,length],(np.r_[a,b],np.r_[b,a])),shape=(len(xyz),len(xyz))).tocsr()
        _,labels=connected_components(graph,directed=False)
        for component in np.unique(labels[np.r_[a,b]]):
            ids=np.where(labels==component)[0]
            local=graph[ids][:,ids]
            first=dijkstra(local,directed=False,indices=0)
            last=dijkstra(local,directed=False,indices=int(np.argmax(first)))
            tracks.append(float(last.max()))
    edge_switch=float(jumps[valid[edges].all(axis=1)].mean()) if len(edges) and valid[edges].all(axis=1).any() else 0
    return dict(points=len(xyz),supported_coverage=float(valid.mean()),
                surface_distance=float(np.median(distance)),coordinate_mae=float(np.mean(abs(error[valid]))) if valid.any() else None,
                winding_switch_frequency=max(edge_switch,float(affinity_switches.mean()) if len(pairs) else 0),
                false_bridge_frequency=float(bridges.mean()) if len(bridges) else 0,
                winding_switch_count=int(jumps.sum()+affinity_switches.sum()),false_bridge_count=int(bridges.sum()),
                median_correct_track_length=float(np.median(tracks)) if tracks else 0,
                metric_note="straight-line probability-supported affinities, winding-constrained when available; track = two-sweep geodesic diameter lower bound; evaluate reconstructed winding correspondence too")


def mesh_evidence(points,mesh):
    """Certified nearest triangle search, with barycentric winding interpolation."""
    vertices=np.asarray(mesh["xyz"],float); faces=np.asarray(mesh["faces"],int)
    triangles=vertices[faces]; centres=triangles.mean(axis=1)
    radius=float(np.linalg.norm(triangles-centres[:,None],axis=2).max())
    centre_tree=cKDTree(centres); vertex_tree=cKDTree(vertices)
    xyz=[]; q=[]; probability=[]
    for point in points:
        upper=vertex_tree.query(point)[0]
        candidates=centre_tree.query_ball_point(point,upper+radius+1e-6)
        tri=triangles[candidates]; a,b,c=np.moveaxis(tri,1,0)
        ab,ac,ap=b-a,c-a,point-a
        aa=np.einsum("ij,ij->i",ab,ab); cc=np.einsum("ij,ij->i",ac,ac)
        bc=np.einsum("ij,ij->i",ab,ac); pa=np.einsum("ij,ij->i",ap,ab); pc=np.einsum("ij,ij->i",ap,ac)
        determinant=aa*cc-bc*bc
        u=(cc*pa-bc*pc)/determinant; v=(aa*pc-bc*pa)/determinant
        bary=np.column_stack([1-u-v,u,v])
        positions=a+u[:,None]*ab+v[:,None]*ac
        distance=np.linalg.norm(positions-point,axis=1); distance[(bary<0).any(axis=1)]=np.inf
        options=[(positions,bary,distance)]
        for i,j in ((0,1),(1,2),(2,0)):
            edge=tri[:,j]-tri[:,i]; t=np.clip(np.einsum("ij,ij->i",point-tri[:,i],edge)/np.einsum("ij,ij->i",edge,edge),0,1)
            p=tri[:,i]+t[:,None]*edge; w=np.zeros((len(tri),3)); w[:,i]=1-t; w[:,j]=t
            options.append((p,w,np.linalg.norm(p-point,axis=1)))
        best=min(((float(d.min()),p,w,int(d.argmin())) for p,w,d in options),key=lambda item:item[0])
        _,p,w,idx=best; face=faces[candidates[idx]]; weights=w[idx]
        xyz.append(p[idx]); q.append(float(np.dot(mesh["q"][face],weights)))
        probability.append(float(np.dot(mesh["evidence_probability"][face],weights)))
    return dict(xyz=np.array(xyz),q=np.array(q),probability=np.array(probability))


def evaluate(a):
    truth=dict(np.load(a.truth)); evidence=dict(np.load(a.evidence))
    result=score_geometry(truth,evidence,0 if a.split=="development" else 1)
    result.update(split=a.split,truth_sha256=digest(a.truth),evidence_sha256=digest(a.evidence),
                  checkpoint_sha256=str(evidence.get("checkpoint_sha256","")),
                  reference_sha256=str(evidence.get("reference_sha256","")),
                  comparison=json.loads(str(evidence.get("comparison","{}"))),
                  threshold=float(evidence.get("threshold",.3)))
    if a.reconstruction:
        root=Path(a.reconstruction)
        reconstruction=json.loads((root/"report.json").read_text())
        if reconstruction["evidence_sha256"]!=result["evidence_sha256"] or reconstruction["checkpoint_sha256"]!=result["checkpoint_sha256"]:
            raise ValueError("reconstruction/evaluation provenance mismatch")
        predicted=mesh_evidence(truth["xyz"],dict(np.load(root/"surface.npz")))
        # Nearest-evidence probabilities at mesh vertices must not turn an
        # inferred completion into an observed surface between those vertices.
        distance,nearest=cKDTree(evidence["xyz"]).query(predicted["xyz"],workers=1)
        supported=(predicted["probability"]>=result["threshold"]) & (distance<=4) & (evidence["probability"][nearest]>=result["threshold"])
        predicted={k:v[supported] for k,v in predicted.items()}
        result.update(reconstruction_sha256=digest(root/"report.json"),
                      reconstruction_metrics=score_geometry(truth,predicted,0 if a.split=="development" else 1))
    atomic_json(a.out,result); print(json.dumps(result,indent=2))


def select(a):
    baseline=json.loads(Path(a.baseline).read_text()); reports=[(path,json.loads(Path(path).read_text())) for path in a.candidates]
    passing=[]
    for path,r in reports:
        if baseline.get("split")!="development" or r.get("split")!="development" or r.get("truth_sha256")!=baseline.get("truth_sha256") or r.get("threshold")!=baseline.get("threshold") or r.get("reference_sha256")!=baseline.get("reference_sha256") or r.get("comparison")!=baseline.get("comparison"):
            raise ValueError("selection requires matched development truth and threshold")
        ok=(r["supported_coverage"]>=.95*baseline["supported_coverage"] and
            r["winding_switch_frequency"]<=.7*baseline["winding_switch_frequency"] and
            r["false_bridge_frequency"]<=.7*baseline["false_bridge_frequency"] and
            r["median_correct_track_length"]>=1.25*baseline["median_correct_track_length"])
        if ok: passing.append((path,r))
    enough_failures=baseline["winding_switch_frequency"]>0 and baseline["false_bridge_frequency"]>0
    if not enough_failures:
        result=dict(status="insufficient baseline failures",action="extend every candidate equally to 5000 updates")
    elif not passing: result=dict(status="no candidate passes",action="retain current production recipe")
    else:
        path,r=min(passing,key=lambda item:(item[1]["winding_switch_frequency"],item[1]["false_bridge_frequency"],-item[1]["median_correct_track_length"]))
        result=dict(status="development candidate selected",report=str(path),report_sha256=digest(path),
                    checkpoint_sha256=r["checkpoint_sha256"],metrics=r,action="evaluate once on final test before export")
    atomic_json(a.out,result); print(json.dumps(result,indent=2))


def export(a):
    out,state=checked_run(a.run); report=json.loads(Path(a.report).read_text()); reconstruction=json.loads(Path(a.reconstruction_report).read_text())
    selection=json.loads(Path(a.selection).read_text())
    cksha=state["checkpoint_sha256"]; refsha=digest(out/"inputs/geometry/reference.json")
    if state["task"]!="surface_winding" or report.get("split")!="test" or reconstruction.get("validation",{}).get("self_intersections")!=0:
        raise ValueError("export needs final-test report and validated reconstruction")
    if selection.get("status")!="development candidate selected" or selection.get("checkpoint_sha256")!=cksha:
        raise ValueError("only the development-selected checkpoint can be exported")
    if report.get("checkpoint_sha256")!=cksha or reconstruction.get("checkpoint_sha256")!=cksha or report.get("reference_sha256")!=refsha or reconstruction.get("reference_sha256")!=refsha:
        raise ValueError("evaluation/reconstruction belongs to another checkpoint or reference")
    if reconstruction.get("evidence_sha256")!=report.get("evidence_sha256"):
        raise ValueError("reconstruction and final-test evaluation must use the same evidence")
    if report.get("reconstruction_sha256")!=digest(a.reconstruction_report) or "reconstruction_metrics" not in report:
        raise ValueError("final-test report must also evaluate reconstructed winding correspondence")
    bundle=Path(a.out)
    if bundle.exists(): raise ValueError("choose a new bundle directory")
    bundle.mkdir(parents=True)
    for source,name in ((out/"inputs/ufsm","ufsm"),(out/"model/last.ckpt","model.ckpt"),
                        (out/"inputs/geometry/reference.json","reference.json"),(Path(a.report),"evaluation.json"),
                        (Path(a.reconstruction_report),"reconstruction.json"),(Path(a.selection),"selection.json")):
        shutil.copy2(source,bundle/name)
    if (out/"inputs/axis.json").exists(): shutil.copy2(out/"inputs/axis.json",bundle/"axis.json")
    shutil.copy2(ROOT/"requirements-sheet.txt",bundle/"requirements.txt")
    for name in ("sheet_pipeline.py","sheet_geometry.py","sheet_reconstruct.py","production.py","eval_holdouts.py","build_training_cover.py"):
        shutil.copy2(ROOT/"tools"/name,bundle/name)
    atomic_json(bundle/"model.json",dict(version=1,task="surface_winding",status="evaluated experimental candidate",
                                         prediction="./ufsm predict model.ckpt ROOT CT OUT --um U --reference reference.json --window 528 --halo 8 --shard 512 --levels 1"+(" --axis axis.json" if (bundle/"axis.json").exists() else ""),
                                         artifacts={p.name:digest(p) for p in bundle.iterdir() if p.is_file()}))
    print(f"exported portable experimental bundle: {bundle}")


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    t=sub.add_parser("audit"); t.add_argument("--geometry",required=True); t.add_argument("--out",required=True); t.add_argument("--z",type=float)
    t=sub.add_parser("train"); t.add_argument("--recipe",default="configs/paris4-sheet704.json")
    for name in ("geometry","sources","cover","resume","out"): t.add_argument("--"+name,required=True)
    t.add_argument("--binary",default=BINARY); t.add_argument("--gpus",default="0,1")
    t.add_argument("--updates",type=int,default=2000); t.add_argument("--variant",type=int,choices=(0,1,2),default=2)
    t.add_argument("--prepare-only",action="store_true")
    t.add_argument("--baseline",action="store_true",help="legacy surface-only control with frozen winding reference")
    t.add_argument("--existing-targets",action="store_true",help="expanded-label/soft=3 control; supply the original sources")
    t=sub.add_parser("resume"); t.add_argument("--run",required=True)
    t=sub.add_parser("reconfigure",help="continue the same coverage plan with explicitly changed augmentation")
    for name in ("run","recipe","out"): t.add_argument("--"+name,required=True)
    t.add_argument("--binary",default=BINARY); t.add_argument("--prepare-only",action="store_true")
    t=sub.add_parser("sweep")
    for name in ("geometry","sources","original-sources","cover","resume","out"): t.add_argument("--"+name,required=True)
    t.add_argument("--recipe",default="configs/paris4-sheet704.json")
    t.add_argument("--binary",default=BINARY); t.add_argument("--gpus",default="0,1")
    t.add_argument("--updates",type=int,default=2000)
    t=sub.add_parser("predict"); t.add_argument("--run",required=True); t.add_argument("--box",required=True)
    t.add_argument("--out",required=True); t.add_argument("--gpu",default="0"); t.add_argument("--window",type=int)
    t=sub.add_parser("extract"); t.add_argument("--prediction",required=True); t.add_argument("--out",required=True)
    t.add_argument("--binary",default=BINARY); t.add_argument("--threshold",type=float,default=.3)
    t.add_argument("--stride",type=int,default=2); t.add_argument("--cell",type=int,default=4)
    t=sub.add_parser("evaluate")
    for name in ("truth","evidence","out"): t.add_argument("--"+name,required=True)
    t.add_argument("--split",choices=("development","test"),default="development")
    t.add_argument("--reconstruction",help="validated reconstruction directory; required for final export")
    t=sub.add_parser("select"); t.add_argument("--baseline",required=True); t.add_argument("--candidates",nargs="+",required=True); t.add_argument("--out",required=True)
    t=sub.add_parser("export")
    for name in ("run","report","reconstruction-report","selection","out"): t.add_argument("--"+name,required=True)
    a=p.parse_args()
    if hasattr(a,"updates") and a.updates<1: p.error("updates must be positive")
    if hasattr(a,"stride") and (a.stride<1 or a.cell<1 or not 0<a.threshold<1): p.error("invalid evidence sampling")
    globals()[a.command](a)


if __name__=="__main__": main()
