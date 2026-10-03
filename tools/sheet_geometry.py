"""Geometry contract for the experimental surface_winding task.

Coordinates are native CT voxel centres in Z,Y,X order; q is in turns.
The dense binary target remains separate. Only audited mesh relationships
enter the sparse loss. This module has no CUDA or network dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

VERSION = 1
MAX_PATH = 32
RECORD = np.dtype([("kind", "<u4"), ("count", "<u4"), ("weight", "<f4"),
                   ("target", "<f4"), ("points", "<f4", (MAX_PATH, 4))])
INDEX = np.dtype([("cell", "<i4", (3,)), ("count", "<u4"), ("offset", "<u8")])
KINDS = {"coordinate": 0, "continuity": 1, "ordering": 2, "path": 3, "gap": 4}
DEFAULT_WEIGHTS = np.array([.25, .25, .25, .1, .1], dtype=np.float64)
MATCH=np.dtype([("a","<i8"),("b","<i8"),("hit","<f8",(3,)),("q","<f8")])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def inside_boxes(xyz, boxes, guard=0):
    result = np.zeros(np.shape(xyz)[:-1], dtype=bool)
    for box in boxes:
        lo, size = np.asarray(box[:3]), np.asarray(box[3:])
        result |= ((xyz >= lo - guard) & (xyz < lo + size + guard)).all(axis=-1)
    return result


def unwrap_mesh(xyz, valid, axis, edge_limit=256):
    """Lift angular phase on valid grid edges, detecting inconsistent cycles.

    Each connected patch has its own integer gauge. Invalid cells are never
    bridged; cycle-inconsistent components are excluded as a whole.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).copy()
    cy = np.interp(xyz[..., 0], axis[:, 0], axis[:, 1])
    cx = np.interp(xyz[..., 0], axis[:, 0], axis[:, 2])
    dy, dx = xyz[..., 1] - cy, xyz[..., 2] - cx
    phase = np.arctan2(dy, dx) / (2 * np.pi)
    valid &= np.isfinite(xyz).all(axis=-1) & (np.hypot(dy, dx) > 32)
    h, w = valid.shape
    q = np.full((h, w), np.nan)
    component = np.full((h, w), -1, np.int32)
    bad, edges, ncomp = [], [], 0
    for r, c in zip(*np.where(valid)):
        if component[r, c] >= 0:
            continue
        queue = deque([(r, c)])
        component[r, c] = ncomp
        q[r, c] = phase[r, c]
        inconsistent = False
        while queue:
            y, x = queue.popleft()
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if not (0 <= yy < h and 0 <= xx < w and valid[yy, xx]):
                    continue
                distance = np.linalg.norm(xyz[y, x] - xyz[yy, xx])
                if distance > edge_limit or distance < 1e-6:
                    continue
                delta = (phase[yy, xx] - phase[y, x] + .5) % 1 - .5
                proposed = q[y, x] + delta
                if component[yy, xx] < 0:
                    component[yy, xx] = ncomp
                    q[yy, xx] = proposed
                    queue.append((yy, xx))
                elif abs(q[yy, xx] - proposed) > .05:
                    inconsistent = True
                if (yy, xx) > (y, x):
                    edges.append((y*w+x, yy*w+xx))
        if inconsistent:
            bad.append(ncomp)
        ncomp += 1
    if bad:
        q[np.isin(component, bad)] = np.nan
    return q, component, np.asarray(edges, dtype=np.int64).reshape(-1, 2), bad


def mesh_normals(xyz, valid):
    """Normals from intact neighbouring grid triangles, aligned outwards later."""
    n = np.zeros_like(xyz, dtype=np.float64)
    a, b = xyz[1:, :-1] - xyz[:-1, :-1], xyz[:-1, 1:] - xyz[:-1, :-1]
    ok = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:]
    cross = np.cross(a, b)
    cross[~ok] = 0
    n[:-1, :-1] += cross
    n[1:, :-1] += cross
    n[:-1, 1:] += cross
    length = np.linalg.norm(n, axis=-1)
    n /= np.maximum(length[..., None], 1e-12)
    return n, length > 1e-8


def ray_matches(xyz,q,components,normals,faces,max_distance=256,max_rays=120000):
    """Nearest registered-triangle intersections; centroid search is conservative."""
    faces=np.asarray(faces,np.int64); tri=xyz[faces]; centres=tri.mean(axis=1)
    radius=float(np.linalg.norm(tri-centres[:,None],axis=2).max())
    tree=cKDTree(centres); e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    def cast(anchors):
        result=[]
        for i in anchors:
            ids=np.asarray(tree.query_ball_point(xyz[i],max_distance+radius),int)
            if not len(ids): continue
            direction=normals[i]; h=np.cross(direction,e2[ids]); det=np.einsum("ij,ij->i",e1[ids],h)
            inv=np.zeros_like(det); np.divide(1,det,out=inv,where=abs(det)>1e-10)
            s=xyz[i]-tri[ids,0]; u=np.einsum("ij,ij->i",s,h)*inv
            cross=np.cross(s,e1[ids]); v=cross@direction*inv; t=np.einsum("ij,ij->i",e2[ids],cross)*inv
            valid=(abs(det)>1e-10)&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(t>=.25)&(t<=max_distance)
            if not valid.any(): continue
            k=int(np.argmin(np.where(valid,t,np.inf))); face=faces[ids[k]]; bary=np.array([1-u[k]-v[k],u[k],v[k]])
            normal=(normals[face]*bary[:,None]).sum(axis=0)
            normal/=max(np.linalg.norm(normal),1e-12)
            # Reject an ambiguous first surface; never jump to a farther one.
            if np.dot(normal,direction)<.8: continue
            row=np.zeros((),MATCH); row["a"]=i; row["b"]=face[int(np.argmax(bary))]
            row["hit"]=xyz[i]+t[k]*direction; row["q"]=np.dot(q[face],bary)
            result.append(row)
        return np.asarray(result,MATCH)
    # Component normal signs use known within-mesh winding adjacency, avoiding
    # unreliable radial covariance on narrow, folded outer patches.
    calibration=[]
    for component in np.unique(components):
        ids=np.where(components==component)[0]
        calibration.extend(ids[::max(1,len(ids)//128)][:128])
    initial=cast(calibration)
    for component in np.unique(components):
        own=(components[initial["a"]]==component)&(components[initial["b"]]==component)
        delta=initial["q"][own]-q[initial["a"][own]]
        delta=delta[(abs(delta)>.75)&(abs(delta)<1.25)]
        if len(delta)>=4 and (delta<0).mean()>=.9: normals[components==component]*=-1
    anchors=np.arange(0,len(xyz),max(1,math.ceil(len(xyz)/max_rays)))
    return cast(anchors)


def align_components(xyz, q, components, normals, max_distance=256, faces=None):
    """Align integer gauges using conservative adjacent-sheet ray matches.

    Contradictory votes/loops are reported. Disconnected components have no
    global supervision; there is no filename-derived fallback.
    """
    if faces is not None:
        matches=ray_matches(xyz,q,components,normals,faces,max_distance)
        pairs=[]
    else:
        matches=None
        pairs=[]
    if matches is None:
        tree = cKDTree(xyz)
        distances, neighbours = tree.query(xyz, k=min(24, len(xyz)), workers=1)
        if distances.ndim == 1:
            distances, neighbours = distances[:, None], neighbours[:, None]
    votes = {}
    for start in range(0,len(xyz) if matches is None else 0,16384):
        end=min(len(xyz),start+16384)
        ids=neighbours[start:end,1:]; dist=distances[start:end,1:]
        displacement=xyz[ids]-xyz[start:end,None,:]
        cosine=np.einsum("nki,ni->nk",displacement,normals[start:end])/np.maximum(dist,1e-12)
        agreement=np.einsum("nki,ni->nk",normals[ids],normals[start:end])
        matched=(dist>=2)&(dist<=max_distance)&(cosine>=.94)&(agreement>=.8)
        first=matched.argmax(axis=1)
        for row in np.where(matched.any(axis=1))[0]:
            i=start+row; j=int(ids[row,first[row]])
            delta_phase = (q[j] - q[i] + .5) % 1 - .5
            expected = 1 + delta_phase
            a, b = int(components[i]), int(components[j])
            if a != b:
                shift = int(round(expected - (q[j] - q[i])))
                key = (min(a, b), max(a, b))
                votes.setdefault(key, []).append(shift if a < b else -shift)
            pairs.append((i, int(j)))
    if matches is None:
        matches=np.zeros(len(pairs),MATCH)
        if len(pairs):
            pair_array=np.asarray(pairs,int); matches["a"],matches["b"]=pair_array.T
            matches["hit"]=xyz[matches["b"]]; matches["q"]=q[matches["b"]]
    else:
        for row in matches:
            i,j=int(row["a"]),int(row["b"]); a,b=int(components[i]),int(components[j])
            if a==b: continue
            delta=(row["q"]-q[i]+.5)%1-.5
            shift=int(round(1+delta-(row["q"]-q[i])))
            key=min(a,b),max(a,b)
            votes.setdefault(key,[]).append(shift if a<b else -shift)
    graph, rejected = {}, []
    for (a, b), shifts in votes.items():
        vals, counts = np.unique(shifts, return_counts=True)
        k = np.argmax(counts)
        if counts[k] < 4 or counts[k] / len(shifts) < .9:
            rejected.append([a, b, "ambiguous gauge votes"])
            continue
        shift = int(vals[k])
        graph.setdefault(a, []).append((b, shift))
        graph.setdefault(b, []).append((a, -shift))
    anchor = int(components[0])
    offsets, queue, contradictions = {anchor: 0}, deque([anchor]), set()
    while queue:
        a = queue.popleft()
        for b, shift in graph.get(a, []):
            value = offsets[a] + shift
            if b not in offsets:
                offsets[b] = value
                queue.append(b)
            elif offsets[b] != value:
                contradictions.update((a, b))
                rejected.append([a, b, "inconsistent gauge cycle"])
    if contradictions:
        # Every gauge reached from a contradictory cycle depends on an
        # arbitrary spanning tree. Reject the whole anchored component.
        contradictions.update(offsets)
    reliable = np.array([int(c) in offsets and int(c) not in contradictions for c in components])
    aligned = q + np.array([offsets.get(int(c), 0) for c in components])
    if len(matches):
        matches["q"]+=np.array([offsets.get(int(c),0) for c in components[matches["b"]]])
        keep = reliable[matches["a"]]&reliable[matches["b"]]
        dq = matches["q"]-aligned[matches["a"]]
        keep &= (dq > .75) & (dq < 1.25)
        matches = matches[keep]
    return aligned, reliable, matches, rejected


def reference_value(reference, xyz):
    xyz = np.asarray(xyz)
    knots = np.asarray(reference["knots"], dtype=np.float64)
    z = xyz[..., 0]
    cy, cx, pitch, offset = [np.interp(z, knots[:, 0], knots[:, k]) for k in range(1, 5)]
    return np.hypot(xyz[..., 1] - cy, xyz[..., 2] - cx) / pitch + offset


def fit_reference(xyz, q, axis, z_step=2048):
    def robust_fit(p,target):
        cy=np.interp(p[:,0],axis[:,0],axis[:,1]); cx=np.interp(p[:,0],axis[:,0],axis[:,2])
        radius=np.hypot(p[:,1]-cy,p[:,2]-cx)
        design=np.stack([radius,np.ones_like(radius)],axis=1); weights=np.ones_like(radius)
        for _ in range(8):
            slope,offset=np.linalg.lstsq(design*weights[:,None],target*weights,rcond=None)[0]
            error=design@[slope,offset]-target
            weights=np.sqrt(np.minimum(1,.25/np.maximum(abs(error),1e-6)))
        return float(slope),float(offset),radius
    global_slope,_,_=robust_fit(xyz,q)
    if global_slope<=0 or not np.isfinite(global_slope):
        raise ValueError(f"global winding pitch is nonpositive (slope={global_slope}); inspect orientation/gauge audit")
    knots = []
    fallbacks=[]
    zlo, zhi = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    for z in np.linspace(zlo, zhi, max(2, math.ceil((zhi-zlo)/z_step)+1)):
        select = abs(xyz[:, 0] - z) <= z_step
        if select.sum() < 32:
            select[:] = True
        p, target = xyz[select], q[select]
        slope,offset,radius=robust_fit(p,target)
        if not np.isfinite(slope) or not global_slope/8<=slope<=global_slope*8:
            # q0 is only a coarse prior. A folded/poorly covered axial region
            # may have a bad local radial fit; preserve the positive global
            # pitch and fit its local offset without changing any mesh q label.
            fallbacks.append(dict(z=float(z),rejected_slope=slope))
            slope=global_slope; offset=float(np.median(target-slope*radius))
        knots.append([z, float(np.interp(z, axis[:, 0], axis[:, 1])),
                      float(np.interp(z, axis[:, 0], axis[:, 2])), float(1/slope), float(offset)])
    return dict(version=VERSION, units="turns", coordinate_order="zyx", knots=knots,
                input_center=float(np.mean(q)), input_scale=max(1., float(np.ptp(q))),
                global_pitch=1/global_slope,local_fit_fallbacks=fallbacks)


def make_record(kind, xyz, q=None, target=0, weight=1):
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    if not 1 <= len(xyz) <= MAX_PATH:
        raise ValueError("invalid sparse record length")
    record = np.zeros((), dtype=RECORD)
    record["kind"], record["count"] = KINDS.get(kind, kind), len(xyz)
    record["target"], record["weight"] = target, weight
    record["points"][:len(xyz), :3] = xyz
    if q is not None:
        record["points"][:len(xyz), 3] = q
    return record


def write_dataset(out, records, reference, audit, cell_size=512, contacts=None, max_soft_sigma=2):
    out = Path(out)
    if out.exists():
        raise ValueError(f"refusing to overwrite geometry dataset: {out}")
    out.mkdir(parents=True)
    records = np.asarray(records, dtype=RECORD)
    cell = np.floor(records["points"][:, 0, :3] / cell_size).astype(np.int32)
    order = np.lexsort((cell[:, 2], cell[:, 1], cell[:, 0]))
    records, cell = records[order], cell[order]
    unique, starts, counts = np.unique(cell, axis=0, return_index=True, return_counts=True)
    index = np.zeros(len(unique), INDEX)
    index["cell"], index["offset"], index["count"] = unique, starts, counts
    records.tofile(out / "records.bin")
    index.tofile(out / "index.bin")
    contacts=np.asarray(contacts if contacts is not None else [],dtype="<f4").reshape(-1,4)
    if not np.isfinite(contacts).all() or (contacts[:,3]<=0).any():
        raise ValueError("invalid contact exclusions")
    contacts=contacts[np.argsort(contacts[:,0])]
    contacts.tofile(out/"contacts.bin")
    atomic_json(out / "reference.json", reference)
    atomic_json(out / "audit.json", audit)
    files = {name: digest(out / name) for name in ("records.bin", "index.bin", "reference.json", "audit.json", "contacts.bin")}
    manifest = dict(version=VERSION, task="surface_winding", coordinate_order="zyx", units="turns",
                    cell_size=cell_size, record_bytes=RECORD.itemsize, index_bytes=INDEX.itemsize,
                    records=len(records), cells=len(index), files=files, weights=DEFAULT_WEIGHTS.tolist(),
                    contacts=len(contacts),max_soft_sigma=float(max_soft_sigma))
    atomic_json(out / "geometry.json", manifest)
    return manifest


def sparse_loss(values, records, weights=DEFAULT_WEIGHTS):
    """FP64 loss/derivative reference, sampled logits [record,32,2]."""
    values = np.asarray(values, np.float64)
    grad = np.zeros_like(values)
    losses = np.zeros(5)
    counts = np.array([sum(float(r["weight"]) for r in records if r["kind"] == k) for k in range(5)])
    for i, r in enumerate(records):
        k, n = int(r["kind"]), int(r["count"])
        f = float(r["weight"]) / max(counts[k], 1e-12)
        q = values[i, :n, 1]
        if k == 0:
            error = q[0] - r["points"][0, 3]
        elif k in (1, 2):
            error = (q[1] - q[0]) - r["target"]
        else:
            logits = values[i, :n, 0]
            if k == 3:
                # soft minimum of logits; BCE support on the weakest path point
                beta = 4.
                probs = np.exp(-beta * (logits - logits.min()))
                probs /= probs.sum()
                score = logits.min() - np.log(np.exp(-beta*(logits-logits.min())).mean()) / beta
                loss = np.logaddexp(0, -score)
                grad[i, :n, 0] = -(1 / (1 + np.exp(np.clip(score, -700, 700)))) * probs * f * weights[k]
            else:
                loss = np.logaddexp(0, logits).mean()
                grad[i, :n, 0] = 1 / (1 + np.exp(-np.clip(logits, -700, 700))) / n * f * weights[k]
            losses[k] += loss * f
            continue
        delta = .1
        loss = .5*error*error/delta if abs(error) <= delta else abs(error)-.5*delta
        derivative = np.clip(error/delta, -1, 1) * f * weights[k]
        grad[i, 0, 1] += derivative if k == 0 else -derivative
        if k in (1, 2):
            grad[i, 1, 1] += derivative
        losses[k] += loss*f
    return float(losses @ weights), grad, losses
