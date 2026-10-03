#!/usr/bin/env python3
"""Fit an injective flow of a canonical sheet to sparse winding evidence.

Input NPZ: xyz (native ZYX), q (turns), probability; optional normal.
Output: surface.obj, surface.npz (UV/q/support), deformation.npz and report.
This deliberately separates geometric completion from observed material.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from sheet_geometry import atomic_json, digest, reference_value


def canonical(reference, q, z, direction=1):
    knots = np.asarray(reference["knots"])
    cy, cx, pitch, offset = [np.interp(z, knots[:, 0], knots[:, k]) for k in range(1, 5)]
    radius = (q-offset)*pitch
    if (radius <= 0).any():
        raise ValueError("canonical sheet crosses the axis; restrict the q/z domain")
    theta = direction*2*np.pi*q
    return np.stack([z, cy+radius*np.sin(theta), cx+radius*np.cos(theta)], axis=-1)


def canonical_domain(reference, q_min):
    """Keep every turn in the fitted rectangular domain away from the axis.

    The radial training prior can undershoot the observed inner winding in a
    folded region. Lower only the canonical initializer's offsets; never change
    the model's reference or observed winding labels.
    """
    if not np.isfinite(q_min): raise ValueError("invalid canonical winding domain")
    knots=np.asarray(reference["knots"],float).copy()
    cap=float(q_min)-.5
    knots[:,4]=np.minimum(knots[:,4],cap)
    return dict(reference,knots=knots.tolist(),canonical_offset_cap=cap)


def fitting_bounds(reference, xyz, q_range, z_range):
    """Cover observed evidence and the entire canonical extraction domain."""
    xyz=np.asarray(xyz,float); knots=np.asarray(reference["knots"],float)
    if not np.isfinite(xyz).all() or not np.isfinite([*q_range,*z_range]).all() or q_range[0]>=q_range[1] or z_range[0]>=z_range[1]:
        raise ValueError("invalid reconstruction domain")
    z=np.r_[z_range,knots[(knots[:,0]>z_range[0])&(knots[:,0]<z_range[1]),0]]
    cy,cx,pitch,offset=[np.interp(z,knots[:,0],knots[:,k]) for k in range(1,5)]
    radius=float(pitch.max()*(q_range[1]-offset.min()))
    lo=np.minimum(xyz.min(axis=0),[z_range[0],cy.min()-radius,cx.min()-radius])
    hi=np.maximum(xyz.max(axis=0),[z_range[1],cy.max()+radius,cx.max()+radius])
    pad=np.maximum((hi-lo)*.05,128)
    return lo-pad,hi+pad


def lipschitz_bound(field, lo, hi):
    """Global Frobenius bound on a trilinearly interpolated velocity derivative."""
    size = np.array(field.shape[2:])
    spacing = (np.asarray(hi)-lo)/(size-1)
    bounds = []
    for axis in range(3):
        derivative = torch.diff(field, dim=axis+2).abs().flatten(2).amax(dim=2) / float(spacing[axis])
        bounds.append(derivative)
    return float(torch.sqrt(sum((b*b).sum() for b in bounds)).detach())


def flow(points, field, lo, hi, steps):
    lo_t, hi_t = [points.new_tensor(a) for a in (lo, hi)]
    p = points
    for _ in range(steps):
        grid = 2*(p-lo_t)/(hi_t-lo_t)-1
        # grid_sample coordinates are X,Y,Z, field dimensions are Z,Y,X.
        grid = grid[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
        velocity = F.grid_sample(field, grid, align_corners=True, padding_mode="border")
        velocity = velocity.reshape(3, -1).T
        p = p + velocity/steps
    return p


def _segment_triangle(p, q, tri, eps=1e-8):
    a, b, c = tri
    e1, e2, direction = b-a, c-a, q-p
    h = np.cross(direction, e2)
    det = np.dot(e1, h)
    if abs(det) < eps:
        return False
    inv = 1/det
    s = p-a
    u = np.dot(s, h)*inv
    v = np.dot(direction, np.cross(s, e1))*inv
    t = np.dot(e2, np.cross(s, e1))*inv
    return -eps <= u <= 1+eps and -eps <= v and u+v <= 1+eps and -eps <= t <= 1+eps


def _coplanar_intersection(a, b):
    normal = np.cross(a[1]-a[0], a[2]-a[0])
    if np.max(abs((b-a[0]) @ normal)) > 1e-7*np.linalg.norm(normal):
        return False
    axis = int(np.argmax(abs(normal)))
    a, b = np.delete(a, axis, 1), np.delete(b, axis, 1)
    def orientation(x, y, z):
        d, e = y-x, z-x
        return d[0]*e[1]-d[1]*e[0]
    def contained(p, tri):
        signs = [orientation(tri[i], tri[(i+1)%3], p) for i in range(3)]
        return min(signs) >= -1e-8 or max(signs) <= 1e-8
    if any(contained(p, b) for p in a) or any(contained(p, a) for p in b):
        return True
    for i in range(3):
        for j in range(3):
            x, y, u, v = a[i], a[(i+1)%3], b[j], b[(j+1)%3]
            if orientation(x,y,u)*orientation(x,y,v) < 0 and orientation(u,v,x)*orientation(u,v,y) < 0:
                return True
    return False


def validate_mesh(vertices, faces):
    """Exhaustive spatial-hash broad phase; no sampled self-intersection check."""
    vertices, faces = np.asarray(vertices,np.float64), np.asarray(faces, np.int64)
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("nonfinite or invalid mesh")
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    if (area < 1e-8).any():
        raise ValueError("degenerate mesh triangle")
    edges = np.sort(np.concatenate([faces[:, [0,1]],faces[:, [1,2]],faces[:, [2,0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if counts.max() > 2:
        raise ValueError("nonmanifold edge")
    links=defaultdict(list)
    for a,b,c in faces:
        links[int(a)].append((int(b),int(c)))
        links[int(b)].append((int(c),int(a)))
        links[int(c)].append((int(a),int(b)))
    for pairs in links.values():
        adjacency=defaultdict(set)
        for a,b in pairs: adjacency[a].add(b); adjacency[b].add(a)
        if any(len(n)>2 for n in adjacency.values()): raise ValueError("nonmanifold vertex")
        seen=set(); pending=[next(iter(adjacency))]
        while pending:
            v=pending.pop()
            if v in seen: continue
            seen.add(v); pending.extend(adjacency[v]-seen)
        if len(seen)!=len(adjacency): raise ValueError("nonmanifold vertex")
    size = max(float(np.median(np.linalg.norm(tri[:,1]-tri[:,0],axis=1))), 1.)
    bins = defaultdict(list)
    checked = set()
    for i, t in enumerate(tri):
        low, high = np.floor(t.min(axis=0)/size).astype(int), np.floor(t.max(axis=0)/size).astype(int)
        if np.prod(high-low+1) > 10000:
            raise ValueError("pathological triangle bounding box")
        for z in range(low[0],high[0]+1):
            for y in range(low[1],high[1]+1):
                for x in range(low[2],high[2]+1):
                    cell = (z,y,x)
                    for j in bins[cell]:
                        pair = (j,i)
                        if pair in checked:
                            continue
                        checked.add(pair)
                        a, b = tri[j], t
                        shared=np.intersect1d(faces[j],faces[i])
                        if len(shared)==3: raise ValueError("duplicate mesh triangle")
                        if len(shared)==2:
                            p0,p1=vertices[shared]; e=p1-p0
                            other_a=vertices[next(v for v in faces[j] if v not in shared)]
                            other_b=vertices[next(v for v in faces[i] if v not in shared)]
                            na,nb=np.cross(e,other_a-p0),np.cross(e,other_b-p0)
                            if np.linalg.norm(np.cross(na,nb))<1e-8*np.linalg.norm(na)*np.linalg.norm(nb) and np.dot(na,nb)>0:
                                raise ValueError("self-intersection across a shared edge")
                            continue
                        if len(shared)==1:
                            # Remove only the permitted common contact. Any
                            # remaining intersection is an actual overlap.
                            a=a.mean(axis=0)+(a-a.mean(axis=0))*(1-1e-6)
                            b=b.mean(axis=0)+(b-b.mean(axis=0))*(1-1e-6)
                        if (a.max(axis=0) < b.min(axis=0)).any() or (b.max(axis=0) < a.min(axis=0)).any():
                            continue
                        if any(_segment_triangle(a[k],a[(k+1)%3],b) or _segment_triangle(b[k],b[(k+1)%3],a) for k in range(3)) or _coplanar_intersection(a,b):
                            raise ValueError(f"self-intersection between triangles {j} and {i}")
                    bins[cell].append(i)
    return dict(vertices=len(vertices), triangles=len(faces), nonmanifold_edges=0,nonmanifold_vertices=0,
                self_intersections=0, boundary_edges=int((counts==1).sum()))


def fit(reference, evidence, bounds, iterations=2000, device="cpu", seed=2, grid_sizes=(8,16,32)):
    torch.manual_seed(seed)
    xyz, q = np.asarray(evidence["xyz"],float), np.asarray(evidence["q"],float)
    probability = np.asarray(evidence["probability"],float)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(q) & np.isfinite(probability) & (probability>=.1)
    xyz, q, probability = xyz[valid], q[valid], probability[valid]
    if len(q)<16:
        raise ValueError("not enough supported winding evidence")
    direction = reference.get("winding_direction",1)
    source = torch.tensor(canonical(reference,q,xyz[:,0],direction),dtype=torch.float32,device=device)
    target = torch.tensor(xyz,dtype=torch.float32,device=device)
    weight = torch.tensor(probability,dtype=torch.float32,device=device)
    normals=None
    if "normal" in evidence:
        normal=np.asarray(evidence["normal"],float)[valid]
        if normal.shape!=xyz.shape or not np.isfinite(normal).all():
            raise ValueError("invalid evidence normals")
        normals=torch.tensor(normal,dtype=torch.float32,device=device)
    lo, hi = np.asarray(bounds[0],float), np.asarray(bounds[1],float)
    if (hi<=lo).any():
        raise ValueError("invalid reconstruction bounds")
    extent = torch.tensor(hi-lo,dtype=torch.float32,device=device)
    field = None
    rejected = 0
    if not grid_sizes or any(size<4 or size>512 for size in grid_sizes) or any(a>=b for a,b in zip(grid_sizes,grid_sizes[1:])):
        raise ValueError("deformation grids must increase, with sizes 4..512")
    for size in grid_sizes:
        if field is None:
            initial=torch.zeros((1,3,size,size,size),device=device)
        else:
            initial=F.interpolate(field.detach(),size=(size,)*3,mode="trilinear",align_corners=True)
        field=torch.nn.Parameter(initial)
        optimizer=torch.optim.Adam([field],lr=float(np.min(hi-lo))*.0001)
        for _ in range(max(1,iterations//len(grid_sizes))):
            bound=lipschitz_bound(field,lo,hi)
            steps=max(8,math.ceil(bound/.4))
            if steps>64:
                raise RuntimeError("deformation exceeds safe integration budget")
            ids=torch.randperm(len(source),device=device)[:4096]
            predicted=flow(source[ids],field,lo,hi,steps)
            distance=F.huber_loss(predicted,target[ids],delta=4,reduction="none").mean(dim=1)
            loss=(distance*weight[ids]).sum()/weight[ids].sum()
            smooth=sum(torch.diff(field/extent.reshape(1,3,1,1,1),dim=a+2).square().mean() for a in range(3))
            loss=loss+10*smooth+.001*(field/extent.reshape(1,3,1,1,1)).square().mean()
            # Local tangential stretch regularization. It also supplies an
            # orientation constraint when reliable extracted normals exist.
            small=ids[:min(512,len(ids))]
            qs=q[small.cpu().numpy()]; zs=xyz[small.cpu().numpy(),0]
            cq=torch.tensor(canonical(reference,qs+1e-4,zs,direction),dtype=source.dtype,device=device)
            cz=torch.tensor(canonical(reference,qs,zs+1,direction),dtype=source.dtype,device=device)
            centre=flow(source[small],field,lo,hi,steps)
            tq,tz=flow(cq,field,lo,hi,steps)-centre,flow(cz,field,lo,hi,steps)-centre
            stretch=(torch.log(tq.norm(dim=1).clamp_min(1e-6)/(cq-source[small]).norm(dim=1).clamp_min(1e-6)).square().mean()+
                     torch.log(tz.norm(dim=1).clamp_min(1e-6)).square().mean())
            loss=loss+10*stretch
            if normals is not None:
                predicted_normal=F.normalize(torch.cross(tq,tz,dim=1),dim=1)
                true_normal=F.normalize(normals[small],dim=1)
                supported=normals[small].norm(dim=1)>.5
                if supported.any(): loss=loss+10*(1-(predicted_normal[supported]*true_normal[supported]).sum(dim=1).square()).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite reconstruction objective")
            old=field.detach().clone()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if not torch.isfinite(field).all() or lipschitz_bound(field,lo,hi)>25.6:
                with torch.no_grad(): field.copy_(old)
                optimizer.state.clear(); rejected+=1
    steps=max(8,math.ceil(lipschitz_bound(field,lo,hi)/.4))
    return field.detach(),steps,dict(rejected_updates=rejected,integration_steps=steps,
                                    lipschitz_bound=lipschitz_bound(field,lo,hi),evidence_points=len(q),
                                    grids=list(grid_sizes),field_spacing_native=((hi-lo)/(grid_sizes[-1]-1)).tolist())


def extract(reference, field, steps, bounds, q_range, z_range, edge=64):
    knots=np.asarray(reference["knots"])
    pitch=float(knots[:,3].max())
    radius=max(1.,pitch*(q_range[1]-float(knots[:,4].min())))
    nq=max(3,math.ceil((q_range[1]-q_range[0])*2*np.pi*radius/edge)+1)
    nz=max(2,math.ceil((z_range[1]-z_range[0])/edge)+1)
    q,z=np.meshgrid(np.linspace(*q_range,nq),np.linspace(*z_range,nz))
    xyz=canonical(reference,q.ravel(),z.ravel(),reference.get("winding_direction",1))
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(xyz),65536):
            p=torch.tensor(xyz[start:start+65536],dtype=torch.float32,device=field.device)
            chunks.append(flow(p,field,*bounds,steps).cpu().numpy())
    vertices=np.concatenate(chunks)
    ids=np.arange(nq*nz).reshape(nz,nq)
    a,b,c,d=[v.ravel() for v in (ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:])]
    faces=np.concatenate([np.stack([a,b,c],axis=1),np.stack([b,d,c],axis=1)])
    # UV is a material-coordinate initialization; metric distortion is reported.
    uv=np.stack([q.ravel(),z.ravel()],axis=1)
    return vertices,faces,uv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",required=True); p.add_argument("--reference",required=True)
    p.add_argument("--out",required=True); p.add_argument("--device",default="cpu")
    p.add_argument("--iterations",type=int,default=2000); p.add_argument("--edge",type=float,default=64)
    p.add_argument("--grids",type=int,nargs="+",default=[8,16,32],help="increasing deformation grid sizes; use finer grids after coarse fitting")
    p.add_argument("--q-range",type=float,nargs=2); p.add_argument("--z-range",type=float,nargs=2)
    a=p.parse_args()
    if a.iterations<3 or a.edge<=0: p.error("invalid fitting/extraction limits")
    reference=json.loads(Path(a.reference).read_text()); evidence=dict(np.load(a.evidence))
    xyz=evidence["xyz"]
    qr=a.q_range or [float(np.nanmin(evidence["q"])),float(np.nanmax(evidence["q"]))]
    zr=a.z_range or [float(xyz[:,0].min()),float(xyz[:,0].max())]
    initializer=canonical_domain(reference,min(qr[0],float(np.nanmin(evidence["q"]))))
    bounds=fitting_bounds(initializer,xyz,
        [min(qr[0],float(np.nanmin(evidence["q"]))),max(qr[1],float(np.nanmax(evidence["q"])))],
        [min(zr[0],float(xyz[:,0].min())),max(zr[1],float(xyz[:,0].max()))])
    field,steps,report=fit(initializer,evidence,bounds,a.iterations,a.device,grid_sizes=tuple(a.grids))
    report["canonical_offset_cap"]=initializer["canonical_offset_cap"]
    for refinement in range(3):
        vertices,faces,uv=extract(initializer,field,steps,bounds,qr,zr,a.edge/2**refinement)
        try:
            validation=validate_mesh(vertices,faces)
            break
        except ValueError:
            if refinement==2: raise
    tree=cKDTree(xyz); distance,nearest=tree.query(vertices,workers=1)
    probability=np.asarray(evidence["probability"])[nearest]
    support=np.where((distance<=4)&(probability>=.5),0,np.where(distance<=16,1,2)).astype(np.uint8)
    out=Path(a.out)
    if out.exists(): raise ValueError("choose a new reconstruction directory")
    out.mkdir(parents=True)
    np.savez_compressed(out/"surface.npz",xyz=vertices,faces=faces,uv=uv,q=uv[:,0],support=support,
                        evidence_distance=distance,evidence_probability=probability)
    np.savez_compressed(out/"deformation.npz",velocity=field.cpu().numpy(),bounds=np.array(bounds),steps=steps)
    with (out/"surface.obj").open("w") as f:
        for z,y,x in vertices: f.write(f"v {x:.8g} {y:.8g} {z:.8g}\n")
        for u,v in uv: f.write(f"vt {u:.8g} {v:.8g}\n")
        for face in faces+1: f.write("f "+" ".join(f"{i}/{i}" for i in face)+"\n")
    report.update(validation=validation,reference_sha256=digest(a.reference),evidence_sha256=digest(a.evidence),
                  checkpoint_sha256=str(evidence.get("checkpoint_sha256","")),
                  support_counts=np.bincount(support,minlength=3).tolist(),
                  support_meanings=["observed","uncertain","inferred completion"],
                  uv_units=["turns","native axial voxels"],status="validated experimental reconstruction")
    atomic_json(out/"report.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
