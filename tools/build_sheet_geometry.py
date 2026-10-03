#!/usr/bin/env python3
"""Build audited, sparse continuous-winding supervision from registered tifxyz.

Example: python tools/build_sheet_geometry.py --meshes-root /path/aws \
 --segments configs/paris4-segments-20260623.txt --axis /path/umbilicus.json \
 --splits /path/splits.json --out /path/paris4-sheet-geometry
Splits must contain development and test lists of [z,y,x,nz,ny,nx].
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from sheet_geometry import (align_components, atomic_json, digest, fit_reference,
                            inside_boxes, make_record, mesh_normals, unwrap_mesh,
    write_dataset)


def sample_edge(raw, a, b, stride, width, axis, mask=None):
    """Follow the original registered grid, rather than a decimated XYZ chord."""
    ia=np.array(divmod(int(a),width))*stride
    ib=np.array(divmod(int(b),width))*stride
    samples=np.linspace(ia,ib,32)
    low=np.floor(samples).astype(int); high=np.minimum(low+1,np.array(raw.shape[:2])-1)
    f=samples-low
    points=np.zeros((32,3),float); valid=np.ones(32,bool)
    for dy in (0,1):
        for dx in (0,1):
            y=np.where(dy,high[:,0],low[:,0]); x=np.where(dx,high[:,1],low[:,1])
            values=raw[y,x]
            weights=(f[:,0] if dy else 1-f[:,0])*(f[:,1] if dx else 1-f[:,1])
            used=weights>1e-7
            valid &= ~used | (np.isfinite(values).all(axis=1)&(values>0).all(axis=1))
            if mask is not None: valid &= ~used | (mask[y,x]>0)
            points+=np.where(used[:,None],values,0)*weights[:,None]
    if not valid.all(): return None
    # Restrict paths to 62 native voxels: 32 samples, <=2 voxels apart.
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if arc[-1]<8: return None
    distance=np.linspace(0,min(62.,arc[-1]),32)
    points=np.stack([np.interp(distance,arc,points[:,d]) for d in range(3)],axis=1)
    cy,cx=[np.interp(points[:,0],axis[:,0],axis[:,d]) for d in (1,2)]
    phase=np.arctan2(points[:,1]-cy,points[:,2]-cx)/(2*np.pi)
    return points,phase


def load_axis(path):
    value = json.loads(Path(path).read_text())
    rows = value.get("control_points", value.get("points", []))
    if not rows:
        raise ValueError("axis has no control_points")
    axis = np.array([[p["z"], p["y"], p["x"]] for p in rows], float)
    axis = axis[np.argsort(axis[:, 0])]
    if len(axis) < 2 or not np.isfinite(axis).all() or (np.diff(axis[:, 0]) <= 0).any():
        raise ValueError("invalid axis")
    return axis


def build(args):
    segments = [s.strip().rstrip("/") for s in Path(args.segments).read_text().splitlines()
                if s.strip() and not s.lstrip().startswith("#")]
    splits = json.loads(Path(args.splits).read_text())
    if not splits.get("development") or not splits.get("test"):
        raise ValueError("reserve development and final-test regions first")
    axis = load_axis(args.axis)
    held = splits["development"] + splits["test"]
    points, phases, normals, components, edges, paths, faces = [], [], [], [], [], [], []
    audit = dict(version=1, splits=splits, guard=args.guard, meshes=[], rejected_relationships=[],
                 scan=dict(scroll="PHercParis4",volume=args.volume,native_um=2.4),
                 axis_sha256=digest(args.axis),segments_sha256=digest(args.segments))
    base, comp_base = 0, 0
    evaluation=[]
    for segment in segments:
        candidates = sorted(Path(args.meshes_root).glob(f"PHercParis4/segments/{segment}/mesh/*-on-{args.volume}-2.4um.tifxyz"))
        if len(candidates) != 1:
            raise ValueError(f"{segment}: expected one registered mesh, got {len(candidates)}")
        root = candidates[0]
        raw = np.stack([tifffile.imread(root / f"{a}.tif", maxworkers=2) for a in "zyx"], axis=-1)
        stride = max(args.stride, int(np.ceil(np.sqrt(raw.shape[0]*raw.shape[1]/args.max_mesh_points))))
        xyz = raw[::stride, ::stride].astype(float)
        valid = np.isfinite(xyz).all(axis=-1) & (xyz > 0).all(axis=-1)
        raw_mask=tifffile.imread(root/"mask.tif",maxworkers=2) if (root/"mask.tif").exists() else None
        if raw_mask is not None: valid &= raw_mask[::stride,::stride]>0
        # Split BEFORE lifting/alignment/reference fitting. Withheld geometry is
        # retained only in the test artifact, never in native training records.
        training = valid & ~inside_boxes(xyz, held, args.guard)
        q, comp, edge, bad = unwrap_mesh(xyz, training, axis, edge_limit=args.edge_limit)
        normal, good_normal = mesh_normals(xyz, training)
        cy, cx = [np.interp(xyz[..., 0], axis[:, 0], axis[:, d]) for d in (1, 2)]
        radial = np.stack([np.zeros_like(cy), xyz[..., 1]-cy, xyz[..., 2]-cx], axis=-1)
        for c in np.unique(comp[comp >= 0]):
            choose = comp == c
            if np.sum(normal[choose]*radial[choose]) < 0:
                normal[choose] *= -1
        good = training & np.isfinite(q) & good_normal
        # Evaluation-only lift never participates in gauge alignment or fitting.
        eq, ec, ee, eb = unwrap_mesh(xyz,valid,axis,edge_limit=args.edge_limit)
        # Evaluation keeps the original grid density; decimated training points
        # cannot resolve narrow gaps or provide meaningful long-track metrics.
        raw_held=inside_boxes(raw,held)&np.isfinite(raw).all(axis=-1)&(raw>0).all(axis=-1)
        if raw_mask is not None: raw_held &= raw_mask>0
        rr,cc=np.where(raw_held); coarse_r=np.minimum(np.rint(rr/stride).astype(int),len(xyz)-1)
        coarse_c=np.minimum(np.rint(cc/stride).astype(int),xyz.shape[1]-1)
        eval_ids=np.full(raw.shape[:2],-1,np.int32); eval_ids[rr,cc]=np.arange(len(rr))
        dense_edges=[]
        for dr,dc in ((1,0),(0,1)):
            nr,nc=rr+dr,cc+dc; inside=(nr<raw.shape[0])&(nc<raw.shape[1])
            a=np.where(inside)[0]; b=eval_ids[nr[inside],nc[inside]]; keep=b>=0
            dense_edges.extend(np.column_stack([a[keep],b[keep]]))
        eval_xyz=raw[rr,cc].astype(float)
        ey,ex=[np.interp(eval_xyz[:,0],axis[:,0],axis[:,d]) for d in (1,2)]
        phase=np.arctan2(eval_xyz[:,1]-ey,eval_xyz[:,2]-ex)/(2*np.pi)
        evaluation.append(dict(q=eq,component=ec,training_good=good,training_base=base,
            dense_xyz=eval_xyz,dense_phase=phase,dense_component=ec[coarse_r,coarse_c],
            dense_q=eq[coarse_r,coarse_c],dense_edges=np.asarray(dense_edges,np.int64).reshape(-1,2)))
        ids = np.full(good.size, -1, np.int64)
        ids[good.ravel()] = np.arange(good.sum()) + base
        edge = edge[(ids[edge] >= 0).all(axis=1)]
        edges.extend(ids[edge].tolist())
        grid=ids.reshape(good.shape)
        a,b,c,d=[x.ravel() for x in (grid[:-1,:-1],grid[1:,:-1],grid[:-1,1:],grid[1:,1:])]
        local_faces=np.concatenate([np.column_stack([a,b,c]),np.column_stack([b,d,c])])
        faces.append(local_faces[(local_faces>=0).all(axis=1)])
        # Sparse path supervision follows the original high-resolution grid.
        h, w = good.shape
        for a,b in edge[::4]:
            sampled=sample_edge(raw,a,b,stride,w,axis,raw_mask)
            if sampled is None: continue
            pp,phase=sampled
            if inside_boxes(pp,held,args.guard).any(): continue
            delta=(phase-phase[0]+.5)%1-.5
            paths.append((int(ids[a]),pp,q.ravel()[a]+delta))
        points.append(xyz[good]); phases.append(q[good]); normals.append(normal[good])
        components.append(comp[good] + comp_base)
        audit["meshes"].append(dict(segment=segment, stride=stride, points=int(good.sum()),
                                     components=int(comp.max()+1), excluded_cycles=bad,
                                     hashes={a: digest(root/f"{a}.tif") for a in "zyx"}))
        print(f"{segment}: {good.sum()} training points; {len(bad)} inconsistent components", flush=True)
        base += int(good.sum()); comp_base += int(comp.max()+1)
    xyz, q, normal, comp = [np.concatenate(a) for a in (points, phases, normals, components)]
    if len(xyz) < 32:
        raise ValueError("not enough audited training geometry")
    # Winding direction is selected geometrically, consistently for all patches.
    signs = []
    for c in np.unique(comp):
        ids = comp == c
        if ids.sum() < 32 or np.ptp(q[ids]) < .5:
            continue
        cy, cx = [np.interp(xyz[ids, 0], axis[:, 0], axis[:, d]) for d in (1, 2)]
        radius = np.hypot(xyz[ids, 1]-cy, xyz[ids, 2]-cx)
        signs.append((float(np.ptp(q[ids])),float(np.cov(radius,q[ids])[0,1])))
    # A narrow folded patch can have misleading radial covariance. Use the
    # widest connected lift to orient the globally common angular coordinate.
    direction = -1 if signs and max(signs)[1] < 0 else 1
    q *= direction
    faces=np.concatenate(faces)
    # Reject triangles spanning invalid grid topology or excessive mesh edges.
    lengths=np.linalg.norm(xyz[faces[:,[1,2,0]]]-xyz[faces],axis=2)
    faces=faces[(lengths<=args.edge_limit).all(axis=1)&(comp[faces]==comp[faces[:,0],None]).all(axis=1)]
    q, reliable, pairs, rejected = align_components(xyz, q, comp, normal, args.neighbour_distance,faces=faces)
    audit.update(winding_direction=direction, points=len(xyz), reliable_points=int(reliable.sum()),
                 ordered_pairs=len(pairs), rejected_relationships=rejected,
                 unanchored_components=np.unique(comp[~reliable]).tolist(),
                 alignment_method="nearest registered triangle along calibrated mesh normal")
    atomic_json(Path(args.out).with_suffix(".audit.json"),audit)
    np.savez_compressed(Path(args.out).with_suffix(".alignment.npz"),xyz=xyz,q=q,
                        normal=normal,component=comp,reliable=reliable)
    if reliable.sum() < 32:
        atomic_json(Path(args.out).with_suffix(".rejected.json"),audit)
        raise ValueError("geometry gauge cannot be established; inspect source meshes")
    reference = fit_reference(xyz[reliable], q[reliable], axis)
    audit["reference_fit_fallbacks"]=reference["local_fit_fallbacks"]
    reference["winding_direction"]=direction
    surface_tree=cKDTree(xyz)
    contacts=[]
    for row in pairs:
        a,b=int(row["a"]),int(row["b"]); hit=row["hit"]
        distance=np.linalg.norm(hit-xyz[a])
        if distance<8:
            contacts.append([*(.5*(xyz[a]+hit)),8.])
    contacts=np.asarray(contacts,float).reshape(-1,4)
    def contact_free(points):
        if contact_tree is None: return True
        return bool((contact_tree.query(np.asarray(points),workers=1)[0]>=8).all())
    # Contact neighbourhoods carry no confident regression/path/gap targets.
    contact_tree=cKDTree(contacts[:,:3]) if len(contacts) else None
    if contact_tree is not None:
        distance,nearest=contact_tree.query(xyz,workers=1)
        reliable &= distance>=contacts[nearest,3]
    records = [make_record("coordinate", [p], [t]) for p, t in zip(xyz[reliable], q[reliable])]
    for a, b in edges:
        if reliable[a] and reliable[b]:
            records.append(make_record("continuity", xyz[[a, b]], q[[a, b]], q[b]-q[a]))
    for row in pairs:
        a,b=int(row["a"]),int(row["b"]); hit=row["hit"]; hit_q=float(row["q"])
        if not (reliable[a] and reliable[b]): continue
        records.append(make_record("ordering", [xyz[a],hit], [q[a],hit_q], hit_q-q[a]))
        # Require resolvable positive space; exact contacts never get gap targets.
        distance = np.linalg.norm(hit-xyz[a])
        if distance >= 8:
            p = .5*(xyz[a]+hit)
            # A third observed sheet must not occupy the proposed negative.
            clearance=surface_tree.query(p)[0]
            if clearance>=max(2.,.2*distance) and contact_free([p]):
                records.append(make_record("gap", [p]))
    original_q=np.concatenate(phases)*direction
    for anchor,pp,local_q in paths:
        if reliable[anchor] and contact_free(pp):
            qq=local_q*direction+(q[anchor]-original_q[anchor])
            records.append(make_record("path",pp,qq))
    audit["records_by_kind"] = {str(k): sum(int(r["kind"]) == k for r in records) for k in range(5)}
    audit.update(contact_exclusions=len(contacts),max_soft_sigma=2.,
                 soft_width_policy="uniform sigma <= 2 native voxels; <= quarter of every resolvable audited gap; exclude gaps <8")
    manifest = write_dataset(args.out, records, reference, audit,contacts=contacts,max_soft_sigma=2.)
    np.savez_compressed(Path(args.out)/"training-mesh.npz", xyz=xyz[reliable], q=q[reliable],
                        normal=normal[reliable], component=comp[reliable])
    eval_xyz,eval_q,eval_edges,eval_region=[],[],[],[]
    eval_base=0
    for item in evaluation:
        fullq=item["q"]*direction
        good=item["training_good"]
        common=np.arange(good.sum())+item["training_base"]
        train_component=item["component"][good]
        shifts={}
        for c in np.unique(train_component):
            use=(train_component==c)&reliable[common]
            if use.any():
                shift=q[common[use]]-fullq[good][use]
                rounded=round(float(np.median(shift)))
                if np.max(abs(shift-rounded))<.05: shifts[int(c)]=rounded
        allowed=np.isin(item["dense_component"],list(shifts)) & np.isfinite(item["dense_q"])
        dense_q=item["dense_q"]*direction
        dense_q=dense_q+(item["dense_phase"]*direction-dense_q+.5)%1-.5
        dense_q+=np.array([shifts.get(int(c),0) for c in item["dense_component"]])
        ids=np.full(len(allowed),-1,np.int64); ids[allowed]=np.arange(allowed.sum())+eval_base
        edges=item["dense_edges"]; edges=edges[(ids[edges]>=0).all(axis=1)]
        eval_xyz.append(item["dense_xyz"][allowed]); eval_q.append(dense_q[allowed]); eval_edges.extend(ids[edges])
        regions=np.full(allowed.shape,-1,np.int8)
        for split_id,name in enumerate(("development","test")):
            regions[inside_boxes(item["dense_xyz"],splits[name])]=split_id
        eval_region.append(regions[allowed]); eval_base+=int(allowed.sum())
    np.savez_compressed(Path(args.out)/"evaluation-mesh.npz",xyz=np.concatenate(eval_xyz),q=np.concatenate(eval_q),
                        edges=np.array(eval_edges,np.int64).reshape(-1,2),region=np.concatenate(eval_region))
    atomic_json(Path(args.out)/"splits.json", splits)
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meshes-root", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument("--axis", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--volume", default="20260411134726")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-mesh-points", type=int, default=200000)
    p.add_argument("--edge-limit", type=float, default=512)
    p.add_argument("--neighbour-distance", type=float, default=256)
    p.add_argument("--guard", type=int, default=704)
    args = p.parse_args()
    if args.stride < 1 or args.max_mesh_points < 32 or args.guard < 0:
        p.error("invalid geometry sampling bounds")
    build(args)


if __name__ == "__main__":
    main()
