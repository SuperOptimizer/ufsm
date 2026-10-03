#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from sheet_pipeline import freeze_geometry, score_geometry, decompress_zstd, mesh_evidence, export, select, sweep, train, verify_inputs, reconfigure, extend
from sheet_geometry import make_record, write_dataset, digest
from build_sheet_geometry import build, sample_edge
from sheet_reconstruct import fit, extract, validate_mesh, canonical, canonical_domain, fitting_bounds
import argparse
import tifffile


class PipelineTests(unittest.TestCase):
    def test_completed_cover_extension_preserves_payload_and_excludes_holdouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); previous=root/"previous"; inputs=previous/"inputs"; inputs.mkdir(parents=True)
            (inputs/"geometry").mkdir(); (previous/"model").mkdir()
            (inputs/"ufsm").write_bytes(b"old binary"); (inputs/"axis.json").write_text("{}")
            (inputs/"recipe.json").write_text(json.dumps(dict(train={"lr":.001},predict={})))
            (inputs/"sources.json").write_text(json.dumps(dict(sources=[dict(axis=str(inputs/"axis.json"),holdout=[1000]*3+[32]*3)])))
            (inputs/"geometry/audit.json").write_text(json.dumps(dict(splits=dict(development=[[1000]*3+[32]*3],test=[[2000]*3+[32]*3]),guard=32)))
            plan=dict(version=1,P=32,level=0,count=2,source_names=["fixture"],tiles=[[0,0,0,0],[0,32,0,0]])
            (inputs/"cover.json").write_text(json.dumps(plan)); (inputs/"resume.ckpt").write_bytes(b"donor")
            saved=dict(step=5,extra=dict(cover=dict(sha256=digest(inputs/"cover.json"),count=2,cursor=2,base_step=3),sheet=dict(geometry_sha256="a"*64,schedule_start=3)))
            payload=("UFSM"+json.dumps(saved)+"\n").encode()+b"weights, optimizer and EMA"
            (previous/"model/last.ckpt").write_bytes(payload)
            state=dict(task="surface_winding",status="trained",updates=2,geometry_sha256="a"*64,
                command=[str(inputs/"ufsm"),"train",str(inputs/"sources.json"),"--out",str(previous/"model"),
                    "--resume",str(inputs/"resume.ckpt"),"--cover",str(inputs/"cover.json"),"--steps","2","--sheet-init","1"],
                inputs={str(p.relative_to(inputs)):digest(p) for p in inputs.rglob("*") if p.is_file()})
            (previous/"state.json").write_text(json.dumps(state))
            new_plan=dict(plan,count=3,tiles=plan["tiles"]+[[0,64,0,0]])
            new_cover=root/"cover.json"; new_cover.write_text(json.dumps(new_plan))
            binary=root/"binary"; binary.write_bytes(b"new binary")
            args=argparse.Namespace(run=previous,cover=new_cover,binary=binary,out=root/"extended",prepare_only=True)
            extend(args)
            result=json.loads((args.out/"state.json").read_text()); verify_inputs(args.out,result)
            self.assertEqual((args.out/"inputs/resume.ckpt").read_bytes(),payload)
            self.assertEqual(result["coverage_extension"]["base_step"],3)
            self.assertEqual(result["coverage_extension"]["added_updates"],1)
            self.assertNotIn("--sheet-init",result["command"])
            self.assertIn("--cover-extend-from",result["command"])
            args.out=root/"invalid"
            new_plan["tiles"][0]=[0,1,0,0]; new_cover.write_text(json.dumps(new_plan))
            with self.assertRaisesRegex(ValueError,"unchanged prefix"): extend(args)
            new_plan["tiles"][0]=plan["tiles"][0]; new_plan["tiles"][-1]=[0,950,1000,1000]; new_cover.write_text(json.dumps(new_plan))
            with self.assertRaisesRegex(ValueError,"guarded geometry holdout"): extend(args)

    def test_augmentation_reconfigure_keeps_cursor_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); previous=root/"previous"; inputs=previous/"inputs"; inputs.mkdir(parents=True)
            (previous/"model").mkdir()
            (inputs/"ufsm").write_bytes(b"old binary")
            (inputs/"sources.json").write_text(json.dumps(dict(sources=[dict(axis=str(inputs/"axis.json"))])))
            (inputs/"axis.json").write_text("{}")
            (inputs/"recipe.json").write_text(json.dumps(dict(train=dict(soft=2),predict={})))
            (inputs/"cover.json").write_text(json.dumps(dict(count=4,tiles=[[0,0,0,0]]*4)))
            (inputs/"resume.ckpt").write_bytes(b"initial donor")
            saved=dict(step=5,extra=dict(cover=dict(sha256=digest(inputs/"cover.json"),count=4,cursor=2,base_step=3),
                sheet=dict(geometry_sha256="a"*64,schedule_start=3)))
            (previous/"model/last.ckpt").write_bytes(("UFSM "+json.dumps(saved)+"\n").encode()+b"optimizer payload")
            state=dict(task="surface_winding",status="interrupted",updates=4,geometry_sha256="a"*64,
                command=[str(inputs/"ufsm"),"train",str(inputs/"sources.json"),"--out",str(previous/"model"),
                    "--resume",str(inputs/"resume.ckpt"),"--cover",str(inputs/"cover.json"),"--steps","4","--soft","2","--sheet-init","1"],
                inputs={p.name:digest(p) for p in inputs.iterdir()})
            (previous/"state.json").write_text(json.dumps(state))
            recipe=root/"recipe.json"; recipe.write_text(json.dumps(dict(train={"soft":1.75,"geometry-aug":1,"axis-jitter":2},predict={})))
            binary=root/"ufsm"; binary.write_bytes(b"new binary")
            args=argparse.Namespace(run=previous,recipe=recipe,binary=binary,out=root/"continued",prepare_only=True)
            reconfigure(args)
            continued=json.loads((args.out/"state.json").read_text()); verify_inputs(args.out,continued)
            self.assertNotIn("--sheet-init",continued["command"])
            self.assertEqual(continued["augmentation_change"]["cursor"],2)
            self.assertEqual(continued["augmentation_change"]["step"],5)
            self.assertEqual((args.out/"inputs/resume.ckpt").read_bytes(),(previous/"model/last.ckpt").read_bytes())
            self.assertIn("--geometry-aug",continued["command"])
            self.assertEqual(continued["geometry_sha256"],state["geometry_sha256"])
            recipe.write_text(json.dumps(dict(train={"lr":.1},predict={}))); args.out=root/"invalid"
            with self.assertRaisesRegex(ValueError,"augmentation only"): reconfigure(args)

    def test_sweep_freezes_one_live_donor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reference=dict(version=1,units="turns",coordinate_order="zyx",
                knots=[[0,50,50,20,0],[100,50,50,20,0]],input_center=0,input_scale=1)
            audit=dict(splits=dict(development=[[1000]*3+[32]*3],test=[[2000]*3+[32]*3]),guard=0)
            write_dataset(root/"geometry",[make_record("coordinate",[[10,50,70]],[1])],reference,audit)
            labels=root/"labels"; labels.mkdir()
            (labels/"provenance.json").write_text(json.dumps(dict(surface_band_chamfer=0)))
            source=dict(sources=[dict(root=str(root),ct="fixture",um=2.4,
                targets=dict(surface=dict(root=str(labels))))])
            (root/"sources.json").write_text(json.dumps(source))
            (root/"cover.json").write_text(json.dumps(dict(P=32,tiles=[[0,0,0,0],[0,32,0,0]])))
            (root/"recipe.json").write_text(json.dumps(dict(train={},predict={})))
            binary=root/"ufsm"; binary.write_bytes(b"fixture binary")
            donor=root/"live.ckpt"; donor.write_bytes(b"initial donor")
            args=argparse.Namespace(geometry=root/"geometry/geometry.json",sources=root/"sources.json",
                original_sources=root/"sources.json",cover=root/"cover.json",resume=donor,
                recipe=root/"recipe.json",binary=binary,gpus="0,1",updates=2,out=root/"sweep")
            def prepare_and_replace(a):
                train(a)
                donor.write_bytes(b"new live production checkpoint")
            with patch("sheet_pipeline.train",side_effect=prepare_and_replace): sweep(args)
            manifest=json.loads((root/"sweep/sweep.json").read_text())
            for name in ("expanded","thin","winding","full"):
                path=root/"sweep"/name; state=json.loads((path/"state.json").read_text())
                self.assertEqual(state["donor_sha256"],manifest["donor_sha256"])
                self.assertEqual((path/"inputs/resume.ckpt").read_bytes(),b"initial donor")
                self.assertEqual(state["status"],"prepared")
                verify_inputs(path,state)

    def test_build_real_tifxyz_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); name="20260623141924-w010-027"
            mesh=root/f"aws/PHercParis4/segments/{name}/mesh/20260623141924-on-20260411134726-2.4um.tifxyz"
            mesh.mkdir(parents=True)
            q,z=np.meshgrid(np.linspace(.1,3.1,161),np.linspace(100,300,11))
            r=80+60*q
            xyz=np.stack([z,500+r*np.sin(2*np.pi*q),500+r*np.cos(2*np.pi*q)],axis=-1)
            for d,axis in enumerate("zyx"): tifffile.imwrite(mesh/f"{axis}.tif",xyz[...,d].astype(np.float32))
            (root/"segments.txt").write_text(name+"\n")
            (root/"axis.json").write_text(json.dumps({"control_points":[dict(z=0,y=500,x=500),dict(z=500,y=500,x=500)]}))
            splits=dict(development=[[100,500,580,50,100,100]],test=[[240,400,500,60,100,100]])
            (root/"splits.json").write_text(json.dumps(splits))
            args=argparse.Namespace(segments=root/"segments.txt",splits=root/"splits.json",axis=root/"axis.json",
                meshes_root=root/"aws",volume="20260411134726",guard=0,stride=1,max_mesh_points=10000,
                edge_limit=512,neighbour_distance=256,out=root/"geometry")
            build(args)
            manifest=json.loads((root/"geometry/geometry.json").read_text())
            self.assertGreater(manifest["records"],100)
            freeze_geometry(root/"geometry/geometry.json",root/"frozen")
            self.assertEqual((root/"frozen/geometry.json").read_bytes(),(root/"geometry/geometry.json").read_bytes())

    def test_fit_and_validate_identity(self):
        reference=dict(version=1,units="turns",coordinate_order="zyx",knots=[[0,500,500,30,-2],[100,500,500,30,-2]],input_center=0,input_scale=3,winding_direction=1)
        q,z=np.meshgrid(np.linspace(.1,1.1,32),np.linspace(10,90,8))
        xyz=canonical(reference,q.ravel(),z.ravel())
        evidence=dict(xyz=xyz,q=q.ravel(),probability=np.ones(q.size))
        bounds=(xyz.min(axis=0)-20,xyz.max(axis=0)+20)
        field,steps,report=fit(reference,evidence,bounds,iterations=3)
        vertices,faces,uv=extract(reference,field,steps,bounds,[.1,1.1],[10,90],edge=40)
        self.assertEqual(validate_mesh(vertices,faces)["self_intersections"],0)
        self.assertLess(report["lipschitz_bound"] / steps,1)

    def test_folded_prior_keeps_inner_canonical_radius_positive(self):
        reference=dict(version=1,units="turns",coordinate_order="zyx",
            knots=[[0,500,500,30,-2],[100,500,500,30,0]],input_center=0,input_scale=3)
        original=json.dumps(reference,sort_keys=True)
        initializer=canonical_domain(reference,-3)
        q,z=np.meshgrid(np.linspace(-3,-2,32),np.linspace(10,90,8))
        xyz=canonical(initializer,q.ravel(),z.ravel())
        self.assertGreaterEqual(np.hypot(xyz[:,1]-500,xyz[:,2]-500).min(),15-1e-9)
        self.assertEqual(json.dumps(reference,sort_keys=True),original)
        bounds=(xyz.min(axis=0)-20,xyz.max(axis=0)+20)
        field,steps,_=fit(initializer,dict(xyz=xyz,q=q.ravel(),probability=np.ones(q.size)),bounds,iterations=3)
        vertices,faces,_=extract(initializer,field,steps,bounds,[-3,-2],[10,90],edge=20)
        self.assertEqual(validate_mesh(vertices,faces)["self_intersections"],0)

    def test_reconstruction_field_covers_canonical_and_observed_points(self):
        reference=dict(knots=[[0,500,500,30,-2],[100,510,490,35,0]])
        initializer=canonical_domain(reference,-3)
        observed=np.array([[10,520,520],[90,550,550]],float)
        lo,hi=fitting_bounds(initializer,observed,[-3,10],[10,90])
        q,z=np.meshgrid(np.linspace(-3,10,1000),np.linspace(10,90,11))
        points=canonical(initializer,q.ravel(),z.ravel())
        self.assertTrue(((points>=lo)&(points<=hi)).all())
        self.assertTrue(((observed>=lo)&(observed<=hi)).all())

    def test_tracking_switch_metric(self):
        xyz=np.array([[0,0,x] for x in range(0,160,8)],float)
        truth=dict(xyz=xyz,q=np.linspace(0,.1,len(xyz)),region=np.zeros(len(xyz),int),
                   edges=np.array([[i,i+1] for i in range(len(xyz)-1)]))
        evidence=dict(xyz=xyz.copy(),q=truth["q"].copy(),probability=np.ones(len(xyz)))
        baseline=score_geometry(truth,evidence,0)
        evidence["q"][10:]+=1
        switched=score_geometry(truth,evidence,0)
        self.assertEqual(baseline["winding_switch_frequency"],0)
        self.assertGreater(switched["winding_switch_frequency"],0)
        self.assertLess(switched["median_correct_track_length"],baseline["median_correct_track_length"])

    def test_isolated_supported_points_do_not_form_a_track(self):
        xyz=np.array([[0,0,x] for x in range(0,160,8)],float)
        truth=dict(xyz=xyz,q=np.zeros(len(xyz)),region=np.zeros(len(xyz),int),
            edges=np.array([[i,i+1] for i in range(len(xyz)-1)]))
        evidence=dict(xyz=xyz,q=truth["q"],probability=np.ones(len(xyz)),
            affinity_edges=truth["edges"],affinity_probability=np.zeros(len(xyz)-1))
        disconnected=score_geometry(truth,evidence,0)
        self.assertEqual(disconnected["supported_coverage"],1)
        self.assertEqual(disconnected["median_correct_track_length"],0)
        evidence["affinity_probability"][:]=1
        connected=score_geometry(truth,evidence,0)
        self.assertGreater(connected["median_correct_track_length"],100)

    def test_triangle_winding_interpolation(self):
        mesh=dict(xyz=np.array([[0,0,0],[0,10,0],[0,0,10]],float),faces=np.array([[0,1,2]]),
                  q=np.array([0,1,2]),evidence_probability=np.ones(3))
        result=mesh_evidence(np.array([[2,2,3],[0,12,12]],float),mesh)
        np.testing.assert_allclose(result["xyz"],[[0,2,3],[0,5,5]],atol=1e-8)
        np.testing.assert_allclose(result["q"],[.8,1.5],atol=1e-8)

    def test_surface_only_bridges_use_probability_affinity(self):
        xyz=np.array([[0,y,x] for y in (0,10) for x in range(0,80,4)],float)
        q=np.r_[np.zeros(20),np.ones(20)]
        truth=dict(xyz=xyz,q=q,region=np.zeros(40,int),edges=np.array([[i,i+1] for i in range(39) if i!=19]))
        pairs=np.array([[i,i+20] for i in range(20)])
        evidence=dict(xyz=xyz,q=q,uses_winding=False,affinity_edges=pairs,affinity_probability=np.ones(20),threshold=.3)
        connected=score_geometry(truth,evidence,0)
        self.assertGreater(connected["false_bridge_frequency"],0)
        evidence["affinity_probability"][:]=0
        separated=score_geometry(truth,evidence,0)
        self.assertEqual(separated["false_bridge_frequency"],0)
        evidence["affinity_probability"][:]=1; evidence["uses_winding"]=True
        ordered=score_geometry(truth,evidence,0)
        self.assertEqual(ordered["false_bridge_frequency"],0)

    def test_original_grid_path_and_mask(self):
        theta=np.linspace(0,.4,21); z=np.array([50,51])
        raw=np.stack(np.meshgrid(z,theta,indexing="ij"),axis=-1)
        raw=np.stack([raw[...,0],500+100*np.sin(raw[...,1]),500+100*np.cos(raw[...,1])],axis=-1)
        axis=np.array([[0,500,500],[100,500,500]])
        pp,phase=sample_edge(raw,0,1,20,2,axis)
        self.assertLess(np.max(abs(np.hypot(pp[:,1]-500,pp[:,2]-500)-100)),.01)
        self.assertGreater(np.ptp(phase),.06)
        mask=np.ones((2,21),np.uint8); mask[:,10]=0
        self.assertIsNone(sample_edge(raw,0,1,20,2,axis,mask))

    def test_export_rejects_other_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/"run"; (run/"inputs/geometry").mkdir(parents=True); (run/"model").mkdir()
            (run/"model/last.ckpt").write_bytes(b"fixture")
            ref=run/"inputs/geometry/reference.json"; ref.write_text("{}")
            cksha=digest(run/"model/last.ckpt"); refsha=digest(ref)
            (run/"state.json").write_text(json.dumps(dict(status="trained",task="surface_winding",inputs={},checkpoint_sha256=cksha)))
            (root/"report.json").write_text(json.dumps(dict(split="test",checkpoint_sha256=cksha,reference_sha256=refsha,evidence_sha256="e")))
            (root/"reconstruction.json").write_text(json.dumps(dict(validation=dict(self_intersections=0),checkpoint_sha256=cksha,reference_sha256=refsha,evidence_sha256="e")))
            (root/"selection.json").write_text(json.dumps(dict(status="development candidate selected",checkpoint_sha256="another checkpoint")))
            args=argparse.Namespace(run=run,report=root/"report.json",reconstruction_report=root/"reconstruction.json",selection=root/"selection.json",out=root/"bundle")
            with self.assertRaisesRegex(ValueError,"development-selected"): export(args)
            self.assertFalse((root/"bundle").exists())


if __name__=="__main__": unittest.main()
