#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from sheet_geometry import (RECORD, INDEX, fit_reference, inside_boxes, make_record,
                            reference_value, sparse_loss, unwrap_mesh, write_dataset, align_components)
from sheet_reconstruct import canonical, flow, lipschitz_bound, validate_mesh
import torch


class GeometryTests(unittest.TestCase):
    def test_spiral_seam_and_holes(self):
        q,z=np.meshgrid(np.linspace(.4,3.4,101),np.linspace(100,140,4))
        r=100+30*q
        xyz=np.stack([z,500+r*np.sin(2*np.pi*q),500+r*np.cos(2*np.pi*q)],axis=-1)
        valid=np.ones(q.shape,bool); valid[1,40:45]=False
        lifted,comp,edges,bad=unwrap_mesh(xyz,valid,np.array([[0,500,500],[200,500,500]]))
        self.assertFalse(bad)
        np.testing.assert_allclose((lifted-q)[valid],0,atol=1e-10)
        self.assertGreater(len(edges),0)
        self.assertTrue(np.isnan(lifted[~valid]).all())

    def test_losses_numerical_gradient(self):
        rows=[make_record("coordinate",[[3,4,5]],[2]),
              make_record("continuity",[[3,4,5],[4,4,5]],target=.03),
              make_record("ordering",[[3,4,5],[3,4,9]],target=1),
              make_record("path",[[i,4,5] for i in range(4)]),
              make_record("gap",[[3,4,7]])]
        rows=np.array(rows,RECORD)
        values=np.random.default_rng(2).normal(size=(5,32,2))
        loss,gradient,_=sparse_loss(values,rows)
        self.assertTrue(np.isfinite(loss))
        for i,row in enumerate(rows):
            for p in range(int(row["count"])):
                for channel in range(2):
                    changed=values.copy(); changed[i,p,channel]+=1e-5
                    high=sparse_loss(changed,rows)[0]
                    changed[i,p,channel]-=2e-5
                    low=sparse_loss(changed,rows)[0]
                    self.assertAlmostEqual(gradient[i,p,channel],(high-low)/2e-5,places=7)

    def test_reference_and_contract(self):
        xyz=np.array([[z,500,500+r] for z in (0,100) for r in range(100,1100,10)],float)
        q=(xyz[:,2]-500)/30-2
        reference=fit_reference(xyz,q,np.array([[0,500,500],[100,500,500]]))
        np.testing.assert_allclose(reference_value(reference,xyz),q,atol=1e-8)
        with tempfile.TemporaryDirectory() as root:
            manifest=write_dataset(Path(root)/"g",[make_record("coordinate",[xyz[0]],[q[0]])],reference,{})
            self.assertEqual(RECORD.itemsize,528); self.assertEqual(INDEX.itemsize,24)
            self.assertEqual(manifest["records"],1)
            self.assertEqual(json.loads((Path(root)/"g/geometry.json").read_text())["task"],"surface_winding")

    def test_folded_local_reference_uses_positive_global_pitch(self):
        r=np.linspace(100,1000,500)
        xyz=np.concatenate([np.column_stack([np.full(500,3000),np.full(500,500),500+r]),
                            np.column_stack([np.zeros(100),np.full(100,500),500+r[:100]])])
        q=np.r_[r/30,20-r[:100]/100]
        reference=fit_reference(xyz,q,np.array([[0,500,500],[3000,500,500]]),z_step=512)
        self.assertTrue(reference["local_fit_fallbacks"])
        self.assertTrue(all(k[3]>0 for k in reference["knots"]))
        np.testing.assert_array_equal(q,np.r_[r/30,20-r[:100]/100])

    def test_holdout_guard(self):
        xyz=np.array([[0,0,0],[10,10,10],[20,20,20]])
        self.assertEqual(inside_boxes(xyz,[[10,10,10,4,4,4]],2).tolist(),[False,True,False])

    def test_inconsistent_cycle_is_excluded(self):
        xyz=np.array([[[50,400,400],[50,400,600]],[[50,600,400],[50,600,600]]],float)
        q,components,edges,bad=unwrap_mesh(xyz,np.ones((2,2),bool),np.array([[0,500,500],[100,500,500]]))
        self.assertEqual(bad,[0]); self.assertTrue(np.isnan(q).all())

    def test_nearest_triangle_gauge_and_ambiguous_first_hit(self):
        xyz=np.array([[z,y,x] for y in (0,5,10) for z,x in ((0,0),(0,10),(10,0),(10,10))],float)
        components=np.repeat(np.arange(3),4)
        q=np.repeat([0.,100.,200.],4)
        faces=np.array([[k,k+1,k+2] for k in (0,4,8)]+[[k+1,k+3,k+2] for k in (0,4,8)])
        normals=np.tile([0.,1.,0.],(12,1))
        aligned,reliable,pairs,rejected=align_components(xyz,q,components,normals,20,faces)
        self.assertTrue(reliable.all()); self.assertFalse(rejected)
        np.testing.assert_allclose(aligned,np.repeat([0,1,2],4))
        self.assertEqual(len(pairs),8)
        normals=np.tile([0.,1.,0.],(12,1)); normals[4:8]*=-1
        aligned,reliable,pairs,rejected=align_components(xyz,q,components,normals,20,faces)
        self.assertFalse(reliable[4:].any())
        self.assertEqual(len(pairs),0,"an ambiguous first layer was skipped")

    def test_injective_flow_bound(self):
        field=torch.zeros(1,3,4,4,4)
        field[:,0]=torch.arange(4).reshape(4,1,1)/2
        bound=lipschitz_bound(field,[0,0,0],[3,3,3])
        self.assertAlmostEqual(bound,.5)
        points=torch.tensor([[.5,1,1],[1.,1,1]])
        output=flow(points,field,[0,0,0],[3,3,3],8)
        self.assertGreater(float(output[1,0]-output[0,0]),.5)

    def test_mesh_validation(self):
        vertices=np.array([[0,0,0],[0,1,0],[0,0,1],[0,1,1]],float)
        report=validate_mesh(vertices,[[0,1,2],[1,3,2]])
        self.assertEqual(report["self_intersections"],0)
        with self.assertRaisesRegex(ValueError,"self-intersection"):
            validate_mesh(np.array([[0,0,0],[0,2,0],[0,0,2],[-1,.5,.5],[1,.5,.5],[0,2,2]]),[[0,1,2],[3,4,5]])
        with self.assertRaisesRegex(ValueError,"nonmanifold"):
            validate_mesh(np.array([[0,0,0],[0,1,0],[0,0,1],[1,0,1],[2,0,1]]),[[0,1,2],[0,1,3],[0,1,4]])
        with self.assertRaisesRegex(ValueError,"nonmanifold vertex"):
            validate_mesh(np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],float),[[0,1,2],[0,3,4]])
        with self.assertRaisesRegex(ValueError,"self-intersection across a shared edge"):
            validate_mesh(np.array([[0,0,0],[1,0,0],[0,1,0],[.5,.5,0]],float),[[0,1,2],[0,1,3]])


if __name__=="__main__": unittest.main()
