#!/usr/bin/env python3
"""Small real CUDA integration test: warm-start, split, resume, floating export.

Uses P=32 and widths=8,8; unlike production-size tests it needs little VRAM.
Run explicitly with an available GPU; it is not part of the CPU test target.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from production import header
from sheet_geometry import make_record, reference_value, write_dataset
from sheet_pipeline import decompress_zstd


def main():
    env={k:v for k,v in os.environ.items() if not k.startswith("UFSM_")}
    with tempfile.TemporaryDirectory(prefix="ufsm-sheet-cli-") as tmp:
        t=Path(tmp)
        subprocess.run([ROOT/"build/make_pipeline_fixture",t],check=True)
        source=t/"sources.json"
        source.write_text(json.dumps({"sources":[dict(name="fixture",root=str(t),ct="ct",um=1,
            targets={"recto":"labels"},holdout=[96,96,96,32,32,32])]}))
        reference=dict(version=1,units="turns",coordinate_order="zyx",
            knots=[[0,-40,-40,10,-2],[128,-40,-40,10,-2]],input_center=5,input_scale=10)
        rows=[]
        for z in (6,15.5,22,38,47.5,54,70,79.5,86):
            xyz=np.array([[z,6,8],[z,6,24]])
            q=reference_value(reference,xyz)+.2
            rows.extend([make_record("coordinate",[xyz[0]],[q[0]]),
                make_record("ordering",xyz,q,q[1]-q[0]),
                make_record("path",[[z,y,8] for y in range(4,12)]),
                make_record("gap",[[z,6,16]])])
        write_dataset(t/"geometry",rows,reference,{})
        def cover(name,tiles):
            p=t/(name+".json")
            p.write_text(json.dumps(dict(version=1,P=32,level=0,count=len(tiles),
                source_names=["fixture"],tiles=[[0,z,0,0] for z in tiles])))
            return p
        donor_cover=cover("donor-cover",[0,32]); experiment_cover=cover("cover",[0,32,64])
        base=[ROOT/"build/ufsm","train",source,"--P","32","--B","1","--widths","8,8",
            "--fp4","0","--mem","auto16","--steps","3","--warmup","10","--seed","2",
            "--workers","1","--val-batches","1","--levels","1","--log-every","1",
            "--ckpt-every","1","--val-every","2","--noaug","1","--soft","2",
            "--down-norm","1","--gn-stats","stored"]
        def execute(command):
            result=subprocess.run(list(map(str,command)),env=env,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,timeout=120)
            assert result.returncode==0,result.stdout
            return result.stdout
        gpu=os.environ.get("UFSM_TEST_GPU","0")
        execute(base+["--gpus",gpu,"--cover",donor_cover,"--out",t/"donor"])
        for mode,gpus in (("single",gpu),("split","0,1")):
            if mode=="split" and os.environ.get("UFSM_TEST_SINGLE_ONLY"): continue
            command=base+["--gpus",gpus,"--cover",experiment_cover,"--geometry",t/"geometry/geometry.json"]
            if mode=="split": command += ["--split","z"]
            full=t/(mode+"-full"); partial=t/(mode+"-partial"); resumed=t/(mode+"-resumed")
            init=["--resume",t/"donor/last.ckpt","--sheet-init","1","--sheet-variant","1"]
            execute(command+["--out",full,*init])
            execute(command+["--out",partial,*init,"--limit-seconds",".000001"])
            hp=header(partial/"last.ckpt"); hf=header(full/"last.ckpt")
            assert hp["step"]==3 and hp["extra"]["cover"]["cursor"]==1
            assert hf["step"]==5 and hf["extra"]["cover"]["cursor"]==3
            execute(command+["--out",resumed,"--resume",partial/"last.ckpt"])
            hr=header(resumed/"last.ckpt")
            assert hr["extra"]["sheet"]==hf["extra"]["sheet"]
            assert hr["extra"]["sheet"]["variant"]==1 and hr["extra"]["sheet"]["schedule_start"]==2
            assert hr["extra"]["cover"]==hf["extra"]["cover"]
            with (full/"geometry.csv").open() as f: full_rows=list(csv.DictReader(f))
            with (resumed/"geometry.csv").open() as f: resumed_rows=list(csv.DictReader(f))
            assert [r["ramp"] for r in full_rows[1:]]==[r["ramp"] for r in resumed_rows]
            assert all(np.isfinite(float(r["weighted"])) for r in full_rows)
            assert float(full_rows[0]["coordinate"])>.1
            def weights(path):
                with path.open("rb") as f: f.readline(); return f.read(hf["nparams"]*4)
            assert weights(full/"last.ckpt")!=weights(partial/"last.ckpt")
            parameters=np.frombuffer(weights(full/"last.ckpt"),"<f4")
            assert np.any(parameters[-10:-2]!=0),"winding-head weights never learned"
            bad=subprocess.run(list(map(str,command+["--out",t/"bad","--resume",partial/"last.ckpt",
                "--sheet-variant","2"])),env=env,capture_output=True,text=True)
            assert bad.returncode!=0 and "preserve its loss variant" in bad.stderr
            pred=t/(mode+"-prediction")
            execute([ROOT/"build/ufsm","predict",full/"last.ckpt",t,"ct",pred,"--um","1",
                "--gpu",gpu,"--reference",t/"geometry/reference.json","--box","0,0,0,16,16,16",
                "--window","32","--halo","4","--shard","128","--levels","1","--ema","0",
                "--fp4","0","--threads","1"])
            manifest=json.loads((pred/"winding/manifest.json").read_text())
            assert manifest["shape"]==[16,16,16] and manifest["units"]=="turns"
            shards=list((pred/"winding").glob("*.q.zst")); assert len(shards)==1
            q=np.frombuffer(decompress_zstd(shards[0],128**3*4),"<f4").reshape(128,128,128)
            assert np.isfinite(q[:16,:16,:16]).any() and np.isnan(q[16:]).all()
            print(f"{mode}: donor cover detached, three updates, resume contract, floating prediction: ok")
            if mode=="single":
                complete=t/"all-losses"
                execute(command+["--out",complete,"--resume",t/"donor/last.ckpt","--sheet-init","1","--sheet-variant","2"])
                with (complete/"geometry.csv").open() as f: losses=list(csv.DictReader(f))
                assert float(losses[0]["path"])>0 and float(losses[0]["gap"])>0


if __name__=="__main__": main()
