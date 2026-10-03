#!/usr/bin/env python3
"""Python/native file contract, task binding and sparse derivative checks."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from sheet_geometry import make_record, write_dataset


class NativeTests(unittest.TestCase):
    def test_dataset(self):
        reference=dict(version=1,units="turns",coordinate_order="zyx",knots=[[0,40,40,10,-2],[100,40,40,10,-2]],input_center=0,input_scale=10)
        records=[make_record("coordinate",[[5,6,7]],[3]),
                 make_record("ordering",[[5,6,7],[5,6,15]],target=1),
                 make_record("continuity",[[5,6,7],[6,6,7]],target=.02),
                 make_record("path",[[i,6,7] for i in range(4,12)]),
                 make_record("gap",[[5,6,11]])]
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/"geometry"
            write_dataset(out,records,reference,{},contacts=[[5,6,7,2]])
            subprocess.run([ROOT/"build/test_sheet",out/"geometry.json"],check=True)
            # Altered teachers must fail before a GPU is initialized.
            with (out/"records.bin").open("ab") as f: f.write(b"changed")
            result=subprocess.run([ROOT/"build/test_sheet",out/"geometry.json"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode,0)


if __name__=="__main__": unittest.main()
