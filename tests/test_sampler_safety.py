#!/usr/bin/env python3
import pathlib, subprocess, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="ufsm-sampler-safety-") as d:
    subprocess.run([ROOT / "build/make_pipeline_fixture", d], check=True)
    subprocess.run([ROOT / "build/test_sampler_safety", d], check=True, timeout=60)
