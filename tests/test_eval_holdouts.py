"""Exercise prediction reuse and failed reruns without using a GPU."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/eval_holdouts.py"
FAKE = '''#!/usr/bin/env python3
import json, os, pathlib, sys
with open(os.environ["UFSM_FAKE_CALLS"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "predict":
    out = pathlib.Path(sys.argv[5]); out.mkdir(parents=True, exist_ok=True)
    (out / "zarr.json").write_text("{}")
    (out / "prediction.txt").write_text(pathlib.Path(sys.argv[2]).read_text())
    if os.environ.get("UFSM_FAKE_FAIL"):
        raise SystemExit(1)
else:
    for t in sys.argv[sys.argv.index("--thr") + 1].split(","):
        print(t, .4, .5, .4444, .4, .6, .7)
'''


class HoldoutReuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.binary = self.root / "ufsm"
        self.binary.write_text(FAKE)
        self.binary.chmod(0o755)
        self.ckpt = self.root / "last.ckpt"
        self.ckpt.write_text("first")
        self.calls = self.root / "calls.jsonl"
        self.cfg = self.root / "sources.json"
        self.source = dict(name="heldout", root="ct-root", ct="scan", um=2.4,
                           holdout=[0, 0, 0, 32, 32, 32],
                           targets=dict(recto=dict(root="labels", group=".")))
        self.write_config()
        self.out = self.root / "predictions"
        self.score = self.root / "scores.json"
        self.env = dict(os.environ, UFSM_BIN=str(self.binary), UFSM_FAKE_CALLS=str(self.calls))

    def write_config(self):
        self.cfg.write_text(json.dumps(dict(sources=[self.source])))

    def run_eval(self, extra=(), **env):
        r = subprocess.run([sys.executable, str(SCRIPT), str(self.ckpt), "--sources", str(self.cfg),
                            "--out", str(self.out), "--scores", str(self.score), *extra],
                           env=dict(self.env, **env), text=True, capture_output=True)
        return r

    def predicted(self):
        return [c for c in map(json.loads, self.calls.read_text().splitlines()) if c[0] == "predict"]

    def assert_ok(self, r):
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_same_prediction_is_reused(self):
        self.assert_ok(self.run_eval())
        self.assert_ok(self.run_eval())
        self.assertEqual(len(self.predicted()), 1)

    def test_source_level_is_explicit_and_respects_available_labels(self):
        self.source['min_level'] = 2
        self.write_config()
        self.assert_ok(self.run_eval(['--level','1','--source-levels','{"heldout":0}']))
        cmd = self.predicted()[0]
        self.assertEqual(cmd[cmd.index('--level')+1], '2')
        self.assertIn('heldout.level2', cmd[4])
        scores = json.loads(self.score.read_text())['scores']['heldout']
        self.assertEqual(scores['level'], 2)

    def test_unknown_or_invalid_source_level_is_rejected_before_prediction(self):
        for levels in ['{"missing":0}', '{"heldout":-1}', '{"heldout":1.5}', '{"heldout":true}']:
            r = self.run_eval(['--source-levels',levels])
            self.assertNotEqual(r.returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_overwritten_checkpoint_invalidates_prediction(self):
        self.assert_ok(self.run_eval())
        self.ckpt.write_text("other")
        self.assert_ok(self.run_eval())
        self.assertEqual(len(self.predicted()), 2)
        self.assertEqual((self.out / "heldout/prediction.txt").read_text(), "other")

    def test_flags_binary_and_precision_manifest_invalidate_prediction(self):
        self.assert_ok(self.run_eval())
        override = "--window 544 --halo 16 --ema 0"
        self.assert_ok(self.run_eval(UFSM_PREDICT_ARGS=override))
        cmd = self.predicted()[-1]
        self.assertEqual(cmd.count("--window"), 1)
        self.assertEqual(cmd[cmd.index("--window") + 1], "544")
        self.binary.write_text(FAKE + "\n")
        self.assert_ok(self.run_eval(UFSM_PREDICT_ARGS=override))
        (self.root / "precision.txt").write_text("act_mx4 1\n")
        self.assert_ok(self.run_eval(UFSM_PREDICT_ARGS=override))
        self.assertEqual(len(self.predicted()), 4)

    def test_failed_replacement_keeps_previous_prediction_and_removes_old_scores(self):
        self.assert_ok(self.run_eval())
        self.ckpt.write_text("other")
        self.assertNotEqual(self.run_eval(UFSM_FAKE_FAIL="1").returncode, 0)
        self.assertFalse(self.score.exists())
        self.assertEqual((self.out / "heldout/prediction.txt").read_text(), "first")
        self.assertFalse(list(self.out.glob(".heldout-*")))
        self.assert_ok(self.run_eval())
        self.assertEqual((self.out / "heldout/prediction.txt").read_text(), "other")

    def test_legacy_prediction_is_regenerated_unless_score_only(self):
        p = self.out / "heldout"
        p.mkdir(parents=True)
        (p / "zarr.json").write_text("{}")
        self.assert_ok(self.run_eval(["--score-only"]))
        self.assertEqual(len(self.predicted()), 0)
        self.assert_ok(self.run_eval())
        self.assertEqual(len(self.predicted()), 1)

    def test_minimum_label_level_is_obeyed(self):
        self.source["min_level"] = 2
        self.write_config()
        self.assert_ok(self.run_eval(["--level", "1"]))
        self.assertTrue((self.out / "heldout.level2/zarr.json").is_file())
        score = json.loads(self.score.read_text())["scores"]["heldout"]
        self.assertEqual(score["level"], 2)


if __name__ == "__main__":
    unittest.main()
