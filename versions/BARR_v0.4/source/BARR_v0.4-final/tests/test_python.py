"""Run with python -m unittest discover -s tests -p 'test_python.py' -v."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "scripts"))
from barr_io import load_npz, write_native, Instance, MAX_TOTAL
from run import parser, execute
from analyze import summarize

class IOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def graph(self, **kwargs):
        data = dict(weights=np.array([10., 6., 6.]), edges=np.array([[0, 1], [0, 2]], dtype=np.int64),
                    initial_mask=np.array([1, 0, 0]), agents=np.array([0, 1, 1]), split="TRAIN", source_group="train_source")
        data.update(kwargs); path = self.root / "g.npz"; np.savez(path, **data); return path
    def test_roundtrip(self):
        g = load_npz(self.graph()); self.assertEqual(g.objective([1, 2]), 12.)
        write_native(self.root / "g.barr", g, g.ticks(1e-6)); self.assertTrue((self.root / "g.barr").read_text().startswith("BARR1 3 2"))
    def test_edge_arrays(self):
        path = self.root / "g.npz"; np.savez(path, weights=[1., 2.], edge_u=np.array([0]), edge_v=np.array([1]))
        g = load_npz(path); self.assertEqual(g.objective(g.initial), 2.)
    def test_invalid_weights(self):
        for w in ([1., -1., 3.], [1., float('nan'), 3.]):
            with self.assertRaises(ValueError): load_npz(self.graph(weights=np.array(w)))
    def test_invalid_edges(self):
        for edges in ([[0, 0]], [[0, 3]], [[0, 1], [1, 0]]):
            with self.assertRaises(ValueError): load_npz(self.graph(edges=np.array(edges)))
    def test_float_edge_ids_rejected(self):
        with self.assertRaises(ValueError): load_npz(self.graph(edges=np.array([[0., 1.]])))
    def test_infeasible_initial_rejected(self):
        with self.assertRaises(ValueError): load_npz(self.graph(initial_mask=np.array([1, 1, 0])))
    def test_invalid_memberships(self):
        g = load_npz(self.graph())
        for ids in ([0, 1], [1, 1], [3], [-1], [1.5]):
            with self.assertRaises(ValueError): g.objective(ids)
    def test_overflow_and_zero(self):
        g = load_npz(self.graph())
        with self.assertRaises(ValueError): g.ticks(1e-25)
        with self.assertRaises(ValueError): g.ticks(0)
        self.assertEqual(g.ticks(100), [0, 0, 0])
    def test_native_cli(self):
        gpath = self.graph(); out = self.root / "run"
        result = execute(parser().parse_args(["--graph", str(gpath), "--out", str(out), "--seconds", ".5"]))
        self.assertEqual(result["status"], "COMPLETE", result.get("error")); self.assertEqual(result["returned_quality"], 12.)
        self.assertFalse(result["native"]["collection_mode"])
    def test_missing_native_keeps_na(self):
        result = execute(parser().parse_args(["--graph", str(self.graph()), "--out", str(self.root / "run"), "--binary", str(self.root / "none")]))
        self.assertEqual(result["status"], "ERROR"); self.assertIsNone(result["complete_native_quality"]); self.assertEqual(result["fallback_quality"], 10.)
    def test_zero_budget_rejected(self):
        result = execute(parser().parse_args(["--graph", str(self.graph()), "--out", str(self.root / "run"), "--seconds", "0"]))
        self.assertEqual(result["status"], "ERROR"); self.assertIsNone(result["complete_native_quality"])
    def test_test_collection_blocked(self):
        result = execute(parser().parse_args(["--graph", str(self.graph(split="TEST")), "--out", str(self.root / "run"), "--trace"]))
        self.assertEqual(result["status"], "ERROR"); self.assertIn("collection", result["error"])
    def test_aggregate_missing_not_success_subset(self):
        rows = [dict(variant="barr", allowance_seconds=1, source_group="s", graph_id="g", status="COMPLETE",
                     complete_native_quality=20, caller_observed_seconds=.8),
                dict(variant="barr", allowance_seconds=1, source_group="s", graph_id="g", status="TIMEOUT",
                     complete_native_quality=None, caller_observed_seconds=1.1)]
        s = summarize(rows)[0]; self.assertIsNone(s["complete_quality"]); self.assertEqual(s["complete_cells"], 1)

@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch unavailable")
class NeuralTests(unittest.TestCase):
    def test_native_python_prediction_parity_and_permutation(self):
        import torch
        from barr_model import RankingNet
        torch.set_num_threads(1); torch.manual_seed(918)
        rng = np.random.default_rng(928)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for depth in (0, 1, 2):
                net = RankingNet(hidden=8, layers=depth).double().eval()
                model = td / "m.barrnn"; net.export(model)
                for n in (1, 7, 31):
                    x = rng.normal(size=(n, 10)); x[:, 3] = rng.integers(0, 2, n)
                    e = np.array([[i, j, int(rng.integers(0, 2))] for i in range(n) for j in range(i+1, n) if rng.random() < .25], dtype=int).reshape(-1, 3)
                    c = rng.normal(size=6); action = dict(x=x.tolist(), edges=e.tolist(), context=c.tolist())
                    expected = net.action(action).detach().numpy()
                    feat = td / "f.txt"
                    feat.write_text(f"BARRFEAT1 {n} {len(e)}\n" + "\n".join(" ".join(format(v, ".17g") for v in row) for row in x)
                                    + "\n" + "\n".join(" ".join(str(v) for v in row) for row in e) + "\n"
                                    + " ".join(format(v, ".17g") for v in c) + "\n")
                    output = subprocess.check_output([str(ROOT / "build/barr_solver"), "--model", str(model), "--policy-test", str(feat)], text=True)
                    actual = np.array([float(v) for v in output.split()]); np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
                    perm = rng.permutation(n); inverse = np.argsort(perm)
                    re = e.copy()
                    if len(re): re[:, :2] = inverse[re[:, :2]]
                    again = net.action(dict(x=x[perm].tolist(), edges=re.tolist(), context=c.tolist())).detach().numpy()
                    np.testing.assert_allclose(again, expected, atol=1e-10, rtol=1e-10)
    def test_training_rejects_test_labels(self):
        from train import read_groups
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "x.jsonl"; f.write_text(json.dumps({"split": "TEST"}) + "\n")
            with self.assertRaises(ValueError): read_groups(f, {"TRAIN"})

if __name__ == "__main__": unittest.main()
