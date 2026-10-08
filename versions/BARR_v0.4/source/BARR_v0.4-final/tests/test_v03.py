"""Additional v0.3 provenance, diagnostic and refusal tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
from run import parser, execute, executor_contract
from barr_io import sha256
from mechanism_suite import frozen, make_chain

def high_order(count):
    return lambda path: frozen(path, [2*count-1,1]+[2]*count,
        [(0,1)]+[(u,v) for u in (0,1) for v in range(2,count+2)], [0], [1], [0])
def chain(count):
    return lambda path: make_chain(path,count)
def write_kernel(path,fixture):
    fixture(path)

class VersionThreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def probe(self, fixture, *args):
        kernel = self.root / 'k.barrk'; write_kernel(kernel, fixture)
        report = self.root / 'report.json'
        p = subprocess.run([str(ROOT/'build/barr_probe'), '--kernel', str(kernel), '--output', str(report),
                            '--seconds', '3', *args], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(report.read_text())
    def graph(self, split='TRAIN'):
        p = self.root/'g.npz'
        np.savez(p, weights=np.array([10.,6.,6.]), edges=np.array([[0,1],[0,2]], dtype=np.int64),
                 initial_mask=np.array([1,0,0]), split=split, source_group='source_'+split)
        return p
    def test_snapshot_test_guard(self):
        a=parser().parse_args(['--graph',str(self.graph('TEST')),'--out',str(self.root/'out'),'--snapshots'])
        r=execute(a);self.assertEqual(r['status'],'ERROR');self.assertIn('snapshots',r['error'])
    def test_executor_contract_changes_with_backend_and_quantization(self):
        a=parser().parse_args(['--graph','g','--out','o']); c=executor_contract(a)
        a.recovery_backend='branch'; self.assertNotEqual(c,executor_contract(a))
        a.recovery_backend='hybrid'; a.tick=.001; self.assertNotEqual(c,executor_contract(a))
    def test_old_model_contract_refused_before_native_launch(self):
        model=self.root/'old.barrnn';model.write_text('unused old model bytes')
        Path(str(model)+'.json').write_text(json.dumps({'model_sha256':sha256(model),'executor_contract':{'algorithm':'BARR-0.2'}}))
        a=parser().parse_args(['--graph',str(self.graph()),'--out',str(self.root/'out'),'--model',str(model),'--rank','gnn'])
        r=execute(a);self.assertEqual(r['status'],'ERROR');self.assertIn('workpoint',r['error']);self.assertIsNone(r['command'])
    def test_probe_high_order_not_pairs(self):
        r=self.probe(high_order(4));m=r['mechanism']
        self.assertTrue(r['exact']);self.assertEqual(m['min_positive_coalition_size'],4)
        self.assertEqual(m['best_size_at_most_two_gain_ticks'],0)
        self.assertEqual(m['exact_gain_ticks'],1)
    def test_large_connected_kernel_not_truncated(self):
        r=self.probe(chain(64));self.assertEqual(r['outsiders'],64);self.assertTrue(r['exact'])
        self.assertEqual(r['factor']['induced_width'],1);self.assertEqual(r['lower_ticks']-r['base_ticks'],62)
        self.assertIsNone(r['mechanism']['min_positive_coalition_size'])
    def test_refusal_not_false_optimality(self):
        r=self.probe(high_order(4),'--boundary','1');self.assertFalse(r['exact']);self.assertIsNone(r['mechanism'])
        self.assertEqual(r['factor']['status'],'boundary_arity_limit')
        self.assertGreater(r['upper_ticks'],r['lower_ticks'])
    def test_backbone_substitution_without_outsider_conflict(self):
        fixture=lambda path: frozen(path,[10,10,6,6],[(0,1),(0,2),(1,3)],[0],[1],[0])
        r=self.probe(fixture);m=r['mechanism']
        self.assertEqual(m['unary_forced_gain_ticks'],[6,6])
        self.assertEqual(m['exact_gain_ticks'],6)
        self.assertEqual(m['additive_selection_actual_safe_gain_ticks'],2)
    def test_wrong_exact_baseline_refused(self):
        kernel=self.root/'k.barrk';write_kernel(kernel, high_order(3))
        lines=kernel.read_text().splitlines();fields=lines[1].split();fields[4]='0';lines[1]=' '.join(fields)
        kernel.write_text('\n'.join(lines)+'\n')
        p=subprocess.run([str(ROOT/'build/barr_probe'),'--kernel',str(kernel),'--output',str(self.root/'r.json')],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('exact backbone optimum',p.stderr)

if __name__=='__main__':unittest.main()
