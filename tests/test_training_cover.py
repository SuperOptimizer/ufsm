"""CPU-only complete-cover, holdout, deterministic order and metadata fixtures."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_cover', ROOT / 'tools/build_training_cover.py')
cover = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cover)


def group(um, path):
    return {'attributes': {'ome': {'multiscales': [{'datasets': [
        {'path': path, 'coordinateTransformations': [{'type': 'scale', 'scale': [um] * 3}]}]}]}}}


def exhaustive_points(box):
    return itertools.product(*(range(box[d], box[d + 3]) for d in range(3)))


class TrainingCover(unittest.TestCase):
    def test_six_slabs_are_exact_disjoint_complement(self):
        domain, hole = (0, 0, 0, 16, 16, 16), (4, 4, 4, 10, 10, 10)
        slabs = cover.subtract_box(domain, hole)
        self.assertEqual(len(slabs), 6)
        points = [set(exhaustive_points(s)) for s in slabs]
        union = set().union(*points)
        self.assertEqual(sum(map(len, points)), len(union))
        self.assertEqual(union, set(exhaustive_points(domain)) - set(exhaustive_points(hole)))

    def test_small_cube_cover_and_tail_preserve_every_occupied_voxel(self):
        occ = cover.Occupancy(bytes([255] * (18 ** 3)), (18,) * 3, (1,) * 3, (18,) * 3)
        hole = (5, 5, 5, 11, 11, 11)
        plan = cover.make_plan(occ, 4, 2, 'fixture', [hole], {})
        seen = set()
        for si, z, y, x in plan['tiles']:
            tile = (z, y, x, z + 4, y + 4, x + 4)
            self.assertFalse(cover.intersects(tile, hole))
            seen.update(exhaustive_points(tile))
        self.assertEqual(seen, set(exhaustive_points(occ.bounds)) - set(exhaustive_points(hole)))
        self.assertEqual(cover.axis_origins(0, 11, 4), [0, 4, 7])
        self.assertTrue(any(7 in row[1:] or 14 in row[1:] for row in plan['tiles']))

    def test_positive_coarse_cells_and_partial_native_end_are_conservative(self):
        raw = bytearray(4 ** 3)
        raw[0] = raw[-1] = 255
        occ = cover.Occupancy(raw, (4,) * 3, (3,) * 3, (11,) * 3)
        plan = cover.make_plan(occ, 4, 2, 'fixture', [], {})
        self.assertEqual(plan['bounds'], [0, 0, 0, 11, 11, 11])
        seen = set()
        for _, z, y, x in plan['tiles']:
            seen.update(exhaustive_points((z, y, x, z + 4, y + 4, x + 4)))
        required = set(exhaustive_points((0, 0, 0, 3, 3, 3))) | set(exhaustive_points((9, 9, 9, 11, 11, 11)))
        self.assertTrue(required <= seen)

    def test_seeded_order_and_membership_verification(self):
        occ = cover.Occupancy(bytes([1] * (12 ** 3)), (12,) * 3, (1,) * 3, (12,) * 3)
        a = cover.make_plan(occ, 4, 2, 'fixture', [], {})
        b = cover.make_plan(occ, 4, 2, 'fixture', [], {})
        c = cover.make_plan(occ, 4, 3, 'fixture', [], {})
        self.assertEqual(a, b)
        self.assertNotEqual(a['tiles'], c['tiles'])
        self.assertEqual(set(map(tuple, a['tiles'])), set(map(tuple, c['tiles'])))
        changed = copy.deepcopy(a); changed['tiles'].pop(); changed['count'] -= 1
        with self.assertRaisesRegex(ValueError, 'complete cover'):
            cover.verify_plan(changed, occ, [])
        changed = copy.deepcopy(a); changed['tiles'][0] = changed['tiles'][1]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            cover.verify_plan(changed, occ, [])

    def test_missing_and_truncated_occupancy_fail(self):
        with self.assertRaisesRegex(ValueError, 'no positive'):
            cover.Occupancy(bytes(8), (2,) * 3, (2,) * 3, (4,) * 3)
        with self.assertRaisesRegex(ValueError, 'byte count'):
            cover.Occupancy(bytes(7), (2,) * 3, (2,) * 3, (4,) * 3)
        occ = cover.Occupancy(bytes([255] * (8 ** 3)), (8,) * 3, (1,) * 3, (8,) * 3)
        with self.assertRaisesRegex(ValueError, 'no occupied training'):
            cover.make_plan(occ, 4, 2, 'fixture', [(0, 0, 0, 8, 8, 8)], {})
        with self.assertRaisesRegex(ValueError, 'cannot fit a safe'):
            cover.make_plan(occ, 4, 2, 'fixture', [(1, 1, 1, 5, 5, 5)], {})

    def test_metadata_bound_raw_fixture_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for folder in ('ct', 'ct/0', 'labels', 'labels/coarse'):
                (path / folder).mkdir(parents=True, exist_ok=True)
            (path / 'ct/zarr.json').write_text(json.dumps(group(2.4, '0')))
            (path / 'ct/0/zarr.json').write_text(json.dumps({'shape': [16] * 3}))
            (path / 'labels/zarr.json').write_text(json.dumps(group(4.8, 'coarse')))
            (path / 'labels/coarse/zarr.json').write_text(json.dumps({
                'shape': [8] * 3, 'attributes': {'ufsm': {'content': 'sampling occupancy'}}}))
            provenance = {'ct': 'ct', 'native_um': 2.4, 'native_shape_zyx': [16] * 3}
            provenance_path = path / 'labels/provenance.json'
            provenance_path.write_text(json.dumps(provenance))
            source = {'name': 'fixture', 'root': str(path), 'ct': 'ct', 'um': 2.4,
                      'targets': {'recto': {'root': str(path / 'labels'), 'group': '.', 'encoding': 'binary'}},
                      'holdout': [4, 4, 4, 6, 6, 6]}
            cfg = path / 'sources.json'; cfg.write_text(json.dumps({'sources': [source]}))
            raw = path / 'occupancy.raw'; raw.write_bytes(bytes([255] * (8 ** 3)))
            out = path / 'cover.json'
            cover.main(['--sources', str(cfg), '--out', str(out), '--P', '4', '--seed', '2',
                        '--occupancy-raw', str(raw), '--binary', '/nonexistent/never-called'])
            plan = json.loads(out.read_text())
            self.assertEqual(plan['binding']['occupancy_sha256'], cover.digest(raw.read_bytes()))
            self.assertEqual(plan['binding']['label_provenance_sha256'], cover.digest(provenance_path.read_bytes()))
            self.assertEqual(plan['binding']['native_shape_zyx'], [16] * 3)
            self.assertEqual(plan['count'], len(plan['tiles']))
            original = out.read_bytes()
            cover.main(['--sources', str(cfg), '--out', str(out), '--P', '4', '--seed', '2',
                        '--occupancy-raw', str(raw)])
            self.assertEqual(original, out.read_bytes())
            (path / 'ct/zarr.json').write_text(json.dumps(group(1, '0')))
            relative = cover.build(cfg, 4, 2, raw)
            self.assertEqual(relative['tiles'], plan['tiles'])
            coarse_path = path / 'labels/coarse/zarr.json'
            coarse_original = coarse_path.read_text()
            coarse_path.write_text(json.dumps({'shape': [8] * 3}))
            with self.assertRaisesRegex(ValueError, 'no sampling occupancy'):
                cover.build(cfg, 4, 2, raw)
            coarse_path.write_text(coarse_original)
            provenance['native_shape_zyx'][0] = 18
            provenance_path.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'another CT'):
                cover.build(cfg, 4, 2, raw)


if __name__ == '__main__':
    unittest.main()
