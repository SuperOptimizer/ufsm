"""Device-list leases must exclude single-GPU work and release partial acquisitions."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import production


class DeviceLeases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ufsm-gpu-leases-')
        self.root = production.ROOT
        production.ROOT = Path(self.tmp.name)

    def tearDown(self):
        production.ROOT = self.root
        self.tmp.cleanup()

    def competing(self, selection):
        script = '''
import pathlib,sys
sys.path.insert(0,sys.argv[1])
import production
production.ROOT=pathlib.Path(sys.argv[2])
try:
    with production.gpu_lock(sys.argv[3]): pass
except RuntimeError:
    sys.exit(3)
'''
        return subprocess.run([sys.executable, '-c', script, str(self.root/'tools'), self.tmp.name, selection],
                              capture_output=True, text=True, timeout=10).returncode

    def test_dual_lease_excludes_each_device_and_canonical_aliases(self):
        with production.gpu_lock('1,00'):
            self.assertEqual(self.competing('0'), 3)
            self.assertEqual(self.competing('1'), 3)
            self.assertEqual(self.competing('1,0'), 3)
            self.assertEqual(self.competing('2'), 0)
            self.assertEqual({p.name for p in (production.ROOT/'runs/.gpu-locks').iterdir()},
                             {'gpu-0.lock', 'gpu-1.lock', 'gpu-2.lock'})
        self.assertEqual(self.competing('0,1'), 0)

    def test_failed_second_acquisition_releases_first(self):
        with production.gpu_lock('1'):
            with self.assertRaises(RuntimeError):
                with production.gpu_lock('0,1'): self.fail('acquired occupied GPU')
            self.assertEqual(self.competing('0'), 0)
            self.assertEqual(self.competing('1'), 3)

    def test_invalid_lists(self):
        for selection in ['', '0,0', '0,00', '-1', '0,', 'x', '0/1', ','.join(map(str, range(9)))]:
            with self.assertRaises(ValueError): production.gpu_devices(selection)
        self.assertEqual(production.gpu_devices(' 1, 00 '), [1, 0])


if __name__ == '__main__': unittest.main()
