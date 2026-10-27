"""Model browser approval, cache and media boundary checks without real jobs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from robot_control import model_library as lib
from robot_control import studio

class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.patch=patch.object(lib,'BASE',self.base);self.patch.start()
        path=self.base/'models/policy.pt';path.parent.mkdir();path.write_bytes(b'fixed-policy')
        self.model=dict(id='mojia-test',robot='mojia',iteration=1,file='models/policy.pt',sha256=lib.digest(path),evaluations={},approval='pending')
        lib.save(dict(models=[self.model]))
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_untested_cannot_be_approved(self):
        with self.assertRaises(ValueError):lib.approve('mojia-test',True)
        self.assertEqual(lib.get_model('mojia-test')['approval'],'pending')
    def test_explicit_approval_preserves_immutable_weights(self):
        self.model['evaluations']={'cuda:0':{'summary':dict(full=True,completion_rate=.9,normalized_return=.8,median_root_rmse_m=.1)}}
        lib.save(dict(models=[self.model]));lib.approve('mojia-test',True,'观察良好')
        dest=self.base/'approved/mojia/mojia-test/policy.pt'
        self.assertEqual(dest.read_bytes(),b'fixed-policy');self.assertEqual(dest.stat().st_ino,lib.policy_path(self.model).stat().st_ino)
        lib.approve('mojia-test',False);self.assertFalse(dest.exists());self.assertTrue(lib.policy_path(self.model).exists())
    def test_modified_snapshot_rejected(self):
        self.model['evaluations']={'cuda:0':{'summary':dict(full=True,completion_rate=1,normalized_return=1,median_root_rmse_m=0)}}
        lib.save(dict(models=[self.model]));lib.policy_path(self.model).write_bytes(b'changed')
        with self.assertRaises(ValueError):lib.approve('mojia-test',True)
    def test_paths_cannot_escape_library(self):
        with self.assertRaises(ValueError):lib.policy_path(dict(file='../outside.pt'))
    def test_job_deduplicates_identical_snapshot_request(self):
        with patch.object(studio,'JOBS',self.base/'jobs'),patch.object(studio.GPU_POOL,'submit') as submit,patch.object(lib,'catalog',return_value={'motions':{'mojia':[dict(id='dance1_subject2',seconds=42)]}}):
            b=dict(model_id='mojia-test',kind='render',seconds=0)
            a=studio.create(b);c=studio.create(b)
            self.assertEqual(a['id'],c['id']);self.assertEqual(a['seconds'],42);self.assertEqual(submit.call_count,1)
    def test_cpu_evaluation_has_independent_queue(self):
        with patch.object(studio,'JOBS',self.base/'jobs'),patch.object(studio.POOL,'submit') as cpu,patch.object(studio.GPU_POOL,'submit') as gpu,patch.object(lib,'catalog',return_value={'motions':{'mojia':[]}}):
            studio.create(dict(model_id='mojia-test',kind='evaluate',device='cpu'))
            self.assertEqual(cpu.call_count,1);self.assertEqual(gpu.call_count,0)
    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):studio.create(dict(model_id='mojia-test',device='shell'))

if __name__=='__main__':unittest.main()
