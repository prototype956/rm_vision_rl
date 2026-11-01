"""The console must show current role runs without losing legacy experiments."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.training.dashboard.readers import list_runs


class RoleRunTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.training=self.root/'artifacts/training'
        for name in ('legacy','old-group/infantry','new-group/infantry','new-group/sentry'):
            path=self.training/name;path.mkdir(parents=True)
            (path/'run.json').write_text(json.dumps(dict(status='running',completed_updates=108)))

    def test_flat_and_nested_role_runs_have_stable_unique_ids(self):
        rows=list_runs(self.training)
        self.assertEqual({r['run_id'] for r in rows},
            {'legacy','old-group::infantry','new-group::infantry','new-group::sentry'})
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['completed_updates']==108 for r in rows))

    def test_role_details_resolve_without_allowing_traversal(self):
        from tools.training.dashboard.manager import TrainingManager
        from tools.training.dashboard.schemas import RequestError
        manager=TrainingManager(self.root);self.addCleanup(manager.close)
        result=manager.get_run('new-group::infantry')
        self.assertEqual(result['run_id'],'new-group::infantry')
        self.assertEqual(Path(result['run_dir']),self.training/'new-group/infantry')
        self.assertEqual(manager.get_run('legacy')['run_id'],'legacy')
        for value in ('..::infantry','new-group::other','../new-group::infantry','new-group::infantry::sentry'):
            with self.subTest(value=value),self.assertRaises(RequestError):manager.get_run(value)


if __name__=='__main__':unittest.main()
