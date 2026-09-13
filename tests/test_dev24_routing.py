import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.dev24_routes import DEFAULT_MANIFEST, resolve_routes
from tools.evaluate_dev24 import REPO, build_plan, main


class RouteRoutingTests(unittest.TestCase):
    def test_shards_cover_the_frozen_subset_without_overlap(self):
        whole = resolve_routes()
        shards = [resolve_routes(shard=i) for i in range(1, 5)]
        ids = [route for shard in shards for route in shard['route_ids']]
        self.assertEqual(len(ids), 24)
        self.assertEqual(len(set(ids)), 24)
        self.assertEqual(set(ids), set(whole['route_ids']))
        self.assertTrue(all(s['route_count'] == 6 for s in shards))

    def test_modified_xml_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = json.loads(DEFAULT_MANIFEST.read_text())
            (tmp/'selection_manifest.json').write_text(json.dumps(manifest))
            source = DEFAULT_MANIFEST.parent / manifest['xml']['file']
            (tmp/source.name).write_bytes(source.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'differs from the frozen'):
                resolve_routes(tmp/'selection_manifest.json')

    def fixture(self, tmp, stale=False):
        tmp = Path(tmp)
        benchmark, carla = tmp/'benchmark', tmp/'carla'
        evaluator = benchmark/'leaderboard/leaderboard/leaderboard_evaluator.py'
        evaluator.parent.mkdir(parents=True)
        evaluator.write_text('# must not run in a dry run\n')
        carla.mkdir()
        (carla/'CarlaUE4.sh').write_text('# must not run in a dry run\n')
        checkpoint = tmp/'weights.pth'
        checkpoint.write_bytes(b'not a model; a dry run must not load it')
        config = tmp/'config.py'
        profile = resolve_routes()
        if stale:
            profile['route_count'] = 94
        config.write_text('closed_loop_eval = '+repr(profile)+'\n')
        args = argparse.Namespace(config=str(config), ckpt=str(checkpoint), gpu=2,
            run_name='route_test', shard=None, seed=0, rpc_port=None, tm_port=None,
            benchmark_root=str(benchmark), carla_root=str(carla), dry_run=True, resume=False)
        return args

    def test_stale_dev94_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Stale closed_loop_eval'):
                build_plan(self.fixture(tmp, stale=True))

    def test_dry_run_uses_this_agent_and_creates_no_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            argv = ['evaluate_dev24.py', '--config', args.config, '--ckpt', args.ckpt,
                    '--gpu', '2', '--run-name', args.run_name, '--benchmark-root', args.benchmark_root,
                    '--carla-root', args.carla_root, '--shard', '3', '--dry-run']
            with patch.object(sys, 'argv', argv), patch('subprocess.call') as launch:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    main()
            launch.assert_not_called()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['route_count'], 6)
            self.assertEqual(plan['shard'], 3)
            self.assertEqual(plan['agent'], str(REPO/'team_code/drivetransformer_b2d_agent.py'))
            self.assertEqual(plan['pythonpath'].split(':')[:2], [str(REPO), str(REPO.parent)])
            self.assertIn('--routes='+plan['routes'], plan['evaluator_arguments'])
            self.assertIn('--agent='+plan['agent'], plan['evaluator_arguments'])
            self.assertFalse(Path(plan['output_dir']).exists())


if __name__ == '__main__':
    unittest.main()
