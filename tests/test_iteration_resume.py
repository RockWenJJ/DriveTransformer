"""Checkpoint iteration and AdamW state must agree with completed updates."""
import logging
import tempfile
import unittest
from pathlib import Path
import torch
from mmcv.runner import IterBasedRunner


class Toy(torch.nn.Linear):
    def train_step(self, data, optimizer):
        return {}


class IterationResume(unittest.TestCase):
    def test_checkpoint_and_resume_do_not_repeat_completed_update(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Toy(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            runner = IterBasedRunner(model=model, optimizer=optimizer,
                work_dir=directory, logger=logging.getLogger('resume-test'),
                max_iters=4, meta={'iter': 123, 'epoch': 456})
            for _ in range(3):
                optimizer.zero_grad()
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step()
            runner._iter = 2  # after third optimizer step, before train() increments
            runner.save_checkpoint(directory)
            path = Path(directory) / 'iter_3.pth'
            saved = torch.load(str(path), map_location='cpu')
            self.assertEqual(saved['meta']['iter'], 3)
            other = Toy(2, 1)
            opt = torch.optim.AdamW(other.parameters(), lr=0.01)
            resumed = IterBasedRunner(model=other, optimizer=opt,
                work_dir=directory, logger=logging.getLogger('resume-test'), max_iters=4)
            resumed.resume(str(path), map_location='cpu')
            self.assertEqual(resumed.iter, 3)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(opt.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
            self.assertTrue(all(int(s['step']) == 3 for s in opt.state.values()))


if __name__ == '__main__':
    unittest.main()
