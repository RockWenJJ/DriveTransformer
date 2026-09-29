"""DT-Qwen08-LearnToken: one soft input token; native Qwen weights stay frozen."""
from pathlib import Path
import sys
import torch
from torch import nn
from mmcv.models.builder import BACKBONES, HEADS, DETECTORS
from .qwen_frozen_sem import FrozenQwenBackbone, QwenFrozenSemHead, QwenFrozenSemDriveTransformer


@BACKBONES.register_module()
class LearnTokenQwenBackbone(FrozenQwenBackbone):
    def __init__(self, *args, token_seed=0, backward_strategy='recompute', batch_consistency='native', **kwargs):
        super().__init__(*args, **kwargs)
        assert backward_strategy in ('recompute', 'retain')
        self.settings['backward_strategy'] = backward_strategy
        assert batch_consistency in ('native', 'single_sample_kernels')
        self.settings['batch_consistency'] = batch_consistency
        # Private RNG leaves the pre-existing DT/adaptor initialization unchanged.
        generator = torch.Generator(device='cpu').manual_seed(token_seed)
        self.q_lang = nn.Parameter(torch.empty(1, 1, 1024).normal_(std=0.02, generator=generator))

    def forward(self, images):
        if self._client is None:
            root = Path(__file__).resolve().parents[4] / 'tools' / 'qwen_frozen'
            sys.path.insert(0, str(root))
            from token_bridge import LearnTokenClient
            self._client = LearnTokenClient(root, images.device, **self.settings)
        visual, language = self._client(images, self.q_lang)
        return self.visual_projection(visual), language


@HEADS.register_module()
class QwenLearnTokenHead(QwenFrozenSemHead):
    def set_language_context(self, hidden):
        if self._language_context is not None:
            raise RuntimeError('Unconsumed language context; refusing stale-frame fusion')
        if hidden.ndim != 3 or hidden.shape[1:] != (1, 1024):
            raise ValueError('Expected [batch, 1, 1024] language hidden state')
        self._language_context = hidden  # Keep the edge to the trainable input token.


@DETECTORS.register_module()
class QwenLearnTokenDriveTransformer(QwenFrozenSemDriveTransformer):
    pass


from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class QwenTokenAuditHook(Hook):
    """First three updates must change the token identically across all DDP ranks."""
    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.initial = model.img_backbone.q_lang.detach().clone()
        self.previous = self.initial.clone()
        self.rows = []

    def after_train_iter(self, runner):
        if len(self.rows) >= 3:
            return
        import json
        import torch.distributed as dist
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        q = model.img_backbone.q_lang.detach()
        assert bool(torch.isfinite(q).all())
        changed = float((q - self.previous).abs().max())
        assert changed > 0, 'Optimizer did not update q_lang'
        stats = model.img_backbone._client.last_backward_stats
        assert stats['token_gradient_norm'] > 0 and stats['qwen_parameter_gradients'] == 0
        assert stats['recompute_max_error'] == 0 and stats['pending'] == 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        values = [torch.empty_like(q) for _ in range(world)]
        if dist.is_initialized():
            dist.all_gather(values, q)
        else:
            values[0].copy_(q)
        assert all(torch.equal(q, value) for value in values), 'DDP token parameters diverged'
        self.rows.append(dict(iter=runner.iter+1, update_max=changed,
            token_norm=float(q.norm()), ranks_equal=True, backward=stats))
        self.previous.copy_(q)
        if not dist.is_initialized() or dist.get_rank() == 0:
            path = Path(runner.work_dir)/'token_audit.json'
            value = dict(passed=len(self.rows) == 3, world_size=world, updates=self.rows,
                         delta_from_initial=float((q-self.initial).norm()))
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(value,indent=2)+'\n')
            temporary.replace(path)


@HOOKS.register_module()
class QwenPerfHook(Hook):
    def after_train_iter(self, runner):
        import json
        import torch.distributed as dist
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        client = model.img_backbone._client
        rank = dist.get_rank() if dist.is_initialized() else 0
        row = dict(iter=runner.iter + 1, forward=client.last_stats,
                   backward=client.last_backward_stats,
                   parent_peak_bytes=torch.cuda.max_memory_allocated())
        with (Path(runner.work_dir) / ('perf_rank%d.jsonl' % rank)).open('a') as stream:
            stream.write(json.dumps(row) + '\n')
