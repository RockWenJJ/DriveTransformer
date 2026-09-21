"""Verify tap identity/order, spatial concatenation, frozen gradients and config."""
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from mmcv import Config
from adzoo.drivetransformer.mmdet3d_plugin.models.backbones.dinov3 import (
    FrozenDINOv3ViT, DINOV3_VARIANTS)


def fake_load(model):
    for name in ('cls_token', 'mask_token', 'register_tokens'):
        torch.nn.init.zeros_(getattr(model.dinov3.embeddings, name))
    model._checkpoint_loaded = True
    if model.freeze_trunk:
        model._freeze_dinov3()


class IntermediateFeatures(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)

    def make(self, **kwargs):
        with patch.dict(DINOV3_VARIANTS, {'tiny16': dict(
                hidden_size=32, intermediate_size=64, num_heads=4)}), \
                patch.object(FrozenDINOv3ViT, '_load_pretrained', fake_load):
            return FrozenDINOv3ViT('tiny16', 'unused', out_channels=32,
                output_stride=16, amp_dtype=None, **kwargs)

    def test_taps_match_actual_blocks_and_original_final_output(self):
        model = self.make(out_indices=(3, 7, 11))
        x = torch.randn(2, 3, 32, 48)
        captured = {}
        handles = [model.dinov3.layer[i].register_forward_hook(
            lambda m, a, y, i=i: captured.__setitem__(i, y.detach().clone()))
            for i in (3, 7, 11)]
        taps = model.dinov3(x, out_indices=(3, 7, 11))
        for handle in handles:
            handle.remove()
        for index, tap in zip((3, 7, 11), taps):
            torch.testing.assert_close(tap, model.dinov3.norm(captured[index]))
            self.assertEqual(tuple(tap.shape), (2, 11, 32))  # 6 patches + 5 prefix
        torch.testing.assert_close(taps[-1], model.dinov3(x), rtol=0, atol=0)

    def test_spatial_concat_chunking_and_gradient_boundary(self):
        model = self.make(out_indices=(3, 7, 11), forward_chunk_size=1).train()
        x = torch.randn(2, 3, 32, 48, requires_grad=True)
        captured = []
        handle = model.adapter[0].register_forward_pre_hook(
            lambda m, a: captured.append(a[0].detach().clone()))
        y = model(x)[0]
        handle.remove()
        taps = model.dinov3(x.detach(), out_indices=(3, 7, 11))
        expected = torch.cat([t[:, 5:].transpose(1, 2).reshape(2, 32, 2, 3)
                              for t in taps], dim=1)
        torch.testing.assert_close(captured[0], expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(tuple(y.shape), (2, 32, 2, 3))
        (y * torch.randn_like(y)).sum().backward()
        self.assertIsNone(x.grad)
        self.assertFalse(model.dinov3.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in model.dinov3.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.adapter.parameters()))
        clone = self.make(out_indices=(3, 7, 11), forward_chunk_size=2)
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(clone(x.detach())[0], y.detach(), rtol=1e-4, atol=1e-5)

    def test_single_layer_compatibility_and_bad_indices(self):
        old = self.make()
        last = self.make(out_indices=(11,))
        last.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(1, 3, 32, 48)
        torch.testing.assert_close(last(x)[0], old(x)[0], rtol=0, atol=0)
        for indices in ((), (3, 3), (11, 3), (-1,), (12,), (True,), (1.5,)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.make(out_indices=indices)

    def test_large_training_contract(self):
        root = Path(__file__).resolve().parents[1]
        cfg = Config.fromfile(str(root / 'adzoo/drivetransformer/configs/drivetransformer/'
            'drivetransformer_large_vitb3_nomap_s16_res480_10ep.py'))
        self.assertEqual(cfg.model.type, 'NoMapDriveTransformer')
        self.assertEqual(cfg.model.pts_bbox_head.type, 'NoMapHead')
        self.assertEqual(tuple(cfg.model.img_backbone.out_indices), (3, 7, 11))
        self.assertTrue(cfg.model.img_backbone.freeze)
        self.assertIsNone(cfg.model.img_neck)
        self.assertEqual(cfg.model.img_backbone.out_channels, 768)
        head = cfg.model.pts_bbox_head
        self.assertEqual(head.img_stride, 16)
        self.assertEqual(head.embed_dims, 768)
        self.assertEqual(head.transformer.decoder.num_layers, 12)
        self.assertEqual(head.transformer.decoder.transformerlayers.ffn_cfgs.feedforward_channels, 3072)
        self.assertTrue(head.transformer.decoder.transformerlayers.attn_cfgs[0].extra_post_norm)
        self.assertEqual(list(head.ego_lcf_feat_idx), list(range(9)))
        for split in ('train', 'val', 'test'):
            self.assertEqual(tuple(cfg.data[split].data_aug_conf.final_dim), (480, 960))
            self.assertEqual(cfg.data[split].ego_accel_mode, 'xy')
            self.assertEqual(cfg.data[split].timestamp_frame_period_s, 0.1)
        self.assertEqual(cfg.inference_contract.ego_accel_mode, 'xy')
        self.assertEqual(cfg.runner.max_iters * cfg.data.samples_per_gpu * 4, 2348000)
        self.assertIsNone(cfg.load_from)
        self.assertIsNone(cfg.resume_from)


if __name__ == '__main__':
    unittest.main()
