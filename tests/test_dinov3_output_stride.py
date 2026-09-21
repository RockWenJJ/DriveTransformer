"""Adapter stride contract without requiring GPUs or allocating a full ViT-B."""
import unittest
from unittest.mock import patch
import torch
from adzoo.drivetransformer.mmdet3d_plugin.models.backbones.dinov3 import FrozenDINOv3ViT,DINOV3_VARIANTS
class StrideContract(unittest.TestCase):
 def make(self,**kwargs):
  with patch.dict(DINOV3_VARIANTS,{'tiny16':dict(hidden_size=32,intermediate_size=64,num_heads=4)}),patch.object(FrozenDINOv3ViT,'_load_pretrained'):
   return FrozenDINOv3ViT('tiny16','unused',out_channels=32,**kwargs)
 def test_default_preserves_stride32(self):
  m=self.make();self.assertEqual(m.output_stride,32);self.assertEqual(m.adapter(torch.randn(1,32,30,60)).shape,(1,32,15,30))
 def test_stride16_retains_spatial_resolution_and_gradients(self):
  m=self.make(output_stride=16);x=torch.randn(1,32,30,60,requires_grad=True);y=m.adapter(x)
  self.assertEqual(y.shape,(1,32,30,60));y.square().mean().backward();self.assertTrue(torch.isfinite(x.grad).all())
  self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.adapter.parameters()))
 def test_checkpoint_shapes_compatible(self):
  a=self.make();b=self.make(output_stride=16);b.load_state_dict(a.state_dict(),strict=True)
  self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
 def test_config_and_pixel_geometry(self):
  from pathlib import Path
  from mmcv import Config
  from adzoo.drivetransformer.mmdet3d_plugin.ours.utils import locations
  root=Path(__file__).resolve().parents[1]
  cfg=Config.fromfile(str(root/'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_vitb_frozen_nomap_res480_stride16_p1_10ep.py'))
  self.assertEqual(cfg.model.img_backbone.output_stride,16);self.assertEqual(cfg.model.pts_bbox_head.img_stride,16)
  pixel=locations((30,60),torch.device('cpu'),cfg.model.pts_bbox_head.img_stride,480,960)*torch.tensor([960.,480.])
  torch.testing.assert_close(pixel[0,0],torch.tensor([8.,8.]));torch.testing.assert_close(pixel[-1,-1],torch.tensor([952.,472.]))
 def test_unsupported_stride_rejected(self):
  for value in [8,24,64]:
   with self.assertRaises(ValueError):self.make(output_stride=value)
if __name__=='__main__':unittest.main()
