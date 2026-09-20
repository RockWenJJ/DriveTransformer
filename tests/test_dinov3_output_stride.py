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
 def test_unsupported_stride_rejected(self):
  for value in [8,24,64]:
   with self.assertRaises(ValueError):self.make(output_stride=value)
if __name__=='__main__':unittest.main()
