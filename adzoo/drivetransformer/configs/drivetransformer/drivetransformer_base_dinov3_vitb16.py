_base_ = ["./drivetransformer_base.py"]

model = dict(
    pretrained=None,
    img_backbone=dict(
        _delete_=True,
        type="FrozenDINOv3ViT",
        variant="vitb16",
        checkpoint="./ckpts/dinov3_vitb16_lvd1689m.safetensors",
        out_channels=512,
        forward_chunk_size=12,
    ),
    img_neck=None,
)

# Match the existing 4-GPU, 10-epoch-equivalent baseline protocol.
data = dict(samples_per_gpu=20)
runner = dict(type="IterBasedRunner", max_iters=25000)
checkpoint_config = dict(interval=500, max_keep_ckpts=3)

