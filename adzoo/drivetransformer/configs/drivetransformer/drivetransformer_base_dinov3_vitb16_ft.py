_base_ = ["./drivetransformer_base_dinov3_vitb16.py"]

# Experiment: FULL fine-tuning of the DINOv3 ViT-B/16 trunk (mirrors the
# vits16_ft recipe). Trunk keeps a 0.1x learning rate so pretrained features
# adapt without being washed out by the head's 1e-4 schedule.
model = dict(
    pretrained=None,
    img_backbone=dict(freeze=False),
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1, decay_mult=1.0),
        }
    )
)

# Same 10-epoch protocol (4-GPU, micro 20 -> 25000 iters).
data = dict(samples_per_gpu=20)
runner = dict(type="IterBasedRunner", max_iters=25000)
checkpoint_config = dict(interval=500, max_keep_ckpts=3)
