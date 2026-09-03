_base_ = ["./drivetransformer_base_dinov3_vits16.py"]

# Experiment: FULL fine-tuning of the DINOv3 ViT-S/16 trunk (all parameters
# trainable). The trunk keeps a 0.1x learning rate so the pretrained features
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

# Same 10-epoch protocol as the frozen variants.
data = dict(samples_per_gpu=20)
runner = dict(type="IterBasedRunner", max_iters=25000)
checkpoint_config = dict(interval=500, max_keep_ckpts=3)
