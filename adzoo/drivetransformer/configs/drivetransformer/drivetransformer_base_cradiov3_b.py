_base_ = ["./drivetransformer_base.py"]

# NVIDIA C-RADIOv3-B: a ViT-B/16 agglomerative model distilled from
# DFN CLIP + SigLIP2 + DINOv2-g + SAM. Size-matched to DINOv3 ViT-B/16
# (90M vs 86M, same patch size), so the two runs isolate the pretraining
# recipe rather than model capacity.
model = dict(
    pretrained=None,
    img_backbone=dict(
        _delete_=True,
        type="FrozenCRADIOViT",
        variant="c-radio_v3-b",
        checkpoint="./ckpts/c-radio_v3-b/model.safetensors",
        out_channels=512,
        forward_chunk_size=12,
    ),
    img_neck=None,
)

# Match the existing 4-GPU, 10-epoch-equivalent baseline protocol.
data = dict(samples_per_gpu=20)
runner = dict(type="IterBasedRunner", max_iters=25000)
checkpoint_config = dict(interval=500, max_keep_ckpts=3)
