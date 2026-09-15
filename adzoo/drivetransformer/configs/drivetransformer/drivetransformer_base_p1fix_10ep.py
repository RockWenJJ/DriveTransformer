_base_ = ['./drivetransformer_base_consistency.py']
# Four P1 fixes only. Keep original Base architecture and motion loss.
model = dict(pretrained=dict(img='./ckpts/resnet50-19c8e357.pth'),
             pts_bbox_head=dict(motion_cls_valid_only=False))
# Retain ResNet image pretraining; no old DT model or optimizer state.
load_from = None
resume_from = None
num_gpus = 4
batch_size = 20
actual_training_frames = 234769
total_epochs = 10
num_iters_per_epoch = 2935
runner = dict(type="IterBasedRunner", max_iters=29350)
data = dict(samples_per_gpu=20, workers_per_gpu=12,
    train=dict(cache_lenth=21), val=dict(cache_lenth=21), test=dict(cache_lenth=21))
checkpoint_config = dict(interval=1000, save_last=True)
# Keep local logs; an external dashboard login must not block training.
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])
work_dir = './adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_p1fix_10ep'
