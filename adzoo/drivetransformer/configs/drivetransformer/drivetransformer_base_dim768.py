_base_ = ['./drivetransformer_base.py']

# Widen the base model while preserving depth, head count and FFN ratio.
_dim_ = 768
num_gpus = 4
batch_size = 20
total_epochs = 10
num_iters_per_epoch = 200000 // (num_gpus * batch_size)

model = dict(
    img_neck=dict(out_channels=_dim_),
    pts_bbox_head=dict(
        embed_dims=_dim_,
        agent_prep_decoder=dict(transformerlayers=dict(
            attn_cfgs=[
                dict(type='AttentionLayer', embed_dims=_dim_, head_dim=96, attn_drop=0.1),
                dict(type='AttentionLayer', embed_dims=_dim_, head_dim=96, attn_drop=0.1)],
            ffn_cfgs=dict(embed_dims=_dim_, feedforward_channels=_dim_ * 4),
            with_cp=True)),
        map_prep_decoder=dict(transformerlayers=dict(
            attn_cfgs=[
                dict(type='AttentionLayer', embed_dims=_dim_, head_dim=96, attn_drop=0.1),
                dict(type='AttentionLayer', embed_dims=_dim_, head_dim=96, attn_drop=0.1)],
            ffn_cfgs=dict(embed_dims=_dim_, feedforward_channels=_dim_ * 4),
            with_cp=True)),
        transformer=dict(
            embed_dims=_dim_,
            decoder=dict(
                embed_dims=_dim_,
                transformerlayers=dict(
                    feedforward_channels=_dim_ * 4,
                    with_cp=True,
                    attn_cfgs=[
                        dict(type='AttentionLayer', embed_dims=_dim_, head_dim=96,
                             attn_drop=0.1, layer_scale=1e-2),
                        dict(type='AttentionLayer', embed_dims=_dim_, head_dim=96,
                             attn_drop=0.1, layer_scale=1e-2),
                        dict(type='AttentionLayer', embed_dims=_dim_, head_dim=96,
                             attn_drop=0.1, no_wq=True)],
                    ffn_cfgs=dict(embed_dims=_dim_, feedforward_channels=_dim_ * 4))))))

data = dict(samples_per_gpu=batch_size,
            train=dict(cache_lenth=batch_size + 1),
            val=dict(cache_lenth=batch_size + 1),
            test=dict(cache_lenth=batch_size + 1))
runner = dict(type='IterBasedRunner', max_iters=total_epochs * num_iters_per_epoch)
checkpoint_config = dict(interval=500, max_keep_ckpts=3)
load_from = None
resume_from = None
