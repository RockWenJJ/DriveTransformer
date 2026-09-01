"""Focused checks for Directed Gated Residual Cross-Temporal Attention."""

import sys

import torch

sys.path.insert(0, '.')
from mmcv import Config
from adzoo.drivetransformer.mmdet3d_plugin.ours.drivetransformer_layers import (
    DriveTransformerDecoderLayer,
)


torch.manual_seed(0)
DEV = 'cuda'
D, B = 64, 2
NA, NM, NE = 6, 4, 1
L, NAP, NMP, NPT = 3, 3, 2, 2
NIMG, FFN = 12, 128


def build_layer(directed=False):
    attn = dict(type='AttentionLayer', embed_dims=D, head_dim=32, attn_drop=0.0)
    return DriveTransformerDecoderLayer(
        attn_cfgs=[dict(attn), dict(attn), dict(attn, no_wq=True)],
        ffn_cfgs=dict(
            type='SwiGLULayer', embed_dims=D,
            feedforward_channels=FFN, ffn_drop=0.0),
        operation_order=(
            'task_self_attn', 'norm', 'temporal_cross_attn', 'norm',
            'sensor_cross_attn', 'norm', 'ffn', 'norm'),
        with_cp=False,
        agent_query_num=NA,
        map_query_num=NM,
        memory_len_frame=L,
        agent_num_propagated=NAP,
        map_num_propagated=NMP,
        map_pts_per_vec=NPT,
        directed_gated_cross_task_temporal_attn=directed,
    ).to(DEV).eval()


def inputs(seed=0):
    generator = torch.Generator(device=DEV).manual_seed(seed)
    randn = lambda *shape: torch.randn(
        *shape, generator=generator, device=DEV)
    return dict(
        agent_query=randn(B, NA, D),
        map_query=randn(B, NM, D),
        ego_query=randn(B, NE, D),
        img_feats=randn(B, NIMG, D),
        img_pos_embed=randn(B, NIMG, D),
        agent_temp_memory=randn(B, L * NAP, D),
        agent_temp_pos=randn(B, L * NAP, D),
        map_temp_memory=randn(B, L * NMP, D),
        map_temp_pos=randn(B, L * NMP, D),
        ego_temp_memory=randn(B, L, D),
        ego_temp_pos=randn(B, L, D),
        agent_pos_embed=randn(B, NA, D),
        map_pts_pos_embed=randn(B, NM * NPT, D),
        map_ins_pos_embed=randn(B, NM, D),
        ego_pos_embed=randn(B, NE, D),
        temp_attn_masks=torch.ones(B, L, 1, device=DEV),
    )


def assert_equal(left, right, label):
    assert torch.equal(left, right), (
        label, (left - right).abs().max().item())
    print(label, 'PASS')


cfg = Config.fromfile(
    'adzoo/drivetransformer/configs/drivetransformer/'
    'drivetransformer_base_dgrcta.py')
layer_cfg = cfg.model.pts_bbox_head.transformer.decoder.transformerlayers
assert layer_cfg.directed_gated_cross_task_temporal_attn is True
print('[config] directed flag PASS')

base = build_layer()
directed = build_layer(directed=True)
missing, unexpected = directed.load_state_dict(base.state_dict(), strict=False)
assert not unexpected
assert missing and all('directed_' in key for key in missing), missing

x = inputs()
base_out = base(**x)
directed_out = directed(**x)
for task, left, right in zip(
        ('agent', 'map', 'ego'), base_out, directed_out):
    assert_equal(left, right, '[zero gate] baseline exact ' + task)

with torch.no_grad():
    directed.directed_cross_temporal_gates[0].fill_(0.5)
map_agent_out = directed(**x)
assert not torch.equal(base_out[0], map_agent_out[0])
assert_equal(base_out[1], map_agent_out[1], '[routing] map unchanged')
assert_equal(base_out[2], map_agent_out[2], '[routing] ego unchanged')
print('[routing] map->agent changes agent PASS')

with torch.no_grad():
    directed.directed_cross_temporal_gates.zero_()
x_agent_perturbed = {key: value.clone() for key, value in x.items()}
x_agent_perturbed['agent_temp_memory'] += torch.randn_like(
    x_agent_perturbed['agent_temp_memory']) * 0.1
closed_out = directed(**x)
closed_perturbed = directed(**x_agent_perturbed)
assert not torch.equal(closed_out[0], closed_perturbed[0])
assert_equal(
    closed_out[1], closed_perturbed[1],
    '[routing] map ignores agent memory')
assert_equal(
    closed_out[2], closed_perturbed[2],
    '[routing] closed agent->ego edge isolates ego')

with torch.no_grad():
    directed.directed_cross_temporal_gates[1].fill_(0.5)
agent_ego_out = directed(**x)
agent_ego_perturbed = directed(**x_agent_perturbed)
assert_equal(
    agent_ego_out[1], agent_ego_perturbed[1],
    '[routing] map ignores agent->ego edge')
assert not torch.equal(agent_ego_out[2], agent_ego_perturbed[2])
print('[routing] agent->ego changes ego PASS')

x_invalid = {key: value.clone() for key, value in x.items()}
x_invalid['temp_attn_masks'].zero_()
invalid_out = directed(**x_invalid)
x_invalid_perturbed = {
    key: value.clone() for key, value in x_invalid.items()}
x_invalid_perturbed['agent_temp_memory'] += torch.randn_like(
    x_invalid_perturbed['agent_temp_memory'])
invalid_perturbed_out = directed(**x_invalid_perturbed)
for task, left, right in zip(
        ('agent', 'map', 'ego'), invalid_out, invalid_perturbed_out):
    assert_equal(left, right, '[mask] invalid memories ignored ' + task)

directed.train()
directed.zero_grad(set_to_none=True)
with torch.no_grad():
    directed.directed_cross_temporal_gates.zero_()
loss = sum(t.float().square().mean() for t in directed(**x))
loss.backward()
assert directed.directed_cross_temporal_gates.grad is not None
assert directed.directed_cross_temporal_gates.grad.abs().sum() > 0
print('[grad] zero gates receive gradient PASS')
closed_parameters = list(
    directed.directed_cross_temporal_attention_layers.parameters())
closed_parameters.append(directed.directed_cross_source_embeddings)
assert all(
    parameter.grad is None or parameter.grad.abs().sum() == 0
    for parameter in closed_parameters)
print('[grad] closed cross paths receive no gradient PASS')

directed.zero_grad(set_to_none=True)
with torch.no_grad():
    directed.directed_cross_temporal_gates.fill_(0.2)
loss = sum(t.float().square().mean() for t in directed(**x))
loss.backward()
cross_grad = directed.directed_cross_temporal_attention_layers[0].w_k[1].weight.grad
assert cross_grad is not None and cross_grad.abs().sum() > 0
print('[grad] opened cross projection receives gradient PASS')
print('ALL DGR-CTA CHECKS PASSED')
