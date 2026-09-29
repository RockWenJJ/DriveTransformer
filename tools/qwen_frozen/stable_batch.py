"""Opt-in BF16 batch consistency for the pinned LearnToken worker runtime.

Keep language Linear/SDPA kernels at their batch-one numerical shape while the
costly gated-delta recurrence stays batched. No weights or attention masks change.
The attention dispatch is local to the dedicated worker process and only applies
to explicitly marked language attention modules; vision uses native SDPA.
"""
import torch


def _sample_linear(original):
    def forward(x):
        if x.ndim == 3 and x.shape[0] > 1:
            return torch.cat([original(part) for part in x.split(1, dim=0)], dim=0)
        return original(x)
    return forward


def install_language_batch_consistency(language_model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    def stable_sdpa(module, query, key, value, attention_mask, **kwargs):
        if not getattr(module, '_learntoken_stable_batch', False) or query.shape[0] == 1:
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
        outputs = []
        for i in range(query.shape[0]):
            mask = attention_mask
            if mask is not None and mask.shape[0] > 1:
                mask = mask[i:i+1]
            output, weights = sdpa_attention_forward(module, query[i:i+1], key[i:i+1],
                                                     value[i:i+1], mask, **kwargs)
            assert weights is None
            outputs.append(output)
        return torch.cat(outputs, dim=0), None

    linear_count = attention_count = 0
    for module in language_model.modules():
        if isinstance(module, torch.nn.Linear):
            assert not getattr(module, '_learntoken_stable_batch', False), 'Already installed'
            module.forward = _sample_linear(module.forward)
            module._learntoken_stable_batch = True
            linear_count += 1
        elif isinstance(module, Qwen3_5Attention):
            assert module.config._attn_implementation == 'sdpa'
            module._learntoken_stable_batch = True
            attention_count += 1
    assert linear_count and attention_count
    ALL_ATTENTION_FUNCTIONS.register('sdpa', stable_sdpa)
    return dict(linear_modules=linear_count, attention_modules=attention_count,
                gated_delta_rule='native batched', convolution='native batched')
