"""Native autograd reference and recomputed VJP parity at the real resolution."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from token_worker import LearnTokenExtractor


def weights_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    torch.manual_seed(19)
    settings = json.loads((root/'worker_validation_settings.json').read_text())
    worker = LearnTokenExtractor(settings, 'cuda:0')
    before = weights_digest(worker.model)
    rng = np.random.RandomState(19)
    images = rng.normal(size=(12,3,480,960)).astype(np.float32)
    token = rng.normal(scale=.02, size=(1,1,1024)).astype(np.float32)
    upstream = rng.normal(size=(2,1,1024)).astype(np.float32)/1024
    start = time.monotonic()
    visual, hidden = worker.extract_token(images, token, 1, True)
    saved = worker.pending[1]
    reference = torch.zeros(1,1,1024, device='cuda')
    outputs = []
    for i in range(2):
        q = torch.tensor(token, device='cuda', requires_grad=True)
        front = saved['front'][i:i+1].to('cuda')
        original = worker.prompt_inputs(1,480,960)
        args = worker.token_inputs(front,q,480,960)
        ids = torch.cat((original['input_ids'], torch.zeros_like(original['input_ids'][:,:1])),dim=1)
        mm = torch.cat((original['mm_token_type_ids'],torch.zeros_like(ids[:,:1])),dim=1)
        pos,_ = worker.model.get_rope_index(ids, mm_token_type_ids=mm,
            image_grid_thw=original['image_grid_thw'],attention_mask=args['attention_mask'])
        assert torch.equal(pos, args['position_ids'])
        assert torch.equal(pos[:,:,:-1],original['position_ids'])
        # Explicit native inputs_embeds graph, independently of recompute bookkeeping.
        embeds = worker.model.get_input_embeddings()(original['input_ids'])
        mask = original['mm_token_type_ids'].bool().unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(mask,front.to(embeds.dtype))
        embeds = torch.cat((embeds,q.to(embeds.dtype)),dim=1)
        y = worker.model.language_model(inputs_embeds=embeds,position_ids=pos,
            attention_mask=args['attention_mask'],use_cache=False,return_dict=True).last_hidden_state[:,-1:,:].float()
        grad, = torch.autograd.grad(y,q,torch.tensor(upstream[i:i+1],device='cuda'))
        reference += grad
        outputs.append(y.detach().cpu().numpy())
        del y,grad,embeds,q,args
    np.testing.assert_array_equal(np.concatenate(outputs),hidden)
    vjp = worker.token_vjp(1,upstream)
    np.testing.assert_array_equal(vjp,reference.cpu().numpy())
    assert np.linalg.norm(vjp)>0 and not worker.pending
    try:
        worker.token_vjp(1,upstream)
        raise RuntimeError('Repeated backward unexpectedly accepted')
    except AssertionError as exc:
        assert 'Unknown' in str(exc)
    # Two forwards consume their own saved inputs, in reverse backward order.
    altered = token + .01
    _, hidden2 = worker.extract_token(images,altered,2,True)
    _, hidden3 = worker.extract_token(images,token,3,True)
    assert not np.array_equal(hidden2,hidden3)
    g3 = worker.token_vjp(3,upstream)
    repeat_relative_l2 = float(np.linalg.norm(g3-vjp)/np.linalg.norm(vjp))
    assert repeat_relative_l2 < .01, repeat_relative_l2
    worker.token_vjp(2,upstream)
    assert not worker.pending
    worker.extract_token(images[:6],token,4,False)
    assert not worker.pending
    after = weights_digest(worker.model)
    assert before == after
    np.savez(root/'native_reference.npz',images=images,token=token,upstream=upstream,
             visual=visual,hidden=hidden,gradient=vjp)
    result = dict(passed=True,worker=worker.receipt,input_shape=list(images.shape),
        appended_sequence_length=original['input_ids'].shape[1]+1,
        native_mrope_prefix_and_append_equal=True,native_forward_max_error=0.,
        native_gradient_max_error=0.,repeat_bf16_gradient_relative_l2=repeat_relative_l2,token_gradient_norm=float(np.linalg.norm(vjp)),
        qwen_weights_before=before,qwen_weights_after=after,all_qwen_weights_unchanged=True,
        qwen_parameter_gradients=0,out_of_order_requests=True,duplicate_backward_rejected=True,
        inference_keeps_no_pending=True,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.monotonic()-start)
    (root/'worker_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
