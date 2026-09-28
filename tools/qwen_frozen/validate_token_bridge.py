"""Legacy torch autograd against the native modern-runtime reference."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from token_bridge import LearnTokenClient

root = Path(sys.argv[1]).resolve()
torch.cuda.set_per_process_memory_fraction(.02)
reference = np.load(root/'native_reference.npz')
settings = json.loads((root/'worker_validation_settings.json').read_text())
settings['memory_fraction'] = .14
python = str(root.parent/'qwen08_frozensem_20260924/qwen_env/bin/python')
client = LearnTokenClient(Path(__file__).resolve().parent, torch.device('cuda:0'), python, **settings)
try:
    q = torch.nn.Parameter(torch.tensor(reference['token'],device='cuda'))
    images = torch.tensor(reference['images'],device='cuda')
    visual, hidden = client(images,q)
    assert not visual.requires_grad and hidden.requires_grad
    np.testing.assert_array_equal(visual.cpu().numpy(),reference['visual'])
    np.testing.assert_array_equal(hidden.detach().cpu().numpy(),reference['hidden'])
    (hidden*torch.tensor(reference['upstream'],device='cuda')).sum().backward()
    actual=q.grad.detach().cpu().numpy()
    error=float(np.linalg.norm(actual-reference['gradient'])/np.linalg.norm(reference['gradient']))
    assert error < .01, error
    before=q.detach().clone()
    torch.optim.AdamW([q],lr=1e-5).step()
    assert not torch.equal(before,q)
    with torch.no_grad():
        _, new_hidden=client(images,q)
        assert not new_hidden.requires_grad
    result=dict(passed=True,gradient_relative_l2=error,gradient_norm=float(q.grad.norm()),
        token_updated=True,forward_native_max_error=0.,worker=client.receipt,
        backward=client.last_backward_stats,forward=client.last_stats)
    (root/'bridge_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
finally:client.close()
