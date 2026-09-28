"""First-order autograd across the legacy DT / frozen Qwen process boundary."""
import time
import torch
from torch.autograd.function import once_differentiable
from bridge import FrozenQwenClient
from protocol import send


class _TokenReadout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token, images, client, track):
        visual, hidden, request_id = client.extract_token(images, token, track)
        ctx.client, ctx.request_id = client, request_id
        ctx.save_for_backward(token)
        ctx.mark_non_differentiable(visual)
        return visual, hidden

    @staticmethod
    @once_differentiable
    def backward(ctx, visual_gradient, hidden_gradient):
        token, = ctx.saved_tensors  # Detect mutation before backward.
        gradient = ctx.client.token_vjp(ctx.request_id, hidden_gradient)
        return gradient.to(dtype=token.dtype), None, None, None


class LearnTokenClient(FrozenQwenClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, worker_script='token_worker.py', **kwargs)
        self.request_counter = 0

    def __call__(self, images, token):
        if images.requires_grad:
            raise ValueError('The frozen visual input must not require gradients')
        track = torch.is_grad_enabled() and token.requires_grad
        return _TokenReadout.apply(token, images, self, track)

    def extract_token(self, images, token, track):
        if images.device != self.device or images.ndim != 4 or len(images) % 6:
            raise ValueError('Worker device or six-camera batch mismatch')
        n, c, h, w = images.shape
        if c != 3 or h % 32 or w % 32:
            raise ValueError('Qwen native patch/merge requires a multiple of 32')
        if token.shape != (1, 1, 1024) or token.device != images.device:
            raise ValueError('Expected one shared [1,1,1024] token on the image device')
        start = time.monotonic()
        self.request_counter += 1
        request_id = self.request_counter
        self._allocate('input', tuple(images.shape))[:] = images.detach().float().cpu().numpy()
        self._allocate('token', (1, 1, 1024))[:] = token.detach().float().cpu().numpy()
        visual = self._allocate('visual', (n, 1024, h//32, w//32))
        hidden = self._allocate('language', (n//6, 1, 1024))
        send(self.sock, dict(command='extract_token', shape=list(images.shape),
                            request_id=request_id, track=track))
        self.last_stats = self._response()
        if self.last_stats['request_id'] != request_id:
            raise RuntimeError('Mismatched language forward response')
        result = (torch.from_numpy(visual).to(device=self.device, copy=True),
                  torch.from_numpy(hidden).to(device=self.device, copy=True))
        self.last_stats['bridge_seconds'] = time.monotonic() - start
        return result[0], result[1], request_id

    def token_vjp(self, request_id, gradient):
        if gradient is None:
            raise RuntimeError('Missing language output gradient')
        self._allocate('hidden_gradient', tuple(gradient.shape))[:] = gradient.detach().float().cpu().numpy()
        result = self._allocate('token_gradient', (1, 1, 1024))
        send(self.sock, dict(command='token_vjp', request_id=request_id))
        self.last_backward_stats = self._response()
        if self.last_backward_stats['request_id'] != request_id:
            raise RuntimeError('Mismatched language backward response')
        return torch.from_numpy(result).to(device=self.device, copy=True)
