"""Frozen native Qwen with one external soft token and recomputed token VJP.

Forward saves detached front features on CPU; backward recomputes one language
chunk at a time. No Qwen parameter gradients or cross-frame KV cache is retained.
"""
import argparse
import ctypes
import os
from pathlib import Path
import shutil
import signal
import socket
import time
import traceback
import numpy as np
import torch
from protocol import receive, send
from worker import FrozenExtractor


class LearnTokenExtractor(FrozenExtractor):
    def __init__(self, settings, device):
        super().__init__(settings, device)
        self.batch_consistency = settings.get('batch_consistency', 'native')
        assert self.batch_consistency in ('native', 'single_sample_kernels')
        if self.batch_consistency == 'single_sample_kernels':
            from stable_batch import install_language_batch_consistency
            self.receipt['stable_batch'] = install_language_batch_consistency(self.model.language_model)
        self.receipt['batch_consistency'] = self.batch_consistency
        self.pending = {}
        self.current_token = None
        self.front_chunks = None
        self.live_graphs = []
        self.strategy = settings.get('backward_strategy', 'recompute')
        assert self.strategy in ('recompute', 'retain')
        self.receipt.update(readout='appended learnable input token, final normalized hidden state',
                            gradient='recomputed first-order VJP to external FP32 token only')

    def token_inputs(self, merged, token, h, w):
        inputs = self.prompt_inputs(len(merged), h, w)
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        assert bool(mask.bool().all()), 'Fixed prompt/geometry must have no padding'
        embeddings = self.model.get_input_embeddings()(ids)
        image_mask = inputs['mm_token_type_ids'].bool().unsqueeze(-1).expand_as(embeddings)
        embeddings = embeddings.masked_scatter(image_mask, merged.to(embeddings.dtype))
        # Expand in FP32 before casting so shared-token gradients sum in FP32.
        # Preserve the original path unless the consistency mode is enabled.
        if self.batch_consistency == 'single_sample_kernels':
            readout_token = token.expand(len(merged), -1, -1).to(embeddings.dtype)
        else:
            readout_token = token.to(embeddings.dtype).expand(len(merged), -1, -1)
        embeddings = torch.cat((embeddings, readout_token), dim=1)
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
        # A text position follows the native multimodal prefix without shifting it.
        next_position = inputs['position_ids'].amax(dim=(0, 2)).view(1, -1, 1) + 1
        positions = torch.cat((inputs['position_ids'], next_position.expand(3, -1, -1)), dim=-1)
        return dict(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                    use_cache=False, return_dict=True)

    def token_language(self, merged, token, h, w):
        output = self.model.language_model(**self.token_inputs(merged, token, h, w))
        return output.last_hidden_state[:, -1:, :]

    def language(self, merged, h, w):
        assert self.current_token is not None
        if self.front_chunks is not None:
            self.front_chunks.append(merged.detach().cpu())
        if self.front_chunks is not None and self.strategy == 'retain':
            with torch.enable_grad():
                token = self.current_token.detach().requires_grad_(True)
                output = self.token_language(merged.detach(), token, h, w).float()
            self.live_graphs.append((output, token))
            return output.detach()
        return self.token_language(merged, self.current_token, h, w)

    @torch.no_grad()
    def extract_token(self, image, token, request_id, track):
        assert request_id not in self.pending
        assert len(self.pending) < 4, 'Unconsumed token forwards; refuse graph leakage'
        token = torch.as_tensor(np.array(token, copy=True), device=self.device)
        assert token.shape == (1, 1, 1024) and bool(torch.isfinite(token).all())
        self.current_token = token
        self.live_graphs = []
        self.front_chunks = [] if track else None
        try:
            visual, hidden = super().extract(image)
            if track:
                self.pending[request_id] = dict(front=torch.cat(self.front_chunks),
                    token=token.cpu(), h=image.shape[-2], w=image.shape[-1],
                    hidden=np.array(hidden, copy=True), graphs=self.live_graphs)
            return visual, hidden
        finally:
            self.current_token = None
            self.front_chunks = None
            self.live_graphs = []

    def token_vjp(self, request_id, gradient):
        assert request_id in self.pending, 'Unknown or already consumed token backward'
        saved = self.pending.pop(request_id)
        assert gradient.shape == saved['hidden'].shape and np.isfinite(gradient).all()
        total = torch.zeros_like(saved['token'], device=self.device)
        max_error = 0.0
        chunk = self.settings['language_chunk']
        with torch.enable_grad():
            for start in range(0, len(saved['front']), chunk):
                if self.strategy == 'retain':
                    output, token = saved['graphs'][start // chunk]
                    front = None
                else:
                    token = saved['token'].to(self.device).detach().requires_grad_(True)
                    front = saved['front'][start:start+chunk].to(self.device)
                    output = self.token_language(front, token, saved['h'], saved['w']).float()
                expected = torch.as_tensor(saved['hidden'][start:start+chunk], device=self.device)
                error = float((output.detach() - expected).abs().max())
                max_error = max(error, max_error)
                assert error == 0.0, 'Forward/recompute mismatch; refuse incorrect token gradient'
                upstream = torch.as_tensor(np.array(gradient[start:start+chunk], copy=True), device=self.device)
                vjp, = torch.autograd.grad(output, token, grad_outputs=upstream)
                assert bool(torch.isfinite(vjp).all())
                total.add_(vjp)
                del output, vjp, token, front, expected, upstream
        assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())
        assert not self.model.training
        self.last_vjp = dict(backward_strategy=self.strategy, recompute_max_error=max_error, token_gradient_norm=float(total.norm()),
                             qwen_parameter_gradients=0, pending=len(self.pending))
        return total.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--parent', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    def terminate(signum, frame):
        shutil.rmtree(args.directory, ignore_errors=True)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    assert ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0
    if os.getppid() != args.parent:
        raise RuntimeError('Trainer exited before worker initialization')
    with socket.socket(fileno=args.fd) as sock:
        try:
            extractor = LearnTokenExtractor(receive(sock), args.device)
            send(sock, extractor.receipt)
            while True:
                message = receive(sock)
                command = message['command']
                if command == 'close':
                    break
                start = time.monotonic()
                stats = dict(request_id=message['request_id'])
                if command == 'extract_token':
                    image = np.load(args.directory/'input.npy', mmap_mode='r')
                    token = np.load(args.directory/'token.npy', mmap_mode='r')
                    assert list(image.shape) == message['shape']
                    visual, hidden = extractor.extract_token(image, token,
                        message['request_id'], message['track'])
                    np.load(args.directory/'visual.npy', mmap_mode='r+')[:] = visual
                    np.load(args.directory/'language.npy', mmap_mode='r+')[:] = hidden
                    stats.update(visual_shape=list(visual.shape), language_shape=list(hidden.shape))
                elif command == 'token_vjp':
                    gradient = np.load(args.directory/'hidden_gradient.npy', mmap_mode='r')
                    vjp = extractor.token_vjp(message['request_id'], gradient)
                    np.load(args.directory/'token_gradient.npy', mmap_mode='r+')[:] = vjp
                    stats.update(extractor.last_vjp)
                else:
                    raise ValueError('Unknown token worker command: '+command)
                stats.update(seconds=time.monotonic()-start,
                             peak_cuda_bytes=torch.cuda.max_memory_allocated())
                send(sock, stats)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            try:
                send(sock, dict(error=error))
            except OSError:
                pass
            raise


if __name__ == '__main__':
    main()
