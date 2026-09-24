"""Standalone frozen Qwen runtime (Python 3.10, torch 2.6, transformers 5.3).

Receives already augmented DT RGB tensors. Keeps the camera geometry unchanged.
All six cameras use the original ViT/merger; only camera zero enters the LLM.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5Model
from protocol import receive, send


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path, revision):
    receipt = json.loads((path.parent.parent/'weights_verified.json').read_text())
    assert receipt['verified'] and receipt['revision'] == revision == path.name
    assert receipt['repo'] == 'Qwen/Qwen3.5-0.8B'
    for name, metadata in receipt['files'].items():
        assert (path/name).stat().st_size == metadata['size'], name
        assert sha256(path/name) == metadata['sha256'], name
    return receipt


def pack_images(rgb_normalized):
    """Native Qwen2VL patch packing, without any resize or pixel rescaling."""
    n, c, h, w = rgb_normalized.shape
    assert c == 3 and h % 32 == 0 and w % 32 == 0
    # A still image is duplicated across the native temporal patch of size two.
    frames = rgb_normalized[:, None].expand(n, 2, c, h, w)
    patches = frames.reshape(n, 1, 2, c, h//32, 2, 16, w//32, 2, 16)
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return patches.reshape(n * (h//16) * (w//16), c * 2 * 16 * 16)


def last_valid(hidden, mask):
    positions = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(mask == 0, -1).max(-1).values
    assert bool((indices >= 0).all())
    return hidden[torch.arange(len(hidden), device=hidden.device), indices].unsqueeze(1)


class FrozenExtractor:
    def __init__(self, settings, device):
        assert transformers.__version__ == '5.3.0', transformers.__version__
        assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__
        self.settings = settings
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.cuda.set_per_process_memory_fraction(settings['memory_fraction'], self.device)
        torch.set_num_threads(1)
        path = Path(settings['model_path'])
        verify_weights(path, settings['revision'])
        self.model, loading = Qwen3_5Model.from_pretrained(str(path), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
        assert not loading.get('missing_keys'), loading
        assert not loading.get('mismatched_keys'), loading
        assert not loading.get('error_msgs'), loading
        assert set(loading.get('unexpected_keys', [])) <= {'lm_head.weight'}, loading
        self.model.to(self.device).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.prompt_cache = {}
        self.mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1,3,1,1)
        self.std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1,3,1,1)
        self.receipt = dict(revision=settings['revision'], torch=torch.__version__,
            transformers=transformers.__version__, device=str(self.device),
            parameters=sum(p.numel() for p in self.model.parameters()),
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            missing_keys=sorted(loading.get('missing_keys', [])),
            unexpected_keys=sorted(loading.get('unexpected_keys', [])),
            readout='last valid input token after native non-thinking assistant prefix',
            prompt_sha256=hashlib.sha256(settings['prompt'].encode()).hexdigest())

    def prompt_inputs(self, batch, h, w):
        key = batch, h, w
        if key not in self.prompt_cache:
            text = self.tokenizer.apply_chat_template([dict(role='user', content=[
                dict(type='image'), dict(type='text', text=self.settings['prompt'])])],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert text.count('<|image_pad|>') == 1
            text = text.replace('<|image_pad|>', '<|image_pad|>' * ((h//32)*(w//32)))
            tokens = self.tokenizer([text]*batch, add_special_tokens=False,
                                    return_tensors='pt', padding=True).to(self.device)
            ids, mask = tokens['input_ids'], tokens['attention_mask']
            grid = torch.tensor([[1,h//16,w//16]]*batch, device=self.device)
            mm = (ids == self.model.config.image_token_id).long()
            assert torch.all(mm.sum(-1) == (h//32)*(w//32))
            positions, _ = self.model.get_rope_index(ids, mm_token_type_ids=mm,
                image_grid_thw=grid, attention_mask=mask)
            self.prompt_cache[key] = dict(input_ids=ids, attention_mask=mask,
                image_grid_thw=grid, mm_token_type_ids=mm, position_ids=positions)
        return self.prompt_cache[key]

    @torch.no_grad()
    def language(self, merged, h, w):
        inputs = self.prompt_inputs(len(merged), h, w)
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        image_mask = inputs['mm_token_type_ids'].bool().unsqueeze(-1).expand_as(embeddings)
        embeddings = embeddings.masked_scatter(image_mask, merged.to(embeddings.dtype))
        output = self.model.language_model(inputs_embeds=embeddings,
            attention_mask=inputs['attention_mask'], position_ids=inputs['position_ids'],
            use_cache=False, return_dict=True)
        return last_valid(output.last_hidden_state, inputs['attention_mask'])

    @torch.no_grad()
    def extract(self, image):
        n, _, h, w = image.shape
        assert n % 6 == 0
        visual = []
        for start in range(0, n, self.settings['vision_chunk']):
            legacy = torch.as_tensor(np.array(image[start:start+self.settings['vision_chunk']], copy=True),
                                     device=self.device)
            qwen_pixels = ((legacy.float()*self.std + self.mean)/255. - 0.5)/0.5
            patches = pack_images(qwen_pixels).to(torch.bfloat16)
            grid = torch.tensor([[1,h//16,w//16]]*len(legacy), device=self.device)
            features = self.model.visual(patches, grid_thw=grid).pooler_output
            visual.append(features.reshape(len(legacy), h//32, w//32, 1024))
        visual = torch.cat(visual)
        front = visual[::6].reshape(n//6, (h//32)*(w//32), 1024)
        hidden = []
        for start in range(0, n//6, self.settings['language_chunk']):
            hidden.append(self.language(front[start:start+self.settings['language_chunk']], h, w))
        hidden = torch.cat(hidden)
        assert not visual.requires_grad and not hidden.requires_grad
        assert bool(torch.isfinite(visual).all()) and bool(torch.isfinite(hidden).all())
        return visual.permute(0,3,1,2).float().cpu().numpy(), hidden.float().cpu().numpy()


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
    # The socket handles normal exit; Linux parent-death signal handles a killed trainer.
    assert ctypes.CDLL(None).prctl(1, signal.SIGTERM) == 0
    if os.getppid() != args.parent:
        raise RuntimeError('Trainer exited before worker initialization')
    with socket.socket(fileno=args.fd) as sock:
        try:
            extractor = FrozenExtractor(receive(sock), args.device)
            send(sock, extractor.receipt)
            while True:
                message = receive(sock)
                if message['command'] == 'close':
                    break
                assert message['command'] == 'extract'
                start = time.monotonic()
                image = np.load(args.directory/'input.npy', mmap_mode='r')
                assert list(image.shape) == message['shape']
                visual, language = extractor.extract(image)
                np.load(args.directory/'visual.npy', mmap_mode='r+')[:] = visual
                np.load(args.directory/'language.npy', mmap_mode='r+')[:] = language
                send(sock, dict(seconds=time.monotonic()-start,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    visual_shape=list(visual.shape), language_shape=list(language.shape)))
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
