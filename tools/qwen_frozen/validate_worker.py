"""Check native preprocessing / reused-merger parity before integrating with DT."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from worker import FrozenExtractor, pack_images, last_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    torch.manual_seed(0)
    pixels = torch.randint(0,256,(2,3,480,960), dtype=torch.int32).float()
    processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
        image_mean=[.5]*3, image_std=[.5]*3)
    native = processor(images=[x.permute(1,2,0).numpy().astype(np.uint8) for x in pixels],
        do_resize=False, return_tensors='pt')
    ours = pack_images((pixels/255.-.5)/.5)
    patch_error = float((native['pixel_values']-ours).abs().max())
    assert patch_error <= 2e-7, patch_error
    assert native['image_grid_thw'].tolist() == [[1,30,60],[1,30,60]]
    h = torch.arange(24).reshape(2,4,3)
    mask = torch.tensor([[0,1,1,0],[1,1,1,1]])
    assert torch.equal(last_valid(h,mask).squeeze(1), torch.stack([h[0,2],h[1,3]]))
    extractor = FrozenExtractor(settings, 'cuda:0')
    start = time.monotonic()
    with torch.no_grad():
        patches = native['pixel_values'][:1800].to(device='cuda', dtype=torch.bfloat16)
        grid = native['image_grid_thw'][:1].cuda()
        merged = extractor.model.visual(patches, grid_thw=grid).pooler_output.reshape(1,450,1024)
        reuse = extractor.language(merged,480,960)
        inputs = extractor.prompt_inputs(1,480,960)
        native_output = extractor.model(pixel_values=patches, use_cache=False,
            return_dict=True, **{k:v for k,v in inputs.items() if k != 'position_ids'})
        native_last = last_valid(native_output.last_hidden_state, inputs['attention_mask'])
        error = float((reuse-native_last).abs().max())
        assert torch.equal(reuse,native_last), error
        again = extractor.language(merged,480,960)
        assert torch.equal(reuse,again)
        different = extractor.language(torch.zeros_like(merged),480,960)
        assert not torch.equal(reuse,different)
        # Six-camera path returns 15x30 raster features without a second ViT pass for front.
        legacy = ((pixels[:1].repeat(6,1,1,1).cuda()-extractor.mean)/extractor.std).cpu().numpy()
        visual, language = extractor.extract(legacy)
        assert visual.shape == (6,1024,15,30) and language.shape == (1,1,1024)
        assert all(np.array_equal(visual[0], x) for x in visual[1:])
        assert np.isfinite(visual).all() and np.isfinite(language).all()
    proof = dict(passed=True, patch_max_abs_error=patch_error,
        reused_features_vs_native_model_max_abs_error=error,
        repeated_forward_exact=True, image_conditioned_hidden=True,
        identical_cameras_identical_features=True, valid_token_padding_test=True,
        visual_shape=list(visual.shape), language_shape=list(language.shape),
        frozen=extractor.receipt, runtime_seconds=time.monotonic()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated())
    proof.update(prompt_tokens=int(inputs['input_ids'].shape[-1]),
        readout_token_id=int(inputs['input_ids'][0,-1]),
        readout_token_text=extractor.tokenizer.decode(inputs['input_ids'][0,-1:]),
        native_prefix_suffix=extractor.tokenizer.decode(inputs['input_ids'][0,-12:]))
    args.output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2), flush=True)


if __name__ == '__main__':
    main()
