# DT-Qwen08-FrozenSem

Configuration: `adzoo/drivetransformer/configs/drivetransformer/drivetransformer_qwen08_frozensem_10ep.py`.

Six 480×960 cameras share the original frozen ViT and 2×2 Patch Merger. The 15×30×1024 raster features feed a trainable 1×1 projection to the 512-dimensional DT decoder. The front-view merger features also feed the frozen language model with the fixed English prompt and native non-thinking assistant prefix. The final normalized hidden state of the last valid input token is detached and projected, then concatenated with the final decoder's ego+position feature before both planning heads. Detection, motion and earlier planning outputs are unchanged by the language input. No learned language token, generated text, navigation text or speed text is used.

## Environments and weights

Keep the existing Python3.8/Torch2.1 DT environment. Create a separate Python3.10 environment for `torch==2.6.0+cu118`, `torchvision==0.21.0+cu118`, `transformers==5.3.0`, `numpy==1.26.4`, `pillow==11.1.0`, `accelerate==1.5.2`. CUDA wheels are available from `https://download.pytorch.org/whl/cu118`. The experiment stores the full resolved requirements lock separately.

Use Qwen/Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`. Expected layout:

```
runtime_root/
  qwen_env/bin/python
  weights_verified.json
  weights/2fc06364715b967f1860aea9cf38778875588b17/...
```

The weight receipt contains `repo`, `revision`, `verified=true`, and `files[name]` with `size` and `sha256` for every downloaded file. The worker verifies the entire snapshot on startup and refuses missing/mismatched model parameters. Override `model.img_backbone.model_path`, `model.img_backbone.worker_python` and `qwen_contract.external_weights_manifest` together when changing hosts. The original Qwen weights are an external immutable dependency, not duplicated in DT checkpoints.

The worker uses the same visible CUDA device as its DT rank. Control messages use an inherited Unix socket; FP32 images/features use a private `/dev/shm` directory. Qwen runs in BF16 eval/no-grad mode. DT input normalization is inverted before native Qwen normalization; images are not resized again. Native patch ordering, duplicated still-image temporal patches, image placeholders and MRoPE are retained. Parent and worker memory ceilings can be specified for a bounded wiring test without changing the formal experiment budget.

## Verification

From the DT checkout, use the legacy environment for `validate_dt.py` and `test_config_prompt.py`; use the separate Qwen environment for `validate_worker.py`. `validate_dt.py --output <receipt>` audits the entire training camera order and inherited data/optimizer contract. `--gpu` additionally executes a real six-camera optimizer step, checks adapter gradients, isolates final planning conditioning, and strictly reloads its checkpoint. `validate_worker.py --settings <worker_validation_settings.json> --output <receipt>` checks preprocessing, native-vs-reused forward parity, frozen parameters, determinism, padding and image conditioning.

The actual training-entry smoke is separate from these tests. The queued formal gate requires a fresh three-step 4×20 run and strict checkpoint verification, followed by a fresh 29350-update run. Single-frame or one-GPU tests do not establish full-budget memory feasibility or driving quality.

Original sources: [pinned Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json), [Transformers v5.3.0 Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
