"""The ablation adds exactly 1024 parameters without changing existing init/config."""
from pathlib import Path
import json
import os
import sys
import tempfile
import torch
root=Path(__file__).resolve().parents[2]
os.chdir(root);sys.path.insert(0,str(root))
from mmcv import Config
from mmcv.models import build_model
import adzoo.drivetransformer.mmdet3d_plugin
folder=root/'adzoo/drivetransformer/configs/drivetransformer'
old=Config.fromfile(str(folder/'drivetransformer_qwen08_frozensem_10ep.py'))
new=Config.fromfile(str(folder/'drivetransformer_qwen08_learntoken_10ep.py'))
for key in ['data','optimizer','optimizer_config','lr_config','runner','inference_contract','fp16']:
    assert old[key]==new[key],key
assert old.scene_prompt==new.scene_prompt
states=[]
for cfg in [old,new]:
    torch.manual_seed(0)
    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    states.append({k:v.clone() for k,v in model.state_dict().items()})
    del model
assert set(states[1])-set(states[0])=={'img_backbone.q_lang'}
assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())
assert states[1]['img_backbone.q_lang'].numel()==1024
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'roundtrip.py'
    new.dump(str(path));restored=Config.fromfile(str(path))
    assert restored.scene_prompt==new.scene_prompt
    assert restored.model==new.model
receipt=dict(passed=True,existing_state_tensors_identical=len(states[0]),extra_parameters=1024,
             data_optimizer_schedule_inference_unchanged=True,fixed_english_prompt_unchanged=True,
             config_roundtrip=True)
(root.parent/'controls_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
