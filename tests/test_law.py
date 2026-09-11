"""CPU tests; load standalone LAW helpers without CUDA/MMCV dependencies."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / 'adzoo/drivetransformer/mmdet3d_plugin'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


law_module = load_file('standalone_law', PLUGIN / 'ours/law.py')
pairing = load_file('standalone_pairing', PLUGIN / 'datasets/law_pairing.py')
LAW = law_module.LAWLatentWorldModel


def make_law(**kwargs):
    torch.manual_seed(12)
    return LAW(embed_dims=16, num_cams=2, pool_size=(2, 2), action_steps=3,
               num_heads=4, num_layers=1, **kwargs)


def test_gradients_reach_current_features_and_predicted_action_only():
    model = make_law()
    current = torch.randn(2, 2, 16, 4, 4, requires_grad=True)
    future = torch.randn_like(current, requires_grad=True)
    action = torch.randn(2, 1, 3, 2, requires_grad=True)
    output = model(model.encode(current), model.select_action(action))
    result = model.loss(output, model.encode(future), torch.tensor([True, False]))
    result['loss_law'].backward()
    assert current.grad[0].abs().sum() > 0
    assert action.grad[0].abs().sum() > 0
    assert current.grad[1].abs().sum() == action.grad[1].abs().sum() == 0
    assert future.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_action_ablation_preserves_parameters_and_blocks_action_gradient():
    model = make_law(use_action=False)
    reference = make_law()
    assert model.state_dict().keys() == reference.state_dict().keys()
    action = torch.randn(2, 1, 3, 2, requires_grad=True)
    latents = model.encode(torch.randn(2, 2, 16, 4, 4))
    output = model(latents, model.select_action(action))
    torch.testing.assert_close(output, model(latents, model.select_action(action + 9)))
    model.loss(output, torch.randn_like(output), torch.ones(2, dtype=torch.bool))['loss_law'].backward()
    assert action.grad is not None and action.grad.abs().sum() == 0


def test_multimodal_selection_is_per_sample_and_keeps_waypoint_gradients():
    model = make_law()
    action = torch.randn(2, 3, 3, 2, requires_grad=True)
    selected = model.select_action(action, torch.tensor([[1., 3., 2.], [4., 1., 2.]]))
    torch.testing.assert_close(selected, torch.stack((action[0, 1], action[1, 0])))
    selected.sum().backward()
    assert action.grad[0, 1].sum() == action.grad[1, 0].sum() == 6
    assert action.grad.sum() == 12
    with pytest.raises(ValueError):
        model.select_action(action)


def test_all_invalid_is_finite_zero_with_full_parameter_graph():
    model = make_law()
    output = model(torch.randn(2, 8, 16), torch.randn(2, 3, 2))
    result = model.loss(output, torch.full_like(output, float('nan')), torch.zeros(2, dtype=torch.bool))
    assert result['loss_law'].item() == result['law_mse'].item() == 0
    result['loss_law'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0
               for p in model.parameters())


def test_target_eval_does_not_update_bn_and_restores_mixed_modes_on_error():
    module = torch.nn.Sequential(torch.nn.BatchNorm1d(16), torch.nn.Dropout(0.8))
    module.train()
    module[1].eval()
    before = copy.deepcopy(module.state_dict())
    with pytest.raises(RuntimeError):
        with torch.no_grad(), law_module.temporary_eval(module):
            assert not any(child.training for child in module.modules())
            module(torch.randn(4, 16))
            raise RuntimeError('test restoration')
    assert module.training and module[0].training and not module[1].training
    for key, value in before.items():
        torch.testing.assert_close(value, module.state_dict()[key])


@pytest.mark.parametrize('reason', ['valid', 'end_of_dataset', 'scene_boundary',
                                   'frame_gap', 'camera_mismatch', 'missing_image'])
def test_pair_boundaries_and_missing_images(tmp_path, reason):
    image = tmp_path / 'frame.jpg'
    image.touch()
    sensors = {'CAM_FRONT': {'data_path': image.name}}
    infos = [dict(frame_idx=i, folder='route_a', sensors=copy.deepcopy(sensors)) for i in range(16)]
    if reason == 'end_of_dataset':
        infos.pop()
    elif reason == 'scene_boundary':
        infos[-1]['folder'] = 'route_b'
    elif reason == 'frame_gap':
        infos[-1]['frame_idx'] = 16
    elif reason == 'camera_mismatch':
        infos[-1]['sensors'] = {'CAM_BACK': {'data_path': image.name}}
    elif reason == 'missing_image':
        image.unlink()
    paths, meta = pairing.find_future_pair(0, len(infos), infos.__getitem__, str(tmp_path), 15)
    assert meta['reason'] == reason
    assert meta['valid'] == (reason == 'valid')
    if reason == 'valid':
        assert paths == [str(image)] and meta['delta_seconds'] == 1.5
    else:
        assert paths is None


def ddp_worker(rank, init_file, result_dir, all_invalid):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2)
    try:
        model = DDP(make_law())
        torch.manual_seed(33 + rank)
        output = model(torch.randn(1, 8, 16), torch.randn(1, 3, 2))
        target = torch.randn_like(output)
        valid = torch.tensor([rank == 0 and not all_invalid])
        result = model.module.loss(output, target, valid)
        result['loss_law'].backward()
        torch.save({name: p.grad for name, p in model.module.named_parameters()},
                   Path(result_dir) / ('grads%d.pt' % rank))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('all_invalid', [False, True])
def test_ddp_uneven_valid_counts_match_single_process_gradient(tmp_path, all_invalid):
    mp.spawn(ddp_worker, args=(str(tmp_path / 'init'), str(tmp_path), all_invalid),
             nprocs=2, join=True)
    model = make_law()
    torch.manual_seed(33)
    output = model(torch.randn(1, 8, 16), torch.randn(1, 3, 2))
    target = torch.randn_like(output)
    model.loss(output, target, torch.tensor([not all_invalid]))['loss_law'].backward()
    for rank in range(2):
        gradients = torch.load(tmp_path / ('grads%d.pt' % rank), weights_only=True)
        for name, p in model.named_parameters():
            torch.testing.assert_close(gradients[name], p.grad, atol=2e-7, rtol=2e-5)
