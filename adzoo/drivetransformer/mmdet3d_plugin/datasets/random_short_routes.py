"""Opt-in randomized contiguous training windows with explicit memory boundaries."""
from bisect import bisect_right
import random
from typing import NamedTuple

from mmcv.datasets import DATASETS
from mmcv.datasets.samplers.sampler import SAMPLER
from .bench2drive_drivetransformer_dataset import B2D_DriveTransformer_Dataset
from .samplers.group_sampler import InfiniteGroupEachSampleInBatchSampler


class SequenceSample(NamedTuple):
    index: int
    start: int
    stop: int
    route_start: int
    route_stop: int
    slot: int
    visit: int


def draw_window(rng, route_start, route_stop, drop_prefix=25, min_length=40, max_length=80):
    """Inclusive start offset25 means dropping offsets0..24 (first usable frame26)."""
    available = route_stop - route_start - drop_prefix
    if available < max_length:
        raise ValueError('Route has fewer than requested maximum eligible frames')
    length = rng.randint(min_length, max_length)
    start = rng.randint(route_start + drop_prefix, route_stop - length)
    return start, start + length


@SAMPLER.register_module()
class RandomShortRouteBatchSampler(InfiniteGroupEachSampleInBatchSampler):
    """Each batch slot advances a fresh random window of its assigned route.

    Route order uses the baseline shared seeded permutation; each global batch
    slot has an independent window RNG. Input exposures are fixed, not unique
    frame epochs. No sample wraps across a physical route boundary.
    """
    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=None,
                 seed=0, drop_prefix=25, min_length=40, max_length=80):
        if not (0 <= drop_prefix and 1 <= min_length <= max_length):
            raise ValueError('Invalid prefix/length bounds')
        if not isinstance(dataset, RandomShortRouteDataset):
            raise TypeError('RandomShortRouteDataset is required to reset memory')
        if dataset.drop_prefix != drop_prefix or dataset.sub_seq_lenth != -1:
            raise ValueError('Dataset/sampler prefix or physical route grouping mismatch')
        super().__init__(dataset, samples_per_gpu, num_replicas, rank, seed)
        if min(self.group_sizes)-drop_prefix < min_length or max(self.group_sizes)-drop_prefix < max_length:
            raise ValueError('Route lengths cannot support the requested window contract')
        self.drop_prefix = drop_prefix
        self.min_length, self.max_length = min_length, max_length
        self.window_rngs = [random.Random(self.seed + 1000003*(self.rank*self.batch_size+i+1))
                            for i in range(self.batch_size)]
        self.visits = [0]*self.batch_size

    def __iter__(self):
        while True:
            batch = []
            for i in range(self.batch_size):
                if not self.buffer_per_local_sample[i]:
                    # Draw length first so the global distribution remains U[40,80].
                    # Short routes remain eligible on visits with a feasible length.
                    length = self.window_rngs[i].randint(self.min_length, self.max_length)
                    while True:
                        group = next(self.group_indices_per_global_sample_idx[i])
                        indices = self.group_idx_to_sample_idxs[group]
                        if len(indices)-self.drop_prefix >= length:
                            break
                    if indices[-1]-indices[0]+1 != len(indices):
                        raise ValueError('Physical route indices must be contiguous')
                    begin, end = indices[0], indices[-1]+1
                    start, stop = draw_window(self.window_rngs[i], begin, end,
                        self.drop_prefix, length, length)
                    slot = self.rank*self.batch_size+i
                    visit = self.visits[i]; self.visits[i] += 1
                    self.buffer_per_local_sample[i] = [SequenceSample(j, start, stop, begin, end, slot, visit)
                                                       for j in range(start, stop)]
                batch.append(self.buffer_per_local_sample[i].pop(0))
            yield batch


@DATASETS.register_module()
class RandomShortRouteDataset(B2D_DriveTransformer_Dataset):
    """Read original annotations but reject dropped input/history frames.

    Future target horizons still use the remaining physical route, matching the
    baseline supervision. Only training input streams and memory boundaries are
    shortened; annotations beyond a window are targets, never future inputs.
    """
    def __init__(self, drop_prefix=25, **kwargs):
        self.drop_prefix = drop_prefix
        self._sequence_sample = None
        super().__init__(**kwargs)
        if self.test_mode or not self.use_splited_data or self.sub_seq_lenth != -1 or self.filter_empty_gt:
            raise ValueError('Use split-data full-route grouping, train mode and filter_empty_gt=False')

    def __getitem__(self, sample):
        if not isinstance(sample, SequenceSample):
            raise TypeError('Training requires an explicit sequence descriptor')
        route = bisect_right(self.divide_nums, sample.index)
        begin = self.divide_nums[route-1] if route else 0
        end = self.divide_nums[route]
        if not (sample.route_start == begin and sample.route_stop == end
                and begin+self.drop_prefix <= sample.start <= sample.index < sample.stop <= end):
            raise ValueError('Invalid sequence/physical route boundaries')
        self._sequence_sample = sample
        try:
            result = self.prepare_train_data(sample.index, self.get_augmentation())
            if result is None:
                raise RuntimeError('Rejected frame: do not silently switch route inside a sequence')
            # Preserve sampled source and window for per-frame audit across workers.
            for meta in result['img_metas'].data.values():
                meta['sampled_sequence'] = sample._asdict()
            return result
        finally:
            self._sequence_sample = None

    def get_data_info(self, index):
        sample = self._sequence_sample
        if sample is None or index != sample.index:
            raise RuntimeError('No active sequence descriptor')
        result = super().get_data_info(index)
        result['prev_exists'] = index != sample.start
        result['scene_token'] += ':random60:%d:%d' % (sample.slot, sample.visit)
        return result

    def is_in_same_route(self, cur_idx, adj_idx):
        # The original history function respects this guard before reading any
        # adjacent frame. Prefix frames must not re-enter through ego history.
        sample = self._sequence_sample
        if sample is not None and adj_idx < sample.route_start+self.drop_prefix:
            return False
        return super().is_in_same_route(cur_idx, adj_idx)
