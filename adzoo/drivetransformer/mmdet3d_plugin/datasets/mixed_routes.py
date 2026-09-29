"""Opt-in per-sequence random/full mixture, always after the physical prefix."""
import random
from mmcv.datasets import DATASETS
from mmcv.datasets.samplers.sampler import SAMPLER
from .random_short_routes import (
    SequenceSample, RandomShortRouteDataset, RandomShortRouteBatchSampler, draw_window)


class RandomWindowSample(SequenceSample):
    __slots__ = ()
    sequence_mode = 'random'


class FullRouteSample(SequenceSample):
    __slots__ = ()
    sequence_mode = 'full'


@DATASETS.register_module()
class MixedRouteDataset(RandomShortRouteDataset):
    def __getitem__(self, sample):
        if not isinstance(sample, (RandomWindowSample, FullRouteSample)):
            raise TypeError('Mixed-route training requires an explicit branch descriptor')
        if isinstance(sample, FullRouteSample) and (
                sample.start != sample.route_start + self.drop_prefix or sample.stop != sample.route_stop):
            raise ValueError('Full branch must cover the entire post-prefix physical route')
        result = super().__getitem__(sample)
        for meta in result['img_metas'].data.values():
            meta['sampled_sequence']['mode'] = sample.sequence_mode
        return result

    def get_data_info(self, index):
        result = super().get_data_info(index)
        result['scene_token'] = result['scene_token'].replace(':random60:', ':mixseq50:')
        return result


@SAMPLER.register_module()
class MixedRouteBatchSampler(RandomShortRouteBatchSampler):
    """Bernoulli branch per new slot sequence, NOT a frame-level mixture.

    Random length is uniform inclusive40..80; the inherited route stream skips
    routes too short for the drawn length. Full branch takes its next route
    whole after prefix25. Separate branch RNGs avoid length-dependent choices.
    Temporal state resets at every sequence start, even on revisiting a route.
    """
    def __init__(self, dataset, *args, random_probability=0.5, **kwargs):
        if not isinstance(dataset, MixedRouteDataset):
            raise TypeError('MixedRouteDataset required')
        if not 0 <= random_probability <= 1:
            raise ValueError('random_probability must be in [0,1]')
        super().__init__(dataset, *args, **kwargs)
        self.random_probability = float(random_probability)
        self.branch_rngs = [random.Random(self.seed + 2000003*(self.rank*self.batch_size+i+1))
                           for i in range(self.batch_size)]

    def __iter__(self):
        while True:
            batch = []
            for i in range(self.batch_size):
                if not self.buffer_per_local_sample[i]:
                    short = self.branch_rngs[i].random() < self.random_probability
                    length = self.window_rngs[i].randint(self.min_length, self.max_length) if short else 1
                    while True:
                        group = next(self.group_indices_per_global_sample_idx[i])
                        indices = self.group_idx_to_sample_idxs[group]
                        if len(indices)-self.drop_prefix >= length:
                            break
                    if indices[-1]-indices[0]+1 != len(indices):
                        raise ValueError('Physical route indices must be contiguous')
                    begin, end = indices[0], indices[-1]+1
                    if short:
                        start, stop = draw_window(self.window_rngs[i], begin, end,
                                                 self.drop_prefix, length, length)
                        sample_type = RandomWindowSample
                    else:
                        start, stop = begin+self.drop_prefix, end
                        sample_type = FullRouteSample
                    slot = self.rank*self.batch_size+i
                    visit = self.visits[i]; self.visits[i] += 1
                    self.buffer_per_local_sample[i] = [sample_type(j, start, stop, begin, end, slot, visit)
                                                       for j in range(start, stop)]
                batch.append(self.buffer_per_local_sample[i].pop(0))
            yield batch
