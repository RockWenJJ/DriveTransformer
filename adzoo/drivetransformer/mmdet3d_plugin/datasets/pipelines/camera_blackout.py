import numpy as np
from mmcv.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomCameraBlackout(object):
    """Zero out one randomly chosen camera image. Applied after
    NormalizeMultiviewImage, where 0.0 equals the per-channel mean color,
    keeping the fill in-distribution."""

    def __init__(self, prob=0.15, num_views=6):
        self.prob = prob
        self.num_views = num_views

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        idx = np.random.randint(min(self.num_views, len(results["img"])))
        results["img"][idx] = np.zeros_like(results["img"][idx])
        return results
