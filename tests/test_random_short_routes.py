"""Boundary and replay tests that do not require training data or a GPU."""
import random
import unittest
import numpy as np
from adzoo.drivetransformer.mmdet3d_plugin.datasets.random_short_routes import (
    RandomShortRouteDataset, RandomShortRouteBatchSampler, draw_window)


def fixture(lengths=(84, 100, 120, 160)):
    ds=object.__new__(RandomShortRouteDataset)
    ds.drop_prefix=25;ds.sub_seq_lenth=-1;ds.use_splited_data=True
    ds.data_infos=dict(divide_nums=np.cumsum(lengths).tolist())
    ds.flag=np.repeat(np.arange(len(lengths)),lengths)
    return ds


class RandomWindows(unittest.TestCase):
    def test_bounds_and_random_lengths(self):
        rng=random.Random(10);windows=[draw_window(rng,1000,1250) for _ in range(2000)]
        self.assertEqual(set(b-a for a,b in windows),set(range(40,81)))
        self.assertGreater(len(set(a for a,b in windows)),100)
        self.assertTrue(all(1025<=a<b<=1250 for a,b in windows))
    def test_short_route_keeps_minimum_and_never_wraps(self):
        rng=random.Random(1)
        for _ in range(200):
            a,b=draw_window(rng,0,84,max_length=59);self.assertTrue(25<=a<b<=84 and 40<=b-a<=59)
        with self.assertRaises(ValueError):draw_window(rng,0,64)
    def test_slot_continuity_and_fresh_boundaries(self):
        sampler=iter(RandomShortRouteBatchSampler(fixture(),2,2,0,seed=4))
        previous=[None]*2;starts=0
        for _ in range(500):
            for i,t in enumerate(next(sampler)):
                p=previous[i]
                self.assertGreaterEqual(t.start,t.route_start+25)
                if t.index==t.start:
                    starts+=1
                    if p:self.assertEqual(p.index,p.stop-1)
                else:
                    self.assertEqual(t.index,p.index+1);self.assertEqual(t.visit,p.visit)
                previous[i]=t
        self.assertGreater(starts,15)
    def test_seed_and_rank_streams(self):
        a=iter(RandomShortRouteBatchSampler(fixture(),2,2,0,seed=7))
        b=iter(RandomShortRouteBatchSampler(fixture(),2,2,0,seed=7))
        c=iter(RandomShortRouteBatchSampler(fixture(),2,2,1,seed=7))
        for _ in range(100):
            x,y,z=next(a),next(b),next(c);self.assertEqual(x,y)
            self.assertEqual([t.slot for t in z],[2,3])
    def test_wrong_dataset_contract_is_rejected(self):
        ds=fixture();ds.drop_prefix=0
        with self.assertRaises(ValueError):RandomShortRouteBatchSampler(ds)
        ds=fixture((64,84,100,120))
        with self.assertRaises(ValueError):RandomShortRouteBatchSampler(ds)

if __name__=='__main__':unittest.main()
