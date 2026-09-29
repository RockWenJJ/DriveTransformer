import pickle
import unittest
import numpy as np
from adzoo.drivetransformer.mmdet3d_plugin.datasets.mixed_routes import (
    MixedRouteDataset, MixedRouteBatchSampler, RandomWindowSample, FullRouteSample)


def fixture():
    lengths=(84,100,120,160,400)
    ds=object.__new__(MixedRouteDataset)
    ds.drop_prefix=25;ds.sub_seq_lenth=-1;ds.use_splited_data=True
    ds.data_infos=dict(divide_nums=np.cumsum(lengths).tolist())
    ds.flag=np.repeat(np.arange(len(lengths)),lengths)
    return ds


class MixedRoutes(unittest.TestCase):
    def test_modes_prefix_and_continuity(self):
        it=iter(MixedRouteBatchSampler(fixture(),4,1,0,seed=4))
        last=[None]*4;counts={'random':0,'full':0};lengths=set()
        for _ in range(50000):
            for i,t in enumerate(next(it)):
                self.assertGreaterEqual(t.start,t.route_start+25)
                self.assertLessEqual(t.stop,t.route_stop)
                if t.sequence_mode=='full':
                    self.assertEqual((t.start,t.stop),(t.route_start+25,t.route_stop))
                else:self.assertTrue(40<=t.stop-t.start<=80)
                p=last[i]
                if t.index==t.start:
                    counts[t.sequence_mode]+=1
                    if t.sequence_mode=='random':lengths.add(t.stop-t.start)
                    if p:self.assertEqual((p.index,p.visit+1),(p.stop-1,t.visit))
                else:self.assertEqual((t.index,t.visit,t.sequence_mode),(p.index+1,p.visit,p.sequence_mode))
                last[i]=t
        self.assertEqual(lengths,set(range(40,81)))
        self.assertTrue(.45<counts['random']/sum(counts.values())<.55)

    def test_seed_rank_and_pickling(self):
        a=iter(MixedRouteBatchSampler(fixture(),2,2,0,seed=7))
        b=iter(MixedRouteBatchSampler(fixture(),2,2,0,seed=7))
        c=iter(MixedRouteBatchSampler(fixture(),2,2,1,seed=7))
        for _ in range(300):
            x,y,z=next(a),next(b),next(c);self.assertEqual(x,y)
            self.assertEqual([t.slot for t in z],[2,3])
            for t in x:
                copy=pickle.loads(pickle.dumps(t));self.assertIs(type(copy),type(t))
                self.assertEqual(copy.sequence_mode,t.sequence_mode)

    def test_probability_endpoints(self):
        for p,kind in [(0,FullRouteSample),(1,RandomWindowSample)]:
            it=iter(MixedRouteBatchSampler(fixture(),2,1,0,random_probability=p))
            for _ in range(500):self.assertTrue(all(isinstance(t,kind) for t in next(it)))
        with self.assertRaises(ValueError):MixedRouteBatchSampler(fixture(),random_probability=1.1)

    def test_full_descriptor_cannot_be_truncated(self):
        ds=fixture()
        with self.assertRaises(ValueError):ds[FullRouteSample(26,26,84,0,84,0,0)]

if __name__=='__main__':unittest.main()
