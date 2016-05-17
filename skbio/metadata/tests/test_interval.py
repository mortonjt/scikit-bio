# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import unittest

#from skbio.metadata import IntervalMetadata, Interval
from skbio.metadata._interval import _polish_interval
from skbio.metadata._interval import Interval, IntervalMetadata

class TestInterval(unittest.TestCase):
    def setUp(self):
        pass

    def test_constructor(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1,2), (4,7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name':'sagA', 'function':'transport'})

        self.assertTrue(f._interval_metadata is not None)
        self.assertListEqual(f.intervals, [(1,2), (4,7)])
        self.assertListEqual(f.boundaries, [(True, False), (False, False)])
        self.assertDictEqual(f.metadata, {'name':'sagA', 'function':'transport'})

    def test_bad_constructor(self):
        with self.assertRaises(ValueError):
            f = Interval(_interval_metadata=IntervalMetadata(),
                         intervals=[1, (4,7)],
                         boundaries=[(True, False), (False, False)],
                         metadata={'name':'sagA', 'function':'transport'})

    def test_getitem(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1,2), (4,7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name':'sagA', 'function':'transport'})
        self.assertEqual(f['name'], 'sagA')
        self.assertEqual(f['function'], 'transport')

    def test_setitem(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1,2), (4,7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name':'sagA', 'function':'transport'})
        f['name'] = 'sagB'
        self.assertEqual(f['name'], 'sagB')
        self.assertEqual(f['function'], 'transport')

    def test_equal(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1,2), (4,7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name':'sagA', 'function':'transport'})

        f1 = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1,2), (4,7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name':'sagA', 'function':'transport'})

        f2 = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1,2), (4,8)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name':'sagA', 'function':'transport'})

        f3 = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1,2), (4,8)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name':'sagB', 'function':'transport'})
        self.assertEqual(f, f1)
        self.assertNotEqual(f, f2)
        self.assertNotEqual(f, f3)

    def test_set_interval(self):
        im = IntervalMetadata()
        f = Interval(_interval_metadata=im,
                     intervals=[(1,2), (4,7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name':'sagA', 'function':'transport'})
        f.intervals = [(1,3), (4,7)]
        self.assertEqual(f.intervals, [(1,3), (4,7)])
        self.assertEqual(im._is_stale_tree, True)


class TestIntervalMetadata(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        im = IntervalMetadata()
        self.assertEqual(im._is_stale_tree, False)

    def test_polish_interval_empty(self):
        res = _polish_interval(())
        self.assertTrue(res is None)

    def test_polish_interval_tuple(self):
        st, end = _polish_interval((1, 2))
        self.assertEqual(st, 1)
        self.assertEqual(end, 2)

    def test_add(self):
        im = IntervalMetadata()
        im.add(intervals=[(1, 2), (4, 7)],
               metadata={'gene':'sagA', 'location':0})

        self.assertEqual(im._metadata[0].intervals,
                         [(1, 2), (4, 7)])
        self.assertEqual(im._metadata[0].metadata,
                         {'gene':'sagA', 'location':0})
        self.assertTrue(im._intervals is not None)

    def test_add_empty(self):
        pass

    def test_query(self):
        im = IntervalMetadata()
        im.add(intervals=[(0, 2), (4, 7)], metadata={'gene': 'sagA', 'location': 0})
        im.add(intervals=[(3, 5)], metadata={'gene': 'sagB', 'location': 0})

        feats = im.query((1, 2))
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].metadata, {'gene': 'sagA', 'location': 0})

        feats = im.query(gene='sagB')
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].metadata, {'gene': 'sagB', 'location': 0})
        self.assertEqual(feats[0].intervals, [(3, 5)])

    def test_query_duplicate1(self):
        im = IntervalMetadata()
        im.add(metadata={'gene':'sagA', 'location':0}, intervals=[(0, 2), (4, 7)])
        im.add(metadata={'gene':'sagB', 'location':0}, intervals=[(3, 5)])

        feats = im.query((1, 5))
        self.assertEqual(len(feats), 3)
        self.assertEqual(feats[0].metadata, {'gene': 'sagA', 'location': 0})
        self.assertEqual(feats[0].intervals, [(0, 2), (4, 7)])
        self.assertEqual(feats[1].metadata, {'gene': 'sagB', 'location': 0})
        self.assertEqual(feats[1].intervals, [(3, 5)])
        self.assertEqual(feats[2].metadata, {'gene': 'sagA', 'location': 0})
        self.assertEqual(feats[2].intervals, [(0, 2), (4, 7)])

#     def test_reverse_complement(self):
#         interval_metadata = IntervalMetadata()
#         interval_metadata.add(Feature(gene='sagB', location=0), (3, 5))
#         iv = interval_metadata.reverse_complement(length=10)
#         feats = iv.query((5, 7))
#         self.assertEqual(feats, [Feature(gene='sagB', location=0)])

    def test_update(self):
        # Test to see if BoundFeatures can be updated
        pass

#     def test_eq(self):
#         interval_metadata1 = IntervalMetadata(features={
#                    Feature(gene='sagA', location='0'): [(0, 2), (4, 7)],
#                    Feature(gene='sagB', location='3'): [(3, 5)]
#                })

#         interval_metadata2 = IntervalMetadata(features={
#                    Feature(gene='sagA', location='0'): [(0, 2), (4, 7)],
#                    Feature(gene='sagB', location='3'): [(3, 5)]
#                })
#         self.assertTrue(interval_metadata1 == interval_metadata2)


if __name__ == '__main__':
    unittest.main()
