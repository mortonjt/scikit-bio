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
#from skbio.metadata._interval import _polish_interval
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


# class TestIntervalMetadataMixin(unittest.TestCase):
#     def setUp(self):
#         pass

#     def test_init(self):
#         d = {Feature(gene='sagA', location=0): [(0, 2), (4, 7)],
#              Feature(gene='sagB', location=0): [(3, 5)]}
#         md = IntervalMetadata(features=d)
#         self.assertEquals(md.features, d)

#     def test_polish_interval_empty(self):
#         res = _polish_interval(())
#         self.assertTrue(res is None)

#     def test_polish_interval_tuple(self):
#         st, end = _polish_interval((1, 2))
#         self.assertEqual(st, 1)
#         self.assertEqual(end, 2)

#     def test_polish_interval_interval(self):
#         st, end = _polish_interval(Interval(1, 2))
#         self.assertEqual(st, 1)
#         self.assertEqual(end, 2)

#     def test_polish_interval_point(self):
#         st, end = _polish_interval(1)
#         self.assertEqual(st, 1)
#         self.assertEqual(end, 2)

#     def test_query(self):
#         interval_metadata = IntervalMetadata(features={
#                            Feature(gene='sagA', location=0): [(0, 2), (4, 7)],
#                            Feature(gene='sagB', location=0): [(3, 5)]
#                        })

#         feats = interval_metadata.query((1, 2))
#         self.assertEqual(feats, [Feature(gene='sagA', location=0)])

#         feats = interval_metadata.query(gene='sagB')
#         self.assertEqual(feats, [Feature(gene='sagB', location=0)])

#     def test_query_duplicate1(self):
#         interval_metadata = IntervalMetadata(features={
#                            Feature(gene='sagA', location=0): [(0, 2), (4, 7)],
#                            Feature(gene='sagB', location=0): [(3, 5)]
#                        })

#         feats = interval_metadata.query((1, 5))
#         self.assertEqual(set(feats),
#                          {Feature(gene='sagA', location=0),
#                           Feature(gene='sagB', location=0)})

#     def test_query_duplicate2(self):
#         interval_metadata = IntervalMetadata(features={
#                            Feature(gene='sagA', location=0): [(0, 2), (4, 7)],
#                            Feature(gene='sagB', location=0): [(3, 5)],
#                            Feature(gene='sagB', location=0): [(14, 29)]
#                        })
#         feats = interval_metadata.query(gene='sagB')
#         self.assertEqual(feats, [Feature(gene='sagB', location=0)])

#     def test_reverse_complement(self):
#         interval_metadata = IntervalMetadata()
#         interval_metadata.add(Feature(gene='sagB', location=0), (3, 5))
#         iv = interval_metadata.reverse_complement(length=10)
#         feats = iv.query((5, 7))
#         self.assertEqual(feats, [Feature(gene='sagB', location=0)])

#     def test_update(self):
#         # Test to see if BoundFeatures can be updated
#         pass

#     def test_add(self):
#         interval_metadata = IntervalMetadata()
#         interval_metadata.add(Feature(gene='sagA', location=0), 1, (4, 7))
#         interval_metadata.add(Feature(gene='sagB', location=0), (3, 5))

#         # Relies on the test_query method to work
#         feats = interval_metadata.query((1, 2))
#         self.assertEqual(feats, [Feature(gene='sagA', location=0)])

#         feats = interval_metadata.query(gene='sagB')
#         self.assertEqual(feats, [Feature(gene='sagB', location=0)])

#     def test_add_empty(self):
#         interval_metadata = IntervalMetadata()
#         interval_metadata.add(Feature(gene='sagA', location=0))
#         feats = interval_metadata.query(gene='sagA')
#         self.assertEqual(feats, [Feature(gene='sagA', location=0)])

#         interval_metadata.add(Feature(gene='sagB', location=0), ())
#         feats = interval_metadata.query(gene='sagB')
#         self.assertEqual(feats, [Feature(gene='sagB', location=0)])

#     def test_concat(self):
#         interval_metadata1 = IntervalMetadata(features={
#             Feature(gene='sagA', location='0'): [(0, 2), (4, 7)],
#             Feature(gene='sagB', location='3'): [(3, 5)]
#         })

#         interval_metadata2 = IntervalMetadata(features={
#                    Feature(gene='sagC', location='10'): [(10, 12), (24, 27)],
#                    Feature(gene='sagD', location='13'): [(13, 15)]
#                })
#         catted_metadata = interval_metadata1.concat(interval_metadata2)
#         self.assertEqual(catted_metadata.features, {
#                          Feature(gene='sagA', location='0'): [(0, 2), (4, 7)],
#                          Feature(gene='sagB', location='3'): [(3, 5)],
#                          Feature(gene='sagC', location='10'): [(10, 12),
#                                                                (24, 27)],
#                          Feature(gene='sagD', location='13'): [(13, 15)]})

#     def test_concat_inplace(self):
#         interval_metadata1 = IntervalMetadata(features={
#                    Feature(gene='sagA', location='0'): [(0, 2), (4, 7)],
#                    Feature(gene='sagB', location='3'): [(3, 5)]
#                })

#         interval_metadata2 = IntervalMetadata(features={
#                    Feature(gene='sagC', location='10'): [(10, 12), (24, 27)],
#                    Feature(gene='sagD', location='13'): [(13, 15)]
#                })
#         interval_metadata1.concat(interval_metadata2, inplace=True)
#         self.assertEqual(interval_metadata1.features, {
#                          Feature(gene='sagA', location='0'): [(0, 2),
#                                                               (4, 7)],
#                          Feature(gene='sagB', location='3'): [(3, 5)],
#                          Feature(gene='sagC', location='10'): [(10, 12),
#                                                                (24, 27)],
#                          Feature(gene='sagD', location='13'): [(13, 15)]})

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
