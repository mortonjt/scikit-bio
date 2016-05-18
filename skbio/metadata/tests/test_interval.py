# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import unittest

from skbio.metadata._interval import _polish_interval
from skbio.metadata._interval import Interval
from skbio.metadata import IntervalMetadata

class TestInterval(unittest.TestCase):
    def setUp(self):
        pass

    def test_constructor(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1, 2), (4, 7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name': 'sagA', 'function': 'transport'})

        self.assertTrue(f._interval_metadata is not None)
        self.assertListEqual(f.intervals, [(1, 2), (4, 7)])
        self.assertListEqual(f.boundaries, [(True, False), (False, False)])
        self.assertDictEqual(f.metadata, {'name': 'sagA',
                                          'function': 'transport'})

    def test_bad_constructor(self):
        with self.assertRaises(ValueError):
            f = Interval(_interval_metadata=IntervalMetadata(),
                         intervals=[1, (4, 7)],
                         boundaries=[(True, False), (False, False)],
                         metadata={'name': 'sagA', 'function': 'transport'})

    def test_repr(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1, 2), (4, 7)],
                     metadata={'name': 'sagA', 'function': 'transport'})
        exp1 = (r"Interval(intervals=[(1, 2), (4, 7)], "
                "metadata={'name': 'sagA', 'function': 'transport'})")
        res = repr(f)
        # because dictionaries are random
        self.assertTrue(res, exp1)

    def test_getitem(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1, 2), (4, 7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name': 'sagA', 'function': 'transport'})
        self.assertEqual(f['name'], 'sagA')
        self.assertEqual(f['function'], 'transport')

    def test_setitem(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1, 2), (4, 7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name': 'sagA', 'function': 'transport'})
        f['name'] = 'sagB'
        self.assertEqual(f['name'], 'sagB')
        self.assertEqual(f['function'], 'transport')

    def test_cmp(self):
        f1 = Interval(_interval_metadata=IntervalMetadata(),
                      intervals=[(1, 2)],
                      boundaries=[(True, False), (False, False)],
                      metadata={'name': 'sagA', 'function': 'transport'})
        f2 = Interval(_interval_metadata=IntervalMetadata(),
                      intervals=[(10, 20)],
                      boundaries=[(True, False), (False, False)],
                      metadata={'name': 'sagA', 'function': 'transport'})
        self.assertTrue(f1 < f2)
        self.assertTrue(f1 <= f2)
        self.assertFalse(f1 > f2)
        self.assertFalse(f1 >= f2)

    def test_equal(self):
        f = Interval(_interval_metadata=IntervalMetadata(),
                     intervals=[(1, 2), (4, 7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name': 'sagA', 'function': 'transport'})

        f1 = Interval(_interval_metadata=IntervalMetadata(),
                      intervals=[(1, 2), (4, 7)],
                      boundaries=[(True, False), (False, False)],
                      metadata={'name': 'sagA', 'function': 'transport'})

        f2 = Interval(_interval_metadata=IntervalMetadata(),
                      intervals=[(1, 2), (4, 8)],
                      boundaries=[(True, False), (False, False)],
                      metadata={'name': 'sagA', 'function': 'transport'})

        f3 = Interval(_interval_metadata=IntervalMetadata(),
                      intervals=[(1, 2), (4, 8)],
                      boundaries=[(True, False), (False, False)],
                      metadata={'name': 'sagB', 'function': 'transport'})
        self.assertEqual(f, f1)
        self.assertNotEqual(f, f2)
        self.assertNotEqual(f, f3)

    def test_set_interval(self):
        im = IntervalMetadata()
        f = Interval(_interval_metadata=im,
                     intervals=[(1, 2), (4, 7)],
                     boundaries=[(True, False), (False, False)],
                     metadata={'name': 'sagA', 'function': 'transport'})
        f.intervals = [(1, 3), (4, 7)]
        self.assertEqual(f.intervals, [(1, 3), (4, 7)])
        self.assertEqual(im._is_stale_tree, True)


class TestIntervalMetadata(unittest.TestCase):

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
               metadata={'gene': 'sagA',  'location': 0})

        self.assertEqual(im._metadata[0].intervals,
                         [(1, 2), (4, 7)])
        self.assertEqual(im._metadata[0].metadata,
                         {'gene': 'sagA', 'location': 0})
        self.assertTrue(im._intervals is not None)

    def test_query(self):
        im = IntervalMetadata()
        im.add(intervals=[(0, 2), (4, 7)],
               metadata={'gene': 'sagA', 'location': 0})
        im.add(intervals=[(3, 5)],
               metadata={'gene': 'sagB', 'location': 0})

        feats = im.query(intervals=[(1, 2)])
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].metadata, {'gene': 'sagA', 'location': 0})

        feats = im.query(metadata={'gene': 'sagB'})
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].metadata, {'gene': 'sagB', 'location': 0})
        self.assertEqual(feats[0].intervals, [(3, 5)])

    def test_query_duplicate1(self):
        im = IntervalMetadata()
        im.add(metadata={'gene': 'sagA', 'location': 0},
               intervals=[(0, 2), (4, 7)])
        im.add(metadata={'gene': 'sagB', 'location': 0},
               intervals=[(3, 5)])

        feats = im.query(intervals=[(1, 5)])
        self.assertEqual(len(feats), 3)
        self.assertEqual(feats[0].metadata, {'gene': 'sagA', 'location': 0})
        self.assertEqual(feats[0].intervals, [(0, 2), (4, 7)])
        self.assertEqual(feats[1].metadata, {'gene': 'sagB', 'location': 0})
        self.assertEqual(feats[1].intervals, [(3, 5)])
        self.assertEqual(feats[2].metadata, {'gene': 'sagA', 'location': 0})
        self.assertEqual(feats[2].intervals, [(0, 2), (4, 7)])

    def test_set_interval_interval(self):
        interval_metadata = IntervalMetadata()
        interval_metadata.add(intervals=[(0, 2), (4, 7)],
                              boundaries=None, metadata={'name': 'sagA'})
        interval_metadata.add(intervals=[(40, 70)],
                              boundaries=None, metadata={'name': 'sagA'})
        interval_metadata.add(intervals=[(3, 4)],
                              boundaries=None, metadata={'name': 'sagB'})
        feats = list(interval_metadata.query(intervals=[(1, 2)]))
        feats[0].intervals = [(1, 2)]
        feats = list(interval_metadata.query(intervals=[(1, 2)]))
        self.assertEqual(feats[0].intervals, [(1, 2)])

    def test_set_interval_attribute(self):
        interval_metadata = IntervalMetadata()
        interval_metadata.add(intervals=[(0, 2), (4, 7)],
                              boundaries=None, metadata={'name': 'sagA'})
        interval_metadata.add(intervals=[(40, 70)],
                              boundaries=None, metadata={'name': 'sagA'})
        interval_metadata.add(intervals=[(3, 4)],
                              boundaries=None, metadata={'name': 'sagB'})
        feats = list(interval_metadata.query(intervals=[(1, 2)]))
        feats[0]['name'] = 'sagC'
        feats = list(interval_metadata.query(intervals=[(1, 2)]))
        self.assertEqual(feats[0]['name'], 'sagC')

    def test_drop(self):
        interval_metadata = IntervalMetadata()
        interval_metadata.add(intervals=[(0, 2), (4, 7)],
                              boundaries=None, metadata={'name': 'sagA'})
        interval_metadata.add(intervals=[(40, 70)],
                              boundaries=None, metadata={'name': 'sagA'})
        interval_metadata.add(intervals=[(3, 4)],
                              boundaries=None, metadata={'name': 'sagB'})
        interval_metadata.drop(metadata={'name': 'sagA'})
        feats = list(interval_metadata.query(intervals=[(1, 2)]))
        self.assertEqual(len(feats), 0)

    def test_reverse_complement(self):
        interval_metadata = IntervalMetadata()
        interval_metadata.add(metadata={'gene': 'sagB', 'location': 0},
                              intervals=[(3, 5)])
        interval_metadata._reverse(length=10)
        feats = interval_metadata.query([(5, 7)])
        exp = Interval(intervals=[(5, 7)],
                       metadata={'gene': 'sagB', 'location': 0})
        self.assertEqual(feats[0], exp)

    def test_eq(self):
        interval_metadata1 = IntervalMetadata()
        interval_metadata1.add(metadata={'gene': 'sagA', 'location': '0'},
                               intervals=[(0, 2), (4, 7)])
        interval_metadata1.add(metadata={'gene': 'sagB', 'location': '3'},
                               intervals=[(3, 5)])

        interval_metadata2 = IntervalMetadata()
        interval_metadata2.add(metadata={'gene': 'sagA', 'location': '0'},
                               intervals=[(0, 2), (4, 7)])
        interval_metadata2.add(metadata={'gene': 'sagB', 'location': '3'},
                               intervals=[(3, 5)])

        interval_metadata3 = IntervalMetadata()
        interval_metadata3.add(metadata={'gene': 'sagA', 'location': '3'},
                               intervals=[(0, 2), (4, 7)])
        interval_metadata3.add(metadata={'gene': 'sagB', 'location': '3'},
                               intervals=[(3, 5)])

        # The ordering shouldn't matter
        interval_metadata4 = IntervalMetadata()
        interval_metadata4.add(metadata={'gene': 'sagB', 'location': '3'},
                               intervals=[(3, 5)])
        interval_metadata4.add(metadata={'gene': 'sagA', 'location': '0'},
                               intervals=[(0, 2), (4, 7)])

        self.assertEqual(interval_metadata1, interval_metadata2)
        self.assertNotEqual(interval_metadata1, interval_metadata3)
        self.assertEqual(interval_metadata1, interval_metadata4)


if __name__ == '__main__':
    unittest.main()
