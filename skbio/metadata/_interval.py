# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from ._feature import Feature
from ._intersection import IntervalTree
from skbio.util._misc import merge_dicts


class Interval():
    '''Store the metadata of a sequence interval.

    It is implemented as frozendict and can be used similarly
    as built-in ``dict``.

    Parameters
    ----------
    intervals : list of tuple of ints
        List of tuples representing start and end coordinates.
    boundaries : list of tuple of bool
        List of tuples, representing the openness of each interval.
    metadata : dict
        Dictionary of attributes storing information of the feature
        such as `strand`, `gene_name` or `product`.
    _interval_metadata : object
        A reference to the `IntervalMetadata` object that this
        interval is associated to.
    '''
    def __init__(self, intervals=None, boundaries=None,
                 metadata=None, _interval_metadata=None):
        self._intervals = []
        for interval in intervals:
            inv = _polish_interval(interval)
            self._intervals.append(inv)
        self.boundaries = boundaries
        self.metadata = metadata
        self._interval_metadata = _interval_metadata

    def __getitem__(self, key):
        return self.metadata[key]

    def __setitem__(self, key, val):
        self.metadata[key] = val

    def __eq__(self, other):
        return ((self.metadata == other.metadata) and
                (self.intervals == other.intervals) and
                (self.boundaries == other.boundaries))

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def intervals(self):
        return self._intervals

    @intervals.setter
    def intervals(self, value):
        self._intervals = value
        self._interval_metadata._is_stale_tree = True


class IntervalMetadata():
    def __init__(self):
        # stores metadata for each feature
        self._metadata = []
        self._intervals = IntervalTree()
        self._is_stale_tree = False

    def reverse_complement(self, length):
        """ Reverse complements IntervalMetadata object.

        Parameters
        ----------
        length : int
            Largest end coordinate to perform reverse complement.
            This typically corresponds to the length of sequence.

        Returns
        -------
        IntervalMetadata
        """
        rvs_features = {}
        for k, v in self.features.items():
            xs = map(_polish_interval, v)
            rvs_features[k] = list(map(lambda x: (length-x[1], length-x[0]),
                                       xs))
        return IntervalMetadata(rvs_features)

    def add(self, intervals, boundaries=None, metadata=None):
        """ Adds a feature to the metadata object.

        Parameters
        ----------
        feature : skbio.sequence.feature
            The feature object being added.
        intervals : iterable of intervals
            A list of intervals associated with the feature

        """
        inv_md = Interval(_interval_metadata=self,
                          intervals=intervals,
                          boundaries=boundaries,
                          metadata=metadata)

        for loc in inv_md.intervals:
            if loc is not None:
                start, end = loc
                self._intervals.add(start, end, inv_md)

        self._metadata.append(inv_md)

    def _query_interval(self, interval):
        start, end = _polish_interval(interval)
        invs = self._intervals.find(start, end)
        return invs

    def _query_attribute(self, key, value):
        queries = []
        for inv in self._metadata:
            if inv[key] == value:
                queries.append(inv)
        return queries

    def query(self, *args, **kwargs):
        """ Looks up features that with intervals and keywords.

        Parameters
        ----------
        args : I1, I2, ...
            Iterable of tuples or Intervals
        kwargs : dict
            Keyword arguments of feature name and feature value, which can
            be passed to ``dict``.  This is used to specify the search
            parameters. If the `location` keyword is passed, then an interval
            lookup will be performed.

        Note
        ----
        There are two types of queries to perform
        1. Query by interval
        2. Query by key/val pair (i.e. gene=sagA)

        Returns
        -------
        list, Feature
            A list of features satisfying the search criteria.
        """
        invs = []

        # Find queries by interval
        for value in args:
            invs += self._query_interval(value)

        # Find queries by feature attribute
        for (key, value) in kwargs.items():
            invs += self._query_attribute(key, value)

        return invs

    def __eq__(self, other):
        # This doesn't look at the interval trees,
        # since the interval trees are strictly built
        # based on the features.
        sivs = list(map(sorted, self.features.values()))
        oivs = list(map(sorted, other.features.values()))

        equalIntervals = sorted(sivs) == sorted(oivs)
        equalFeatures = self.features.keys() == other.features.keys()

        return equalIntervals and equalFeatures


def _polish_interval(interval):
    if isinstance(interval, tuple):
        if len(interval) == 0:
            return None
        start, end = interval
        if (len(interval) != 2 or
            ((not isinstance(start, int)) or
             (not isinstance(end, int)))):
            raise ValueError("`start` and `end` aren't correctly specified")
    else:
        raise ValueError('The args must be associated with'
                         'a tuple when querying')
    return start, end
