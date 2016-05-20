# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from ._intersection import IntervalTree


class Interval():
    '''Store the metadata of a sequence interval.

    Parameters
    ----------
    intervals : list of tuple of ints
        List of tuples representing start and end coordinates.
    boundaries : list of tuple of bool
        List of tuples, representing the openness of each interval.
    metadata : dict
        Dictionary of attributes storing information of the feature
        such as `strand`, `gene_name` or `product`.
    interval_metadata : object
        A reference to the `IntervalMetadata` object that this
        interval is associated to.
    '''
    def __init__(self, intervals=None, boundaries=None,
                 metadata=None, interval_metadata=None):
        iv = []
        for interval in intervals:
            iv.append(_polish_interval(interval))

        if boundaries is not None:
            self.boundaries = boundaries
        else:
            self.boundaries = []

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = {}

        self._interval_metadata = interval_metadata

        if intervals is not None:
            self.intervals = iv
        else:
            self.intervals = []

    def __getitem__(self, key):
        return self.metadata[key]

    def __setitem__(self, key, val):
        self.metadata[key] = val

    # This is required for creating unique sets of intervals
    def __hash__(self):
        return hash(tuple(sorted(self.metadata.items()) +
                          self.intervals +
                          self.boundaries))

    def __lt__(self, other):
        return self.intervals < other.intervals

    def __gt__(self, other):
        return self.intervals > other.intervals

    def __le__(self, other):
        return self.intervals <= other.intervals

    def __ge__(self, other):
        return self.intervals >= other.intervals

    def __eq__(self, other):
        return ((self.metadata == other.metadata) and
                (self.intervals == other.intervals) and
                (self.boundaries == other.boundaries))

    def __contains__(self, key):
        return key in self.metadata

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return ''.join([
            "Interval("
            "intervals=" + str(self.intervals),
            ", metadata=" + str(self.metadata),
            ")"])

    def drop(self):
        self._interval_metadata.drop(intervals=self.intervals,
                                     boundaries=self.boundaries,
                                     metadata=self.metadata)
        self.boundaries = None
        self.intervals = None
        self._interval_metadata = None

    @property
    def intervals(self):
        return self._intervals

    @intervals.setter
    def intervals(self, value):
        self._intervals = value
        if self._interval_metadata is not None:
            self._interval_metadata._is_stale_tree = True


class IntervalMetadata():
    def __init__(self):
        # stores metadata for each feature
        self._metadata = []
        self._intervals = IntervalTree()
        self._is_stale_tree = False

    def _reverse(self, length):
        """ Reverse complements IntervalMetadata object.

        Parameters
        ----------
        length : int
            Largest end coordinate to perform reverse complement.
            This typically corresponds to the length of sequence.
        """
        for f in self._metadata:
            # staled doesn't need to be called, since the setter for
            # Interval will take care of this
            f.intervals = list(map(lambda x: (length-x[1], length-x[0]),
                                   f.intervals))

    def add(self, intervals, boundaries=None, metadata=None):
        """ Adds a feature to the metadata object.

        Parameters
        ----------
        intervals : iterable of tuple of ints
            A list of intervals associated with the Interval object.
        boundaries : iterable of tuple of bool
            A list of boundaries associated with the Interval object.
        metadata : dict
            A dictionary of key word attributes associated with the
            Interval object.
        Examples
        --------
        >>> from skbio.metadata import IntervalMetadata
        >>> interval_metadata = IntervalMetadata()
        >>> interval_metadata.add(intervals=[(0, 2), (4, 7)],
        ...                       boundaries=None, metadata={'name': 'sagA'})
        >>> interval_metadata.add(intervals=[(40, 70)],
        ...                       boundaries=None, metadata={'name': 'sagB'})
        >>> interval_metadata.query(intervals=[(1, 2)])
        [Interval(intervals=[(0, 2), (4, 7)], metadata={'name': 'sagA'})]

        """
        inv_md = Interval(interval_metadata=self,
                          intervals=intervals,
                          boundaries=boundaries,
                          metadata=metadata)

        # Add directly to the tree.  So no need for _is_stale_tree
        for loc in inv_md.intervals:
            if loc is not None:
                start, end = loc
                self._intervals.add(start, end, inv_md)

        self._metadata.append(inv_md)

    def _rebuild_tree(self, intervals):
        self._intervals = IntervalTree()
        for f in intervals:
            for inv in f.intervals:
                start, end = inv
                self._intervals.add(start, end, f)

    def _query_interval(self, interval):
        start, end = _polish_interval(interval)
        invs = self._intervals.find(start, end)
        return invs

    def _query_attribute(self, intervals, metadata):
        if metadata is None:
            return []

        queries = []
        for inv in intervals:
            for (key, value) in metadata.items():
                if inv[key] != value:
                    continue
                queries.append(inv)
        return queries

    def query(self, intervals=None, boundaries=None, metadata=None):
        """ Looks up Interval objects with the intervals, boundaries and keywords.

        Parameters
        ----------
        intervals : iterable of tuple of ints
            A list of intervals associated with the Interval object.
        boundaries : iterable of tuple of bool
            A list of boundaries associated with the Interval object.
        metadata : dict
            A dictionary of key word attributes associated with the
            Interval object.

        Returns
        -------
        list, Interval
            A list of Intervals satisfying the search criteria.

        Examples
        --------
        >>> from skbio.metadata import IntervalMetadata
        >>> interval_metadata = IntervalMetadata()
        >>> interval_metadata.add(intervals=[(0, 2), (4, 7)],
        ...                       boundaries=None, metadata={'name': 'sagA'})
        >>> interval_metadata.add(intervals=[(40, 70)],
        ...                       boundaries=None, metadata={'name': 'sagB'})
        >>> interval_metadata.query(intervals=[(1, 2)])
        [Interval(intervals=[(0, 2), (4, 7)], metadata={'name': 'sagA'})]

        Note
        ----
        There are two types of queries to perform
        1. Query by interval.
        2. Query by key/val pair (i.e. gene=sagA).

        """
        if self._is_stale_tree:
            self._rebuild_tree(self._metadata)
            self._is_stale_tree = False

        invs = set()

        # Find queries by interval
        if intervals is not None:
            for value in intervals:
                invs.update(self._query_interval(value))

        # Find queries by feature attribute
        if len(invs) == 0 and metadata is not None:
            invs = set(self._metadata)

        if metadata is not None:
            invs = self._query_attribute(list(invs), metadata)
        return list(invs)

    def drop(self, intervals=None, boundaries=None, metadata=None):
        """ Drops Interval objects according to a specified query.

        Drops all Interval objects that matches the query.

        Parameters
        ----------
        intervals : iterable of tuple of ints
            A list of intervals associated with the Interval object.
        boundaries : iterable of tuple of bool
            A list of boundaries associated with the Interval object.
        metadata : dict
            A dictionary of key word attributes associated with the
            Interval object.

        Examples
        --------
        >>> from skbio.metadata import IntervalMetadata
        >>> interval_metadata = IntervalMetadata()
        >>> interval_metadata.add(intervals=[(0, 2), (4, 7)],
        ...                       boundaries=None, metadata={'name': 'sagA'})
        >>> interval_metadata.add(intervals=[(40, 70)],
        ...                       boundaries=None, metadata={'name': 'sagB'})
        >>> interval_metadata.drop(metadata={'name': 'sagA'})
        >>> interval_metadata.query(metadata={'name': 'sagA'})
        []
        """
        if intervals is None:
            intervals = []
        if metadata is None:
            metadata = {}

        queried_invs = self.query(intervals=intervals,
                                  boundaries=boundaries,
                                  metadata=metadata)
        new_invs = []
        # iterate through queries and drop them
        for inv in self._metadata:
            if inv not in queried_invs:
                new_invs.append(inv)
        self._metadata = new_invs
        self._is_stale_tree = True

    def __eq__(self, other):
        return sorted(self._metadata) == sorted(other._metadata)

    def __ne__(self, other):
        return not self.__eq__(other)


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
        raise ValueError('The args must be associated with '
                         'a tuple when querying')
    return start, end
