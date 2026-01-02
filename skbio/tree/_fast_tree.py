# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

r"""Fast Tree Node Implementation using Balanced Parentheses.

This module provides a memory-efficient tree representation using the
balanced parentheses data structure. The ``FastTreeNode`` class provides
an API similar to ``TreeNode`` while offering significant memory and
performance benefits for large trees.

The key differences from ``TreeNode`` are:

1. The tree structure is immutable after construction
2. All tree operations are performed on a shared underlying data structure
3. Node lookups and tree traversals are more memory-efficient
4. LCA (lowest common ancestor) queries are very fast

"""

import numpy as np

from skbio._base import SkbioObject
from ._bp import BP, parse_newick, write_newick
from ._exception import NoLengthError, NoParentError, MissingNodeError


class FastTreeNode(SkbioObject):
    """Memory-efficient tree representation using balanced parentheses.

    This class provides a similar interface to ``TreeNode`` but uses a
    balanced parentheses representation internally for better memory
    efficiency and faster operations on large trees.

    Unlike ``TreeNode``, ``FastTreeNode`` trees are immutable after
    construction. Operations that would modify the tree return new
    trees instead.

    Parameters
    ----------
    bp : BP
        The balanced parentheses tree structure.
    pos : int, optional
        The position in the BP array for this node. Default is 0 (root).

    Attributes
    ----------
    name : str or None
        The name of this node.
    length : float or None
        The branch length from this node to its parent.

    See Also
    --------
    TreeNode
        The mutable tree implementation with more features.

    Notes
    -----
    The FastTreeNode is a view into a shared BP data structure. Multiple
    FastTreeNode instances can reference different nodes in the same tree
    without duplicating the underlying data.

    Examples
    --------
    >>> from skbio.tree import FastTreeNode
    >>> from io import StringIO
    >>> tree = FastTreeNode.read(StringIO("((a:1,b:2)c:3,d:4)root;"))
    >>> tree.name
    'root'
    >>> tree.count()
    5
    >>> tree.count(tips=True)
    3
    """

    default_write_format = 'newick'

    def __init__(self, bp=None, pos=0, name=None, length=None):
        if bp is None:
            # Create a simple single-node tree
            B = np.array([1, 0], dtype=np.uint8)
            names = np.array([name, None], dtype=object)
            length_val = length if length is not None else np.nan
            lengths = np.array([length_val, np.nan], dtype=np.float64)
            bp = BP(B, names=names, lengths=lengths)
            pos = 0

        self._bp = bp
        self._pos = pos

    @property
    def name(self):
        """Return the name of this node."""
        return self._bp.name(self._pos)

    @property
    def length(self):
        """Return the branch length from this node to its parent."""
        return self._bp.length(self._pos)

    def __repr__(self):
        """Return summary of the tree.

        Returns
        -------
        str
            A summary of this node and all descendants.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c, d)root;"))
        >>> repr(tree)
        '<FastTreeNode, name: root, internal node count: 1, tips count: 3>'
        """
        n_nodes = self._bp.subtree(self._pos)
        n_tips = self._count_tips_in_subtree()
        n_nontips = n_nodes - n_tips - 1  # exclude self
        name = self.name if self.name is not None else "unnamed"

        return "<%s, name: %s, internal node count: %d, tips count: %d>" % \
               (self.__class__.__name__, name, n_nontips, n_tips)

    def _count_tips_in_subtree(self):
        """Count tips in the subtree rooted at this node."""
        count = 0
        close = self._bp.close(self._pos)
        for i in range(self._pos, close + 1):
            if self._bp.B[i] and self._bp.isleaf(i):
                count += 1
        return count

    def __str__(self):
        """Return Newick string representation.

        Returns
        -------
        str
            A Newick representation of the subtree.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> str(tree)
        '((a,b)c);'
        """
        if self._pos == 0:
            return write_newick(self._bp)
        else:
            # Create a subtree BP and write it
            subtree_bp = self._extract_subtree_bp()
            return write_newick(subtree_bp)

    def _extract_subtree_bp(self):
        """Extract the BP for the subtree rooted at this node."""
        close = self._bp.close(self._pos)
        new_B = self._bp.B[self._pos:close + 1].copy()
        new_names = self._bp._names[self._pos:close + 1].copy()
        new_lengths = self._bp._lengths[self._pos:close + 1].copy()
        return BP(new_B, names=new_names, lengths=new_lengths)

    def __eq__(self, other):
        """Check equality based on position in same tree."""
        if not isinstance(other, FastTreeNode):
            return False
        return (self._bp is other._bp and self._pos == other._pos)

    def __hash__(self):
        """Hash based on position."""
        return hash((id(self._bp), self._pos))

    def is_tip(self):
        """Return True if the node is a tip (leaf).

        Returns
        -------
        bool
            True if the node has no children.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> tree.is_tip()
        False
        >>> tree.find('a').is_tip()
        True
        """
        return self._bp.isleaf(self._pos)

    def is_root(self):
        """Return True if the node is the root.

        Returns
        -------
        bool
            True if the node has no parent.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> tree.is_root()
        True
        >>> tree.find('a').is_root()
        False
        """
        return self._pos == 0

    def has_children(self):
        """Return True if the node has children.

        Returns
        -------
        bool
            True if the node has children.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> tree.has_children()
        True
        >>> tree.find('a').has_children()
        False
        """
        return not self.is_tip()

    @property
    def parent(self):
        """Return the parent node.

        Returns
        -------
        FastTreeNode or None
            The parent node, or None if this is the root.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> tree.find('a').parent.name
        'c'
        """
        parent_pos = self._bp.parent(self._pos)
        if parent_pos < 0:
            return None
        return FastTreeNode(self._bp, parent_pos)

    @property
    def children(self):
        """Return a list of child nodes.

        Returns
        -------
        list of FastTreeNode
            The child nodes.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> [c.name for c in tree.children]
        ['c']
        """
        children = []
        child_pos = self._bp.fchild(self._pos)
        while child_pos >= 0:
            children.append(FastTreeNode(self._bp, child_pos))
            child_pos = self._bp.nsibling(child_pos)
        return children

    def root(self):
        """Return the root of the tree.

        Returns
        -------
        FastTreeNode
            The root node.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c)root;"))
        >>> tree.find('a').root().name
        'root'
        """
        return FastTreeNode(self._bp, 0)

    def ancestors(self):
        """Return all ancestors back to the root.

        Returns
        -------
        list of FastTreeNode
            The path toward the root, excluding self.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c)root;"))
        >>> [n.name for n in tree.find('a').ancestors()]
        ['c', 'root']
        """
        result = []
        current = self.parent
        while current is not None:
            result.append(current)
            current = current.parent
        return result

    def siblings(self):
        """Return all sibling nodes.

        Returns
        -------
        list of FastTreeNode
            The sibling nodes.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b,c)d);"))
        >>> [n.name for n in tree.find('b').siblings()]
        ['a', 'c']
        """
        if self.is_root():
            return []

        parent = self.parent
        return [c for c in parent.children if c._pos != self._pos]

    def neighbors(self, ignore=None):
        """Return all connected nodes.

        Parameters
        ----------
        ignore : FastTreeNode, optional
            A node to ignore.

        Returns
        -------
        list of FastTreeNode
            All connected nodes.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c)root;"))
        >>> node_c = tree.find('c')
        >>> [n.name for n in node_c.neighbors()]
        ['a', 'b', 'root']
        """
        nodes = self.children + ([self.parent] if self.parent else [])
        if ignore is None:
            return nodes
        return [n for n in nodes if n != ignore]

    def depth(self):
        """Return the depth of this node.

        Returns
        -------
        int
            The number of ancestors (root has depth 0).

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c)root;"))
        >>> tree.depth()
        0
        >>> tree.find('a').depth()
        2
        """
        return int(self._bp.depth(self._pos))

    def height(self):
        """Return the height of the subtree.

        Returns
        -------
        int
            The maximum depth of any leaf relative to this node.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c)root;"))
        >>> tree.height()
        2
        >>> tree.find('a').height()
        0
        """
        return int(self._bp.height(self._pos))

    def count(self, tips=False):
        """Return the number of nodes in the subtree.

        Parameters
        ----------
        tips : bool, optional
            If True, only count tips. Default is False.

        Returns
        -------
        int
            The number of nodes or tips.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c,d)root;"))
        >>> tree.count()
        5
        >>> tree.count(tips=True)
        3
        """
        if tips:
            return int(self._count_tips_in_subtree())
        return int(self._bp.subtree(self._pos))

    def preorder(self, include_self=True):
        """Iterate over nodes in preorder.

        Parameters
        ----------
        include_self : bool, optional
            Include the initial node. Default is True.

        Yields
        ------
        FastTreeNode
            Nodes in preorder.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> [n.name for n in tree.preorder()]
        [None, 'c', 'a', 'b']
        """
        close = self._bp.close(self._pos)
        for i in range(self._pos, close + 1):
            if self._bp.B[i]:  # Opening parenthesis
                if include_self or i != self._pos:
                    yield FastTreeNode(self._bp, i)

    def postorder(self, include_self=True):
        """Iterate over nodes in postorder.

        Parameters
        ----------
        include_self : bool, optional
            Include the initial node. Default is True.

        Yields
        ------
        FastTreeNode
            Nodes in postorder.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> [n.name for n in tree.postorder()]
        ['a', 'b', 'c', None]
        """
        close = self._bp.close(self._pos)
        # Postorder: visit node when we see its closing parenthesis
        for i in range(self._pos, close + 1):
            if not self._bp.B[i]:  # Closing parenthesis
                # Find corresponding open
                open_pos = self._bp._bwdsearch(i, 0)
                if include_self or open_pos != self._pos:
                    yield FastTreeNode(self._bp, open_pos)

    def levelorder(self, include_self=True):
        """Iterate over nodes in level order (breadth-first).

        Parameters
        ----------
        include_self : bool, optional
            Include the initial node. Default is True.

        Yields
        ------
        FastTreeNode
            Nodes in level order.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c,(d,e)f);"))
        >>> [n.name for n in tree.levelorder()]
        [None, 'c', 'f', 'a', 'b', 'd', 'e']
        """
        queue = [self]
        while queue:
            node = queue.pop(0)
            if include_self or node._pos != self._pos:
                yield node
            queue.extend(node.children)

    def tips(self, include_self=False):
        """Iterate over tips in postorder.

        Parameters
        ----------
        include_self : bool, optional
            Include self if it's a tip. Default is False.

        Yields
        ------
        FastTreeNode
            Tip nodes.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c,(d,e)f);"))
        >>> [n.name for n in tree.tips()]
        ['a', 'b', 'd', 'e']
        """
        for node in self.postorder(include_self=include_self):
            if node.is_tip():
                yield node

    def non_tips(self, include_self=False):
        """Iterate over non-tip nodes in postorder.

        Parameters
        ----------
        include_self : bool, optional
            Include self if applicable. Default is False.

        Yields
        ------
        FastTreeNode
            Non-tip nodes.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c,(d,e)f);"))
        >>> [n.name for n in tree.non_tips()]
        ['c', 'f']
        """
        for node in self.postorder(include_self=include_self):
            if not node.is_tip():
                yield node

    def traverse(self, self_before=True, self_after=False, include_self=True):
        """Iterate over nodes depth-first.

        Parameters
        ----------
        self_before : bool, optional
            Yield nodes before their descendants. Default is True.
        self_after : bool, optional
            Yield nodes after their descendants. Default is False.
        include_self : bool, optional
            Include the initial node. Default is True.

        Yields
        ------
        FastTreeNode
            Traversed nodes.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> [n.name for n in tree.traverse()]
        [None, 'c', 'a', 'b']
        """
        if self_before:
            if self_after:
                return self._pre_and_postorder(include_self)
            else:
                return self.preorder(include_self)
        else:
            if self_after:
                return self.postorder(include_self)
            else:
                return self.tips(include_self)

    def _pre_and_postorder(self, include_self=True):
        """Iterate visiting each node before and after its descendants."""
        close = self._bp.close(self._pos)
        for i in range(self._pos, close + 1):
            if self._bp.B[i]:  # Opening
                if include_self or i != self._pos:
                    yield FastTreeNode(self._bp, i)
            else:  # Closing
                # Use O(1) lookup for matching open position
                open_pos = self._bp._open_idx[i]
                if not self._bp.isleaf(open_pos):
                    if include_self or open_pos != self._pos:
                        yield FastTreeNode(self._bp, open_pos)

    def find(self, name):
        """Find a node by name.

        Parameters
        ----------
        name : str or FastTreeNode
            The name of the node to find.

        Returns
        -------
        FastTreeNode
            The found node.

        Raises
        ------
        MissingNodeError
            If the node is not found.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> tree.find('a').name
        'a'
        """
        if isinstance(name, FastTreeNode):
            return name

        # Use O(1) name lookup cache
        positions = self._bp.find_by_name(name)
        if positions:
            return FastTreeNode(self._bp, positions[0])

        raise MissingNodeError(f"Node {name} is not in self")

    def find_all(self, name):
        """Find all nodes with a given name.

        Parameters
        ----------
        name : str or FastTreeNode
            The name of the nodes to find.

        Returns
        -------
        list of FastTreeNode
            The found nodes.

        Raises
        ------
        MissingNodeError
            If no nodes are found.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c,(d,e)c);"))
        >>> len(tree.find_all('c'))
        2
        """
        if isinstance(name, FastTreeNode):
            return [name]

        # Use O(1) name lookup cache
        positions = self._bp.find_by_name(name)
        if not positions:
            raise MissingNodeError(f"Node {name} is not in self")

        return [FastTreeNode(self._bp, pos) for pos in positions]

    def find_by_func(self, func):
        """Find all nodes matching a function.

        Parameters
        ----------
        func : callable
            A function that takes a FastTreeNode and returns bool.

        Yields
        ------
        FastTreeNode
            Nodes for which func returns True.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> list(tree.find_by_func(lambda x: x.is_tip()))
        ... # doctest: +ELLIPSIS
        [<FastTreeNode...>, <FastTreeNode...>]
        """
        for node in self.traverse(include_self=True):
            if func(node):
                yield node

    def lowest_common_ancestor(self, tipnames):
        """Find the lowest common ancestor of given tips.

        Parameters
        ----------
        tipnames : list of str or FastTreeNode
            The tips of interest.

        Returns
        -------
        FastTreeNode
            The lowest common ancestor.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c,(d,e)f)root;"))
        >>> tree.lowest_common_ancestor(['a', 'b']).name
        'c'
        >>> tree.lowest_common_ancestor(['a', 'e']).name
        'root'
        """
        if len(tipnames) == 1:
            return self.find(tipnames[0])

        tips = [self.find(name) for name in tipnames]
        positions = [t._pos for t in tips]

        # Use the BP lca function for efficiency
        lca_pos = positions[0]
        for pos in positions[1:]:
            lca_pos = self._bp.lca(lca_pos, pos)

        return FastTreeNode(self._bp, lca_pos)

    lca = lowest_common_ancestor

    def subset(self):
        """Return the set of tip names descending from this node.

        Returns
        -------
        frozenset
            The set of tip names.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> sorted(tree.subset())
        ['a', 'b']
        """
        return frozenset({tip.name for tip in self.tips(include_self=True)})

    def distance(self, other):
        """Return the distance between self and other.

        Parameters
        ----------
        other : FastTreeNode
            The target node.

        Returns
        -------
        float
            The distance between the two nodes.

        Raises
        ------
        NoLengthError
            If a node without length is encountered.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> newick = "((a:1,b:2)c:3,(d:4,e:5)f:6)root;"
        >>> tree = FastTreeNode.read(StringIO(newick))
        >>> tree.find('a').distance(tree.find('d'))
        14.0
        """
        if self == other:
            return 0.0

        lca = self.lowest_common_ancestor([self, other])
        dist = 0.0

        # Distance from self to LCA
        current = self
        while current != lca:
            if current.length is None:
                raise NoLengthError(f"No length on node {current.name}")
            dist += current.length
            current = current.parent

        # Distance from other to LCA
        current = other
        while current != lca:
            if current.length is None:
                raise NoLengthError(f"No length on node {current.name}")
            dist += current.length
            current = current.parent

        return float(dist)

    def accumulate_to_ancestor(self, ancestor):
        """Return the sum of distances between self and ancestor.

        Parameters
        ----------
        ancestor : FastTreeNode
            The ancestor node.

        Returns
        -------
        float
            The sum of lengths.

        Raises
        ------
        NoParentError
            If ancestor is not actually an ancestor.
        NoLengthError
            If a node without length is encountered.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a:1,b:2)c:3)root;"))
        >>> tree.find('a').accumulate_to_ancestor(tree)
        4.0
        """
        accum = 0.0
        current = self
        while current != ancestor:
            if current.is_root():
                raise NoParentError("Provided ancestor is not in the path")
            if current.length is None:
                raise NoLengthError(f"No length on node {current.name}")
            accum += current.length
            current = current.parent
        return float(accum)

    def shear(self, names):
        """Return a new tree with only the specified tips.

        Parameters
        ----------
        names : list of str
            The tip names to keep.

        Returns
        -------
        FastTreeNode
            A new tree containing only the specified tips.

        Raises
        ------
        ValueError
            If names are not a subset of the tree's tips.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c,(d,e)f)root;"))
        >>> sheared = tree.shear(['a', 'd'])
        >>> sorted([t.name for t in sheared.tips()])
        ['a', 'd']
        """
        all_tips = {t.name for t in self.tips()}
        names_set = set(names)

        if not names_set.issubset(all_tips):
            raise ValueError("names are not a subset of the tree's tips")

        # Find tip positions
        tip_positions = set()
        for node in self.tips():
            if node.name in names_set:
                tip_positions.add(node._pos)

        # Create sheared tree
        new_bp = self._bp.shear(tip_positions)

        # Collapse single-child nodes
        new_bp = new_bp.collapse()

        return FastTreeNode(new_bp)

    def copy(self):
        """Return a copy of the tree.

        Returns
        -------
        FastTreeNode
            A new tree with copied data.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> copy = tree.copy()
        >>> copy is tree
        False
        """
        new_bp = BP(
            self._bp.B.copy(),
            names=self._bp._names.copy(),
            lengths=self._bp._lengths.copy()
        )
        return FastTreeNode(new_bp, self._pos)

    def ascii_art(self, show_internal=True, compact=False):
        """Return ASCII art representation of the tree.

        Parameters
        ----------
        show_internal : bool, optional
            Show internal node names. Default is True.
        compact : bool, optional
            Use compact format. Default is False.

        Returns
        -------
        str
            ASCII art of the tree.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> print(tree.ascii_art())
        ... # doctest: +SKIP
        """
        return self._ascii_art(show_internal=show_internal,
                               compact=compact)[0]

    def _ascii_art(self, char1='-', show_internal=True, compact=False):
        """Generate ASCII art lines and midpoint."""
        LEN = 10
        PAD = ' ' * LEN
        PA = ' ' * (LEN - 1)
        namestr = self._node_label()

        if self.children:
            mids = []
            result = []
            for i, c in enumerate(self.children):
                if i == 0:
                    char2 = '/'
                elif i == len(self.children) - 1:
                    char2 = '\\'
                else:
                    char2 = '-'
                (clines, mid) = c._ascii_art(char2, show_internal, compact)
                mids.append(mid + len(result))
                result.extend(clines)
                if not compact:
                    result.append('')
            if not compact and result:
                result.pop()
            (lo, hi, end) = (mids[0], mids[-1], len(result))
            prefixes = [PAD] * (lo + 1) + [PA + '|'] * \
                (hi - lo - 1) + [PAD] * (end - hi)
            mid = int((lo + hi) / 2)
            prefixes[mid] = char1 + '-' * (LEN - 2) + prefixes[mid][-1]
            result = [p + l for (p, l) in zip(prefixes, result)]
            if show_internal:
                stem = result[mid]
                result[mid] = stem[0] + namestr + stem[len(namestr) + 1:]
            return ('\n'.join(result), mid)
        else:
            return (char1 + '-' + namestr, 0)

    def _node_label(self):
        """Return a label for this node."""
        name = self.name if self.name else ''
        return str(name)

    @classmethod
    def read(cls, fp, format='newick', **kwargs):
        """Read a tree from a file.

        Parameters
        ----------
        fp : file-like object or str
            The source to read from.
        format : str, optional
            The file format. Default is 'newick'.

        Returns
        -------
        FastTreeNode
            The parsed tree.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> tree.count()
        4
        """
        if format != 'newick':
            raise ValueError(f"Unsupported format: {format}")

        if hasattr(fp, 'read'):
            data = fp.read()
        else:
            with open(fp, 'r') as f:
                data = f.read()

        bp = parse_newick(data)
        return cls(bp)

    def write(self, fp, format='newick', **kwargs):
        """Write the tree to a file.

        Parameters
        ----------
        fp : file-like object or str
            The destination to write to.
        format : str, optional
            The file format. Default is 'newick'.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tree = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> out = StringIO()
        >>> tree.write(out)
        >>> out.getvalue()
        '((a,b)c);'
        """
        if format != 'newick':
            raise ValueError(f"Unsupported format: {format}")

        newick_str = str(self)

        if hasattr(fp, 'write'):
            fp.write(newick_str)
        else:
            with open(fp, 'w') as f:
                f.write(newick_str)

    @classmethod
    def from_tree_node(cls, tree_node):
        """Create a FastTreeNode from a TreeNode.

        Parameters
        ----------
        tree_node : TreeNode
            The source tree.

        Returns
        -------
        FastTreeNode
            A balanced parentheses representation.

        Examples
        --------
        >>> from skbio import TreeNode
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> tn = TreeNode.read(StringIO("((a,b)c);"))
        >>> ftn = FastTreeNode.from_tree_node(tn)
        >>> ftn.count()
        4
        """
        # Allocate arrays
        B = []
        names = []
        lengths = []

        # Traverse and build BP
        # Note: traverse with self_before=True, self_after=True visits internal
        # nodes twice (before and after children) but tips only once.
        for node in tree_node.traverse(self_before=True, self_after=True,
                                       include_self=True):
            if not hasattr(node, '_bp_visited'):
                # First visit - opening parenthesis
                node._bp_visited = True
                B.append(1)
                names.append(node.name)
                length = node.length if node.length is not None else np.nan
                lengths.append(length)

                # Tips are only visited once, so add closing paren immediately
                if node.is_tip():
                    B.append(0)
                    names.append(None)
                    lengths.append(np.nan)
            else:
                # Second visit - closing parenthesis (internal nodes only)
                B.append(0)
                names.append(None)
                lengths.append(np.nan)

        # Clean up
        for node in tree_node.traverse(include_self=True):
            if hasattr(node, '_bp_visited'):
                delattr(node, '_bp_visited')

        B = np.array(B, dtype=np.uint8)
        names = np.array(names, dtype=object)
        lengths = np.array(lengths, dtype=np.float64)

        bp = BP(B, names=names, lengths=lengths)
        return cls(bp)

    def to_tree_node(self):
        """Convert to a TreeNode.

        Returns
        -------
        TreeNode
            A mutable tree representation.

        Examples
        --------
        >>> from skbio.tree import FastTreeNode
        >>> from io import StringIO
        >>> ftn = FastTreeNode.read(StringIO("((a,b)c);"))
        >>> tn = ftn.to_tree_node()
        >>> type(tn).__name__
        'TreeNode'
        """
        from ._tree import TreeNode

        # Build TreeNode from BP
        nodes = {}
        stack = []

        for i in range(self._bp.size):
            if self._bp.B[i]:  # Opening
                name = self._bp.name(i)
                length = self._bp.length(i)
                node = TreeNode(name=name, length=length)
                nodes[i] = node
                if stack:
                    parent_pos = stack[-1]
                    nodes[parent_pos].append(node)
                stack.append(i)
            else:  # Closing
                stack.pop()

        return nodes[0]
