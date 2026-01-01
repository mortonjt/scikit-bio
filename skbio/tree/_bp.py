# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

r"""Balanced Parentheses Tree Representation.

This module provides a memory-efficient tree representation using balanced
parentheses. The implementation is based on the succinct data structure
described by Cordova and Navarro.

In a balanced parentheses representation, each node is represented by a pair
of parentheses: an opening parenthesis '(' (represented as 1) and a closing
parenthesis ')' (represented as 0). The tree structure is encoded by the
nesting of these parentheses.

For example, the tree::

         root
        /    \
       a      b
      / \
     c   d

Would be represented as: ((()()) ()) which in binary is: 1 1 1 0 1 0 0 1 0 0

This representation enables O(1) or O(log n) time complexity for many tree
operations while using only 2n + o(n) bits for an n-node tree.
"""

import numpy as np


def _build_rmm_tree(B, block_size=None):
    """Build a range min-max tree for efficient excess queries.

    Parameters
    ----------
    B : np.ndarray of uint8
        The balanced parentheses array (1 for open, 0 for close).
    block_size : int, optional
        The size of blocks for the RMM tree. If None, defaults to
        max(1, int(np.log2(len(B) + 1))).

    Returns
    -------
    dict
        A dictionary containing:
        - 'excess': Cumulative excess at each block boundary
        - 'min_excess': Minimum excess within each block
        - 'max_excess': Maximum excess within each block
        - 'block_size': The block size used
    """
    n = len(B)
    if block_size is None:
        block_size = max(1, int(np.log2(n + 1)))

    num_blocks = (n + block_size - 1) // block_size

    # Compute cumulative excess (number of opens minus closes)
    excess = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        excess[i + 1] = excess[i] + (1 if B[i] else -1)

    # Compute block-level statistics
    block_excess = np.zeros(num_blocks + 1, dtype=np.int32)
    block_min = np.zeros(num_blocks, dtype=np.int32)
    block_max = np.zeros(num_blocks, dtype=np.int32)

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, n)
        block_excess[b + 1] = excess[end]

        # Min and max excess within this block
        block_min[b] = np.min(excess[start:end + 1])
        block_max[b] = np.max(excess[start:end + 1])

    return {
        'excess': excess,
        'block_excess': block_excess,
        'min_excess': block_min,
        'max_excess': block_max,
        'block_size': block_size,
        'num_blocks': num_blocks
    }


class BP:
    """Balanced Parentheses succinct tree representation.

    This class implements a memory-efficient tree representation using
    balanced parentheses. Each node is represented by a pair of bits:
    1 for opening parenthesis (entering node) and 0 for closing
    parenthesis (leaving node).

    Parameters
    ----------
    B : np.ndarray of bool or uint8
        The balanced parentheses topology array where 1 represents
        an opening parenthesis and 0 represents a closing parenthesis.
    names : np.ndarray of object, optional
        Node names indexed by node position. Default is None.
    lengths : np.ndarray of float64, optional
        Branch lengths indexed by node position. Default is None.

    Attributes
    ----------
    B : np.ndarray
        The balanced parentheses array.
    size : int
        The size of the array (2 * number of nodes).

    Notes
    -----
    The balanced parentheses representation enables O(1) or O(log n)
    time complexity for operations like parent, first child, next sibling,
    and lowest common ancestor queries.

    See Also
    --------
    FastTreeNode
        High-level tree interface using this representation.

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.tree._bp import BP
    >>> # Simple tree: ((a, b)c)root
    >>> B = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.uint8)
    >>> names = np.array(['root', 'c', 'a', None, 'b', None,
    ...                   None, None, None, None], dtype=object)
    >>> bp = BP(B, names=names)
    >>> bp.ntips()
    2
    """

    def __init__(self, B, names=None, lengths=None):
        # Ensure B is the correct type
        self._B = np.asarray(B, dtype=np.uint8)
        self._size = len(self._B)

        # Initialize names array
        if names is not None:
            self._names = np.asarray(names, dtype=object)
        else:
            self._names = np.empty(self._size, dtype=object)
            self._names.fill(None)

        # Initialize lengths array
        if lengths is not None:
            self._lengths = np.asarray(lengths, dtype=np.float64)
        else:
            self._lengths = np.empty(self._size, dtype=np.float64)
            self._lengths.fill(np.nan)

        # Build the RMM tree for efficient queries
        self._rmm = _build_rmm_tree(self._B)

        # Pre-compute excess array reference
        self._excess = self._rmm['excess']

        # Build lookup indices for efficient operations
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices for close operations."""
        # Map from open position to close position
        n = self._size
        self._close_idx = np.zeros(n, dtype=np.intp)

        stack = []
        for i in range(n):
            if self._B[i]:  # Opening parenthesis
                stack.append(i)
            else:  # Closing parenthesis
                if stack:
                    open_idx = stack.pop()
                    self._close_idx[open_idx] = i

    @property
    def B(self):
        """Return the balanced parentheses array."""
        return self._B

    @property
    def size(self):
        """Return the size of the BP array."""
        return self._size

    def name(self, i):
        """Return the name of the node at position i.

        Parameters
        ----------
        i : int
            The position in the BP array (must be an opening parenthesis).

        Returns
        -------
        str or None
            The name of the node at position i.
        """
        return self._names[i]

    def length(self, i):
        """Return the branch length of the node at position i.

        Parameters
        ----------
        i : int
            The position in the BP array (must be an opening parenthesis).

        Returns
        -------
        float or None
            The branch length of the node at position i.
        """
        length = self._lengths[i]
        if np.isnan(length):
            return None
        return length

    def set_names(self, names):
        """Set the names array.

        Parameters
        ----------
        names : np.ndarray of object
            Array of node names.
        """
        self._names = np.asarray(names, dtype=object)

    def set_lengths(self, lengths):
        """Set the lengths array.

        Parameters
        ----------
        lengths : np.ndarray of float64
            Array of branch lengths.
        """
        self._lengths = np.asarray(lengths, dtype=np.float64)

    def excess(self, i):
        """Return the excess at position i.

        The excess is the number of opening parentheses minus the number
        of closing parentheses from position 0 to i (inclusive).

        Parameters
        ----------
        i : int
            The position in the BP array.

        Returns
        -------
        int
            The excess at position i.
        """
        return self._excess[i + 1]

    def rank(self, t, i):
        """Return the number of occurrences of t in B[0:i].

        Parameters
        ----------
        t : int
            The symbol to count (0 or 1).
        i : int
            The position up to which to count.

        Returns
        -------
        int
            The number of occurrences of t in B[0:i].
        """
        if i <= 0:
            return 0
        count = np.sum(self._B[:i] == t)
        return int(count)

    def select(self, t, k):
        """Return the position of the k-th occurrence of t in B.

        Parameters
        ----------
        t : int
            The symbol to find (0 or 1).
        k : int
            The occurrence to find (1-indexed).

        Returns
        -------
        int
            The position of the k-th occurrence of t.
        """
        if k <= 0:
            return -1
        count = 0
        for i in range(self._size):
            if self._B[i] == t:
                count += 1
                if count == k:
                    return i
        return -1

    def open(self, i):
        """Check if position i is an opening parenthesis.

        Parameters
        ----------
        i : int
            The position to check.

        Returns
        -------
        bool
            True if position i is an opening parenthesis.
        """
        return bool(self._B[i])

    def close(self, i):
        """Return the position of the closing parenthesis for open at i.

        Parameters
        ----------
        i : int
            The position of an opening parenthesis.

        Returns
        -------
        int
            The position of the matching closing parenthesis.
        """
        if not self._B[i]:
            return i
        return self._close_idx[i]

    def _fwdsearch(self, i, d):
        """Forward search for excess at position i.

        Find the smallest j > i such that excess(j) = excess(i) + d.

        Parameters
        ----------
        i : int
            Starting position.
        d : int
            Target excess difference.

        Returns
        -------
        int
            Position j where excess matches, or -1 if not found.
        """
        target = self._excess[i + 1] + d
        for j in range(i + 1, self._size + 1):
            if self._excess[j] == target:
                return j - 1
        return -1

    def _bwdsearch(self, i, d):
        """Backward search for excess at position i.

        Find the largest j < i such that excess(j) = excess(i) + d.

        Parameters
        ----------
        i : int
            Starting position.
        d : int
            Target excess difference.

        Returns
        -------
        int
            Position j where excess matches, or -1 if not found.
        """
        target = self._excess[i + 1] + d
        for j in range(i, -1, -1):
            if self._excess[j] == target:
                return j
        return -1

    def enclose(self, i):
        """Return the opening position of the tightest enclosing node.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            Position of the enclosing open parenthesis, or -1 if root.
        """
        if not self._B[i]:
            raise ValueError("Position must be an opening parenthesis")
        if i == 0:
            return -1  # Root has no enclosing node
        result = self._bwdsearch(i - 1, -2)
        if result == -1:
            return -1
        return result

    def parent(self, i):
        """Return the position of the parent of node at position i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            Position of the parent's opening parenthesis, or -1 if root.
        """
        if i == 0:
            return -1  # Root has no parent

        if self._B[i]:
            # Opening parenthesis - find enclosing
            return self.enclose(i)
        else:
            # Closing parenthesis - find parent of corresponding open
            pass

        return self.enclose(i)

    def isleaf(self, i):
        """Check if node at position i is a leaf.

        A node is a leaf if its opening parenthesis is immediately
        followed by its closing parenthesis.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        bool
            True if the node is a leaf.
        """
        if not self._B[i]:
            return False  # Not an opening parenthesis
        return not self._B[i + 1]

    def fchild(self, i):
        """Return the position of the first child of node at position i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            Position of the first child's opening parenthesis, or -1 if leaf.
        """
        if self.isleaf(i):
            return -1
        return i + 1

    def lchild(self, i):
        """Return the position of the last child of node at position i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            Position of the last child's opening parenthesis, or -1 if leaf.
        """
        if self.isleaf(i):
            return -1
        # Find the opening position before the closing of i
        close_i = self.close(i)
        # The last child is at the position just after the last close
        # before close_i
        j = close_i - 1
        # Back up to find the matching open
        while j >= i + 1 and not self._B[j]:
            j = self._bwdsearch(j, 0)
        if j < i + 1:
            return i + 1  # Only one child
        # Find the open that closes at j
        return self._bwdsearch(j - 1, -1) + 1

    def nsibling(self, i):
        """Return the position of the next sibling of node at position i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            Position of the next sibling's opening parenthesis, or -1 if none.
        """
        close_i = self.close(i)
        if close_i + 1 >= self._size:
            return -1
        if self._B[close_i + 1]:
            return close_i + 1
        return -1

    def psibling(self, i):
        """Return the position of the previous sibling of node at position i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            Position of previous sibling's opening parenthesis, or -1 if none.
        """
        if i == 0:
            return -1
        if not self._B[i - 1]:
            # Previous position is a close, find its matching open
            j = self._bwdsearch(i - 1, 0)
            if j >= 0:
                return j
        return -1

    def preorder(self, i):
        """Return the preorder rank of node at position i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            The preorder rank (1-indexed).
        """
        return self.rank(1, i + 1)

    def preorderselect(self, k):
        """Return the position of the k-th node in preorder.

        Parameters
        ----------
        k : int
            The preorder rank (1-indexed).

        Returns
        -------
        int
            Position of the k-th node in preorder.
        """
        return self.select(1, k)

    def postorder(self, i):
        """Return the postorder rank of node at position i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            The postorder rank (1-indexed).
        """
        close_i = self.close(i)
        return self.rank(0, close_i + 1)

    def postorderselect(self, k):
        """Return the position of the k-th node in postorder.

        Parameters
        ----------
        k : int
            The postorder rank (1-indexed).

        Returns
        -------
        int
            Position of the k-th node in postorder.
        """
        close_pos = self.select(0, k)
        if close_pos == -1:
            return -1
        # Find matching open
        return self._bwdsearch(close_pos, 0)

    def depth(self, i):
        """Return the depth of node at position i.

        The depth is the number of ancestors (root has depth 0).

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            The depth of the node.
        """
        return self._excess[i + 1] - 1

    def height(self, i):
        """Return the height of the subtree rooted at node i.

        The height is the maximum depth of any leaf in the subtree
        relative to the node.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            The height of the subtree.
        """
        if self.isleaf(i):
            return 0
        close_i = self.close(i)
        base_excess = self._excess[i + 1]
        max_excess = np.max(self._excess[i + 1:close_i + 1])
        return max_excess - base_excess

    def subtree(self, i):
        """Return the size of the subtree rooted at node i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            The number of nodes in the subtree.
        """
        close_i = self.close(i)
        return (close_i - i + 1) // 2

    def isancestor(self, i, j):
        """Check if node i is an ancestor of node j.

        Parameters
        ----------
        i : int
            Position of a potential ancestor.
        j : int
            Position of a potential descendant.

        Returns
        -------
        bool
            True if i is an ancestor of j.
        """
        return i <= j <= self.close(i)

    def lca(self, i, j):
        """Return the lowest common ancestor of nodes i and j.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.
        j : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            Position of the LCA's opening parenthesis.
        """
        if i == j:
            return i
        if i > j:
            i, j = j, i

        # Find the minimum excess in the range [i, j]
        close_i = self.close(i)
        if close_i >= j:
            # i is an ancestor of j
            return i

        # Find the enclosing node of the minimum
        min_excess = np.min(self._excess[i + 1:j + 1])
        # Find position of minimum
        for k in range(i, j + 1):
            if self._excess[k + 1] == min_excess:
                min_pos = k
                break

        # The LCA is the enclosing of the position just after the minimum
        return self._bwdsearch(min_pos, -1)

    def levelancestor(self, i, d):
        """Return the ancestor of node i at depth d.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.
        d : int
            Target depth.

        Returns
        -------
        int
            Position of the ancestor at depth d, or -1 if not found.
        """
        current_depth = self.depth(i)
        if d > current_depth:
            return -1
        if d == current_depth:
            return i

        # Go up (d - current_depth) levels
        target_excess = d + 1  # depth = excess - 1
        return self._bwdsearch(i, target_excess - self._excess[i + 1])

    def deepestnode(self, i):
        """Return the deepest node in the subtree rooted at i.

        Parameters
        ----------
        i : int
            Position of an opening parenthesis.

        Returns
        -------
        int
            Position of the deepest node.
        """
        close_i = self.close(i)
        max_excess = self._excess[i + 1]
        max_pos = i
        for k in range(i + 1, close_i + 1):
            if self._excess[k] > max_excess:
                max_excess = self._excess[k]
                max_pos = k - 1

        return max_pos

    def ntips(self):
        """Return the number of tips (leaves) in the tree.

        Returns
        -------
        int
            The number of tips.
        """
        count = 0
        for i in range(self._size - 1):
            if self._B[i] == 1 and self._B[i + 1] == 0:
                count += 1
        return count

    def numnodes(self):
        """Return the total number of nodes in the tree.

        Returns
        -------
        int
            The total number of nodes.
        """
        return self._size // 2

    def root(self):
        """Return the position of the root.

        Returns
        -------
        int
            The position of the root (always 0).
        """
        return 0

    def shear(self, tip_indices):
        """Return a new BP tree containing only the specified tips.

        Parameters
        ----------
        tip_indices : set or list
            The positions of tips to keep (as opening parenthesis positions).

        Returns
        -------
        BP
            A new BP tree containing only the specified tips and their
            ancestors.
        """
        tip_indices = set(tip_indices)
        if not tip_indices:
            raise ValueError("tip_indices cannot be empty")

        # Mark all nodes to keep
        keep = set()
        for tip_idx in tip_indices:
            if not self.isleaf(tip_idx):
                raise ValueError(f"Position {tip_idx} is not a leaf")
            # Add this tip and all ancestors
            current = tip_idx
            while current >= 0:
                if current in keep:
                    break
                keep.add(current)
                current = self.parent(current)

        # Build new topology
        new_B = []
        new_names = []
        new_lengths = []

        # Stack for tracking open parentheses
        for i in range(self._size):
            if self._B[i]:  # Opening
                if i in keep:
                    new_B.append(1)
                    new_names.append(self._names[i])
                    new_lengths.append(self._lengths[i])
            else:  # Closing
                # Find corresponding open
                # Count back to find the open position
                excess = 0
                for j in range(i, -1, -1):
                    if self._B[j]:
                        excess += 1
                    else:
                        excess -= 1
                    if excess == 0:
                        open_pos = j
                        break
                if open_pos in keep:
                    new_B.append(0)
                    new_names.append(None)
                    new_lengths.append(np.nan)

        new_B = np.array(new_B, dtype=np.uint8)
        new_names = np.array(new_names, dtype=object)
        new_lengths = np.array(new_lengths, dtype=np.float64)

        return BP(new_B, names=new_names, lengths=new_lengths)

    def collapse(self):
        """Return a new BP tree with single-child nodes removed.

        Returns
        -------
        BP
            A new BP tree with single-child nodes collapsed.
        """
        # Identify single-child internal nodes
        single_child = set()
        for i in range(self._size):
            if self._B[i] and not self.isleaf(i):
                # Check if single child
                first = self.fchild(i)
                if first >= 0 and self.nsibling(first) == -1:
                    single_child.add(i)

        if not single_child:
            # Return a copy
            return BP(self._B.copy(), names=self._names.copy(),
                      lengths=self._lengths.copy())

        # Build new topology excluding single-child nodes
        new_B = []
        new_names = []
        new_lengths = []

        for i in range(self._size):
            if self._B[i]:  # Opening
                if i not in single_child:
                    new_B.append(1)
                    new_names.append(self._names[i])
                    length = self._lengths[i]
                    # Add parent's length if parent is single-child
                    parent = self.parent(i)
                    while parent in single_child:
                        parent_len = self._lengths[parent]
                        if not np.isnan(parent_len):
                            if np.isnan(length):
                                length = parent_len
                            else:
                                length += parent_len
                        parent = self.parent(parent)
                    new_lengths.append(length)
            else:  # Closing
                # Find corresponding open
                excess = 0
                for j in range(i, -1, -1):
                    if self._B[j]:
                        excess += 1
                    else:
                        excess -= 1
                    if excess == 0:
                        open_pos = j
                        break
                if open_pos not in single_child:
                    new_B.append(0)
                    new_names.append(None)
                    new_lengths.append(np.nan)

        new_B = np.array(new_B, dtype=np.uint8)
        new_names = np.array(new_names, dtype=object)
        new_lengths = np.array(new_lengths, dtype=np.float64)

        return BP(new_B, names=new_names, lengths=new_lengths)

    def __repr__(self):
        return (f"<BP, nodes: {self.numnodes()}, tips: {self.ntips()}>")


def parse_newick(data):
    """Parse a Newick string into a BP tree.

    Parameters
    ----------
    data : str
        A Newick format string representing a tree.

    Returns
    -------
    BP
        A balanced parentheses tree representation.

    Examples
    --------
    >>> from skbio.tree._bp import parse_newick
    >>> bp = parse_newick("((a:1,b:2)c:3,d:4)root;")
    >>> bp.ntips()
    3
    """
    data = data.strip()
    if not data.endswith(';'):
        raise ValueError("Newick string must end with ';'")

    data = data[:-1].strip()  # Remove trailing semicolon

    # Parse the Newick string
    B = []
    names = []
    lengths = []
    stack = []  # Stack of (position, has_children)

    i = 0
    while i < len(data):
        char = data[i]

        if char == '(':
            # Opening parenthesis - start of internal node
            B.append(1)
            names.append(None)
            lengths.append(np.nan)
            stack.append((len(B) - 1, True))
            i += 1

        elif char == ')':
            # Closing parenthesis - end of internal node
            # Parse the name/length that follows
            i += 1
            name, length, i = _parse_node_info(data, i)
            if stack:
                pos, _ = stack.pop()
                names[pos] = name
                lengths[pos] = length if length is not None else np.nan
            B.append(0)
            names.append(None)
            lengths.append(np.nan)

        elif char == ',':
            i += 1

        elif char in ' \t\n':
            i += 1

        else:
            # Leaf node
            name, length, i = _parse_node_info(data, i)
            B.append(1)
            names.append(name)
            lengths.append(length if length is not None else np.nan)
            B.append(0)
            names.append(None)
            lengths.append(np.nan)

    # Handle root name if the tree was just a name at the root level
    if len(stack) == 0 and len(B) == 0:
        # Just a root node name
        name, length, _ = _parse_node_info(data, 0)
        B = [1, 0]
        names = [name, None]
        lengths = [length if length is not None else np.nan, np.nan]

    B = np.array(B, dtype=np.uint8)
    names = np.array(names, dtype=object)
    lengths = np.array(lengths, dtype=np.float64)

    return BP(B, names=names, lengths=lengths)


def _parse_node_info(data, i):
    """Parse name and length from Newick string starting at position i.

    Parameters
    ----------
    data : str
        The Newick string.
    i : int
        The starting position.

    Returns
    -------
    tuple
        (name, length, new_position)
    """
    name = None
    length = None
    n = len(data)

    # Skip whitespace
    while i < n and data[i] in ' \t\n':
        i += 1

    if i >= n:
        return name, length, i

    # Check for quoted name
    if data[i] == "'":
        i += 1
        start = i
        while i < n and data[i] != "'":
            i += 1
        name = data[start:i]
        if i < n and data[i] == "'":
            i += 1
    else:
        # Unquoted name
        start = i
        while i < n and data[i] not in '():,;':
            if data[i] == ':':
                break
            i += 1
        name = data[start:i].strip()
        if not name:
            name = None

    # Skip whitespace
    while i < n and data[i] in ' \t\n':
        i += 1

    # Check for length
    if i < n and data[i] == ':':
        i += 1
        start = i
        while i < n and data[i] not in '():,;':
            i += 1
        try:
            length = float(data[start:i].strip())
        except ValueError:
            length = None

    return name, length, i


def write_newick(bp, include_lengths=True):
    """Write a BP tree to Newick format string.

    Parameters
    ----------
    bp : BP
        The balanced parentheses tree.
    include_lengths : bool, optional
        Whether to include branch lengths. Default is True.

    Returns
    -------
    str
        A Newick format string.

    Examples
    --------
    >>> from skbio.tree._bp import parse_newick, write_newick
    >>> bp = parse_newick("((a:1,b:2)c:3,d:4)root;")
    >>> write_newick(bp)
    '((a:1.0,b:2.0)c:3.0,d:4.0)root;'
    """
    result = []
    stack = []

    i = 0
    while i < bp.size:
        if bp.B[i]:  # Opening
            if not bp.isleaf(i):
                result.append('(')
                stack.append(i)
            else:
                # Leaf
                name = bp.name(i)
                length = bp.length(i)
                if name is not None:
                    # Quote if needed
                    if any(c in str(name) for c in '():,;'):
                        result.append(f"'{name}'")
                    else:
                        result.append(str(name))
                if include_lengths and length is not None:
                    result.append(f':{length}')
            i += 1
        else:  # Closing
            if stack:
                open_pos = stack.pop()
                # Check if we just closed an internal node
                result.append(')')
                name = bp.name(open_pos)
                length = bp.length(open_pos)
                if name is not None:
                    if any(c in str(name) for c in '():,;'):
                        result.append(f"'{name}'")
                    else:
                        result.append(str(name))
                if include_lengths and length is not None:
                    result.append(f':{length}')

            # Check if next is opening (sibling) or closing
            i += 1
            if i < bp.size and bp.B[i] and stack:
                result.append(',')

    result.append(';')
    return ''.join(result)
