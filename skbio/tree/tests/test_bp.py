# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Tests for the balanced parentheses tree implementation."""

import unittest

import numpy as np
import numpy.testing as npt

from skbio.tree._bp import BP, parse_newick, write_newick


class TestBP(unittest.TestCase):
    """Tests for the BP class."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple tree: ((a, b)c, d)root
        # Structure in BP: 1 1 1 0 1 0 0 1 0 0 (r c a   b     d)
        self.simple_B = np.array(
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.uint8)
        self.simple_names = np.array(
            ['root', 'c', 'a', None, 'b', None, None, 'd', None, None],
            dtype=object)
        self.simple_lengths = np.array(
            [np.nan, 3.0, 1.0, np.nan, 2.0, np.nan, np.nan, 4.0, np.nan,
             np.nan], dtype=np.float64)
        self.simple_bp = BP(self.simple_B, names=self.simple_names,
                            lengths=self.simple_lengths)

    def test_init(self):
        """Test BP initialization."""
        bp = BP(self.simple_B)
        self.assertEqual(bp.size, 10)
        self.assertEqual(bp.numnodes(), 5)
        self.assertEqual(bp.ntips(), 3)

    def test_init_with_names(self):
        """Test BP initialization with names."""
        bp = self.simple_bp
        self.assertEqual(bp.name(0), 'root')
        self.assertEqual(bp.name(1), 'c')
        self.assertEqual(bp.name(2), 'a')
        self.assertEqual(bp.name(4), 'b')
        self.assertEqual(bp.name(7), 'd')

    def test_init_with_lengths(self):
        """Test BP initialization with lengths."""
        bp = self.simple_bp
        self.assertIsNone(bp.length(0))  # root has no length
        self.assertEqual(bp.length(1), 3.0)
        self.assertEqual(bp.length(2), 1.0)
        self.assertEqual(bp.length(4), 2.0)
        self.assertEqual(bp.length(7), 4.0)

    def test_B_property(self):
        """Test B property returns the array."""
        bp = self.simple_bp
        npt.assert_array_equal(bp.B, self.simple_B)

    def test_size_property(self):
        """Test size property."""
        self.assertEqual(self.simple_bp.size, 10)

    def test_open(self):
        """Test open() method."""
        bp = self.simple_bp
        self.assertTrue(bp.open(0))
        self.assertTrue(bp.open(1))
        self.assertTrue(bp.open(2))
        self.assertFalse(bp.open(3))
        self.assertTrue(bp.open(4))
        self.assertFalse(bp.open(5))

    def test_close(self):
        """Test close() method for matching parentheses."""
        bp = self.simple_bp
        # root at 0 closes at 9
        self.assertEqual(bp.close(0), 9)
        # c at 1 closes at 6
        self.assertEqual(bp.close(1), 6)
        # a at 2 closes at 3
        self.assertEqual(bp.close(2), 3)
        # b at 4 closes at 5
        self.assertEqual(bp.close(4), 5)
        # d at 7 closes at 8
        self.assertEqual(bp.close(7), 8)

    def test_isleaf(self):
        """Test isleaf() method."""
        bp = self.simple_bp
        self.assertFalse(bp.isleaf(0))  # root
        self.assertFalse(bp.isleaf(1))  # c
        self.assertTrue(bp.isleaf(2))   # a
        self.assertTrue(bp.isleaf(4))   # b
        self.assertTrue(bp.isleaf(7))   # d

    def test_parent(self):
        """Test parent() method."""
        bp = self.simple_bp
        # root has no parent
        self.assertEqual(bp.parent(0), -1)
        # c's parent is root
        self.assertEqual(bp.parent(1), 0)
        # a's parent is c
        self.assertEqual(bp.parent(2), 1)
        # b's parent is c
        self.assertEqual(bp.parent(4), 1)
        # d's parent is root
        self.assertEqual(bp.parent(7), 0)

    def test_fchild(self):
        """Test fchild() method (first child)."""
        bp = self.simple_bp
        # root's first child is c at position 1
        self.assertEqual(bp.fchild(0), 1)
        # c's first child is a at position 2
        self.assertEqual(bp.fchild(1), 2)
        # a is a leaf, no first child
        self.assertEqual(bp.fchild(2), -1)
        # d is a leaf, no first child
        self.assertEqual(bp.fchild(7), -1)

    def test_nsibling(self):
        """Test nsibling() method (next sibling)."""
        bp = self.simple_bp
        # c's next sibling is d
        self.assertEqual(bp.nsibling(1), 7)
        # a's next sibling is b
        self.assertEqual(bp.nsibling(2), 4)
        # b has no next sibling
        self.assertEqual(bp.nsibling(4), -1)
        # d has no next sibling
        self.assertEqual(bp.nsibling(7), -1)

    def test_psibling(self):
        """Test psibling() method (previous sibling)."""
        bp = self.simple_bp
        # c has no previous sibling
        self.assertEqual(bp.psibling(1), -1)
        # a has no previous sibling
        self.assertEqual(bp.psibling(2), -1)
        # b's previous sibling is a
        self.assertEqual(bp.psibling(4), 2)
        # d's previous sibling is c
        self.assertEqual(bp.psibling(7), 1)

    def test_depth(self):
        """Test depth() method."""
        bp = self.simple_bp
        # root has depth 0
        self.assertEqual(bp.depth(0), 0)
        # c has depth 1
        self.assertEqual(bp.depth(1), 1)
        # a has depth 2
        self.assertEqual(bp.depth(2), 2)
        # b has depth 2
        self.assertEqual(bp.depth(4), 2)
        # d has depth 1
        self.assertEqual(bp.depth(7), 1)

    def test_height(self):
        """Test height() method."""
        bp = self.simple_bp
        # root has height 2 (deepest is a or b)
        self.assertEqual(bp.height(0), 2)
        # c has height 1
        self.assertEqual(bp.height(1), 1)
        # a is leaf, height 0
        self.assertEqual(bp.height(2), 0)
        # d is leaf, height 0
        self.assertEqual(bp.height(7), 0)

    def test_subtree(self):
        """Test subtree() method (subtree size)."""
        bp = self.simple_bp
        # root has 5 nodes total
        self.assertEqual(bp.subtree(0), 5)
        # c has 3 nodes (c, a, b)
        self.assertEqual(bp.subtree(1), 3)
        # a has 1 node
        self.assertEqual(bp.subtree(2), 1)
        # d has 1 node
        self.assertEqual(bp.subtree(7), 1)

    def test_isancestor(self):
        """Test isancestor() method."""
        bp = self.simple_bp
        # root is ancestor of all
        self.assertTrue(bp.isancestor(0, 1))
        self.assertTrue(bp.isancestor(0, 2))
        self.assertTrue(bp.isancestor(0, 4))
        self.assertTrue(bp.isancestor(0, 7))
        # c is ancestor of a and b
        self.assertTrue(bp.isancestor(1, 2))
        self.assertTrue(bp.isancestor(1, 4))
        # c is not ancestor of d
        self.assertFalse(bp.isancestor(1, 7))
        # leaves are ancestors of themselves
        self.assertTrue(bp.isancestor(2, 2))

    def test_lca(self):
        """Test lca() method (lowest common ancestor)."""
        bp = self.simple_bp
        # LCA of a and b is c
        self.assertEqual(bp.lca(2, 4), 1)
        # LCA of a and d is root
        self.assertEqual(bp.lca(2, 7), 0)
        # LCA of c and d is root
        self.assertEqual(bp.lca(1, 7), 0)
        # LCA of same node is itself
        self.assertEqual(bp.lca(2, 2), 2)

    def test_preorder(self):
        """Test preorder() method."""
        bp = self.simple_bp
        # Preorder rank: root=1, c=2, a=3, b=4, d=5
        self.assertEqual(bp.preorder(0), 1)
        self.assertEqual(bp.preorder(1), 2)
        self.assertEqual(bp.preorder(2), 3)
        self.assertEqual(bp.preorder(4), 4)
        self.assertEqual(bp.preorder(7), 5)

    def test_preorderselect(self):
        """Test preorderselect() method."""
        bp = self.simple_bp
        self.assertEqual(bp.preorderselect(1), 0)  # root
        self.assertEqual(bp.preorderselect(2), 1)  # c
        self.assertEqual(bp.preorderselect(3), 2)  # a
        self.assertEqual(bp.preorderselect(4), 4)  # b
        self.assertEqual(bp.preorderselect(5), 7)  # d

    def test_postorder(self):
        """Test postorder() method."""
        bp = self.simple_bp
        # Postorder rank: a=1, b=2, c=3, d=4, root=5
        self.assertEqual(bp.postorder(2), 1)  # a
        self.assertEqual(bp.postorder(4), 2)  # b
        self.assertEqual(bp.postorder(1), 3)  # c
        self.assertEqual(bp.postorder(7), 4)  # d
        self.assertEqual(bp.postorder(0), 5)  # root

    def test_postorderselect(self):
        """Test postorderselect() method."""
        bp = self.simple_bp
        self.assertEqual(bp.postorderselect(1), 2)  # a
        self.assertEqual(bp.postorderselect(2), 4)  # b
        self.assertEqual(bp.postorderselect(3), 1)  # c
        self.assertEqual(bp.postorderselect(4), 7)  # d
        self.assertEqual(bp.postorderselect(5), 0)  # root

    def test_ntips(self):
        """Test ntips() method."""
        self.assertEqual(self.simple_bp.ntips(), 3)

    def test_numnodes(self):
        """Test numnodes() method."""
        self.assertEqual(self.simple_bp.numnodes(), 5)

    def test_root(self):
        """Test root() method."""
        self.assertEqual(self.simple_bp.root(), 0)

    def test_excess(self):
        """Test excess() method."""
        bp = self.simple_bp
        # excess at each position
        self.assertEqual(bp.excess(0), 1)   # 1 open
        self.assertEqual(bp.excess(1), 2)   # 2 opens
        self.assertEqual(bp.excess(2), 3)   # 3 opens
        self.assertEqual(bp.excess(3), 2)   # 3-1 = 2
        self.assertEqual(bp.excess(4), 3)   # 3+1-1 = 3
        self.assertEqual(bp.excess(5), 2)   # 3+1-1-1 = 2
        self.assertEqual(bp.excess(6), 1)   # ...
        self.assertEqual(bp.excess(7), 2)
        self.assertEqual(bp.excess(8), 1)
        self.assertEqual(bp.excess(9), 0)

    def test_rank(self):
        """Test rank() method."""
        bp = self.simple_bp
        # rank(1, i) counts opening parens up to position i
        self.assertEqual(bp.rank(1, 0), 0)
        self.assertEqual(bp.rank(1, 1), 1)
        self.assertEqual(bp.rank(1, 3), 3)
        self.assertEqual(bp.rank(1, 5), 4)
        self.assertEqual(bp.rank(1, 10), 5)

        # rank(0, i) counts closing parens
        self.assertEqual(bp.rank(0, 0), 0)
        self.assertEqual(bp.rank(0, 4), 1)
        self.assertEqual(bp.rank(0, 10), 5)

    def test_select(self):
        """Test select() method."""
        bp = self.simple_bp
        # select(1, k) finds k-th opening paren
        self.assertEqual(bp.select(1, 1), 0)
        self.assertEqual(bp.select(1, 2), 1)
        self.assertEqual(bp.select(1, 3), 2)
        self.assertEqual(bp.select(1, 4), 4)
        self.assertEqual(bp.select(1, 5), 7)

        # select(0, k) finds k-th closing paren
        self.assertEqual(bp.select(0, 1), 3)
        self.assertEqual(bp.select(0, 2), 5)
        self.assertEqual(bp.select(0, 3), 6)

    def test_repr(self):
        """Test __repr__() method."""
        bp = self.simple_bp
        repr_str = repr(bp)
        self.assertIn('BP', repr_str)
        self.assertIn('nodes: 5', repr_str)
        self.assertIn('tips: 3', repr_str)

    def test_shear(self):
        """Test shear() method."""
        bp = self.simple_bp
        # Shear to keep only tips a and d
        tip_positions = {2, 7}  # positions of a and d
        sheared = bp.shear(tip_positions)

        self.assertEqual(sheared.ntips(), 2)
        # After shearing and collapsing, we should have just root and two tips

    def test_shear_single_tip(self):
        """Test shear() with a single tip."""
        bp = self.simple_bp
        tip_positions = {2}  # just a
        sheared = bp.shear(tip_positions)
        self.assertEqual(sheared.ntips(), 1)

    def test_shear_empty_raises(self):
        """Test shear() with empty set raises ValueError."""
        with self.assertRaises(ValueError):
            self.simple_bp.shear(set())

    def test_shear_non_tip_raises(self):
        """Test shear() with non-tip position raises ValueError."""
        with self.assertRaises(ValueError):
            self.simple_bp.shear({0})  # root is not a tip

    def test_collapse(self):
        """Test collapse() method removes single-child nodes."""
        # Create a tree with single-child nodes: (((a)b)c)root
        B = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        names = np.array(['root', 'c', 'b', 'a', None, None, None, None],
                         dtype=object)
        lengths = np.array(
            [np.nan, 1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan],
            dtype=np.float64)
        bp = BP(B, names=names, lengths=lengths)

        collapsed = bp.collapse()
        # After collapse, should just have root and a
        self.assertEqual(collapsed.ntips(), 1)
        self.assertEqual(collapsed.numnodes(), 2)

    def test_collapse_no_single_child(self):
        """Test collapse() on tree without single-child nodes."""
        bp = self.simple_bp
        collapsed = bp.collapse()
        # Should be unchanged
        self.assertEqual(collapsed.numnodes(), bp.numnodes())
        self.assertEqual(collapsed.ntips(), bp.ntips())


class TestParseNewick(unittest.TestCase):
    """Tests for parse_newick function."""

    def test_simple_tree(self):
        """Test parsing a simple tree."""
        bp = parse_newick("((a,b)c,d)root;")
        self.assertEqual(bp.numnodes(), 5)
        self.assertEqual(bp.ntips(), 3)

    def test_tree_with_lengths(self):
        """Test parsing a tree with branch lengths."""
        bp = parse_newick("((a:1,b:2)c:3,d:4)root;")
        self.assertEqual(bp.numnodes(), 5)
        # Check lengths are preserved
        self.assertEqual(bp.length(bp.preorderselect(3)), 1.0)  # a
        self.assertEqual(bp.length(bp.preorderselect(4)), 2.0)  # b

    def test_tree_unnamed_internal_nodes(self):
        """Test parsing tree with unnamed internal nodes."""
        bp = parse_newick("((a,b),d);")
        self.assertEqual(bp.numnodes(), 5)
        self.assertEqual(bp.ntips(), 3)

    def test_single_node(self):
        """Test parsing a single node tree."""
        bp = parse_newick("a;")
        self.assertEqual(bp.numnodes(), 1)
        self.assertEqual(bp.ntips(), 1)
        self.assertEqual(bp.name(0), 'a')

    def test_tree_with_quotes(self):
        """Test parsing tree with quoted names."""
        bp = parse_newick("(('a:1',b)c);")
        self.assertEqual(bp.numnodes(), 4)
        # The quoted name should be preserved
        self.assertEqual(bp.name(bp.preorderselect(2)), 'a:1')

    def test_multifurcating_tree(self):
        """Test parsing a multifurcating tree."""
        bp = parse_newick("((a,b,c)d,e,f)root;")
        self.assertEqual(bp.numnodes(), 7)
        self.assertEqual(bp.ntips(), 5)

    def test_no_semicolon_raises(self):
        """Test that missing semicolon raises ValueError."""
        with self.assertRaises(ValueError):
            parse_newick("((a,b)c,d)root")


class TestWriteNewick(unittest.TestCase):
    """Tests for write_newick function."""

    def test_simple_tree(self):
        """Test writing a simple tree."""
        bp = parse_newick("((a,b)c,d)root;")
        result = write_newick(bp)
        self.assertTrue(result.endswith(';'))
        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertIn('c', result)
        self.assertIn('d', result)
        self.assertIn('root', result)

    def test_tree_with_lengths(self):
        """Test writing a tree with lengths."""
        bp = parse_newick("((a:1.0,b:2.0)c:3.0,d:4.0)root;")
        result = write_newick(bp)
        self.assertIn(':1.0', result)
        self.assertIn(':2.0', result)
        self.assertIn(':3.0', result)
        self.assertIn(':4.0', result)

    def test_roundtrip(self):
        """Test that parse->write preserves structure."""
        original = "((a:1.0,b:2.0)c:3.0,d:4.0)root;"
        bp = parse_newick(original)
        result = write_newick(bp)
        # Parse again and compare
        bp2 = parse_newick(result)
        self.assertEqual(bp.numnodes(), bp2.numnodes())
        self.assertEqual(bp.ntips(), bp2.ntips())


class TestBPEdgeCases(unittest.TestCase):
    """Tests for edge cases in BP class."""

    def test_single_node_tree(self):
        """Test a tree with a single node."""
        B = np.array([1, 0], dtype=np.uint8)
        names = np.array(['root', None], dtype=object)
        bp = BP(B, names=names)

        self.assertEqual(bp.numnodes(), 1)
        self.assertEqual(bp.ntips(), 1)
        self.assertTrue(bp.isleaf(0))
        self.assertEqual(bp.parent(0), -1)
        self.assertEqual(bp.depth(0), 0)
        self.assertEqual(bp.height(0), 0)

    def test_two_node_tree(self):
        """Test a tree with two nodes (root and one child)."""
        B = np.array([1, 1, 0, 0], dtype=np.uint8)
        names = np.array(['root', 'a', None, None], dtype=object)
        bp = BP(B, names=names)

        self.assertEqual(bp.numnodes(), 2)
        self.assertEqual(bp.ntips(), 1)
        self.assertFalse(bp.isleaf(0))
        self.assertTrue(bp.isleaf(1))
        self.assertEqual(bp.parent(1), 0)

    def test_deep_tree(self):
        """Test a deep linear tree."""
        # Tree: ((((a)b)c)d)root
        B = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        names = np.array(
            ['root', 'd', 'c', 'b', 'a', None, None, None, None, None],
            dtype=object)
        bp = BP(B, names=names)

        self.assertEqual(bp.numnodes(), 5)
        self.assertEqual(bp.ntips(), 1)
        self.assertEqual(bp.depth(bp.preorderselect(5)), 4)
        self.assertEqual(bp.height(0), 4)

    def test_wide_tree(self):
        """Test a wide tree with many children at root."""
        # Tree: (a,b,c,d,e)root
        B = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
        names = np.array(['root', 'a', None, 'b', None, 'c', None, 'd', None,
                          'e', None, None], dtype=object)
        bp = BP(B, names=names)

        self.assertEqual(bp.numnodes(), 6)
        self.assertEqual(bp.ntips(), 5)
        self.assertEqual(bp.height(0), 1)

        # Check siblings
        self.assertEqual(bp.nsibling(1), 3)  # a -> b
        self.assertEqual(bp.nsibling(3), 5)  # b -> c
        self.assertEqual(bp.nsibling(5), 7)  # c -> d
        self.assertEqual(bp.nsibling(7), 9)  # d -> e
        self.assertEqual(bp.nsibling(9), -1)  # e has no next sibling


if __name__ == '__main__':
    unittest.main()
