# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Tests for the FastTreeNode class."""

import io
import unittest

from skbio import TreeNode
from skbio.tree import FastTreeNode
from skbio.tree._exception import MissingNodeError, NoParentError


class TestFastTreeNodeBasics(unittest.TestCase):
    """Tests for basic FastTreeNode functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.simple_tree = FastTreeNode.read(
            io.StringIO("((a,b)c,(d,e)f)root;"))
        self.tree_with_lengths = FastTreeNode.read(
            io.StringIO("((a:1,b:2)c:3,(d:4,e:5)f:6)root;"))

    def test_read_newick(self):
        """Test reading a Newick string."""
        tree = FastTreeNode.read(io.StringIO("((a,b)c);"))
        self.assertIsInstance(tree, FastTreeNode)
        self.assertEqual(tree.count(), 4)

    def test_name_property(self):
        """Test name property."""
        tree = self.simple_tree
        self.assertEqual(tree.name, 'root')

    def test_length_property(self):
        """Test length property."""
        tree = self.tree_with_lengths
        self.assertIsNone(tree.length)  # root has no length
        tip_a = tree.find('a')
        self.assertEqual(tip_a.length, 1.0)

    def test_is_tip(self):
        """Test is_tip() method."""
        tree = self.simple_tree
        self.assertFalse(tree.is_tip())
        self.assertTrue(tree.find('a').is_tip())
        self.assertTrue(tree.find('b').is_tip())
        self.assertFalse(tree.find('c').is_tip())

    def test_is_root(self):
        """Test is_root() method."""
        tree = self.simple_tree
        self.assertTrue(tree.is_root())
        self.assertFalse(tree.find('a').is_root())
        self.assertFalse(tree.find('c').is_root())

    def test_has_children(self):
        """Test has_children() method."""
        tree = self.simple_tree
        self.assertTrue(tree.has_children())
        self.assertFalse(tree.find('a').has_children())

    def test_parent_property(self):
        """Test parent property."""
        tree = self.simple_tree
        self.assertIsNone(tree.parent)
        self.assertEqual(tree.find('a').parent.name, 'c')
        self.assertEqual(tree.find('c').parent.name, 'root')

    def test_children_property(self):
        """Test children property."""
        tree = self.simple_tree
        children_names = [c.name for c in tree.children]
        self.assertIn('c', children_names)
        self.assertIn('f', children_names)
        self.assertEqual(len(children_names), 2)

    def test_root_method(self):
        """Test root() method."""
        tree = self.simple_tree
        tip_a = tree.find('a')
        self.assertEqual(tip_a.root().name, 'root')

    def test_ancestors(self):
        """Test ancestors() method."""
        tree = self.simple_tree
        tip_a = tree.find('a')
        ancestors = [n.name for n in tip_a.ancestors()]
        self.assertEqual(ancestors, ['c', 'root'])

    def test_siblings(self):
        """Test siblings() method."""
        tree = self.simple_tree
        tip_a = tree.find('a')
        siblings = [n.name for n in tip_a.siblings()]
        self.assertEqual(siblings, ['b'])

    def test_neighbors(self):
        """Test neighbors() method."""
        tree = self.simple_tree
        node_c = tree.find('c')
        neighbors = [n.name for n in node_c.neighbors()]
        self.assertIn('a', neighbors)
        self.assertIn('b', neighbors)
        self.assertIn('root', neighbors)

    def test_depth(self):
        """Test depth() method."""
        tree = self.simple_tree
        self.assertEqual(tree.depth(), 0)
        self.assertEqual(tree.find('c').depth(), 1)
        self.assertEqual(tree.find('a').depth(), 2)

    def test_height(self):
        """Test height() method."""
        tree = self.simple_tree
        self.assertEqual(tree.height(), 2)
        self.assertEqual(tree.find('c').height(), 1)
        self.assertEqual(tree.find('a').height(), 0)

    def test_count(self):
        """Test count() method."""
        tree = self.simple_tree
        self.assertEqual(tree.count(), 7)
        self.assertEqual(tree.count(tips=True), 4)

    def test_repr(self):
        """Test __repr__() method."""
        tree = self.simple_tree
        repr_str = repr(tree)
        self.assertIn('FastTreeNode', repr_str)
        self.assertIn('root', repr_str)

    def test_str(self):
        """Test __str__() method."""
        tree = self.simple_tree
        newick_str = str(tree)
        self.assertTrue(newick_str.endswith(';'))

    def test_eq(self):
        """Test __eq__() method."""
        tree = self.simple_tree
        tip_a = tree.find('a')
        tip_a2 = tree.find('a')
        self.assertEqual(tip_a, tip_a2)
        self.assertNotEqual(tip_a, tree.find('b'))

    def test_hash(self):
        """Test __hash__() method."""
        tree = self.simple_tree
        tip_a = tree.find('a')
        tip_a2 = tree.find('a')
        self.assertEqual(hash(tip_a), hash(tip_a2))


class TestFastTreeNodeTraversal(unittest.TestCase):
    """Tests for FastTreeNode traversal methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.tree = FastTreeNode.read(io.StringIO("((a,b)c,(d,e)f)root;"))

    def test_preorder(self):
        """Test preorder() traversal."""
        names = [n.name for n in self.tree.preorder()]
        # Should be: root, c, a, b, f, d, e
        self.assertEqual(names[0], 'root')
        self.assertIn('c', names)
        self.assertIn('f', names)
        self.assertEqual(len(names), 7)

    def test_preorder_exclude_self(self):
        """Test preorder() with include_self=False."""
        names = [n.name for n in self.tree.preorder(include_self=False)]
        self.assertNotIn('root', names)
        self.assertEqual(len(names), 6)

    def test_postorder(self):
        """Test postorder() traversal."""
        names = [n.name for n in self.tree.postorder()]
        # Tips come first, then internal nodes
        self.assertEqual(names[-1], 'root')
        # a and b before c; d and e before f
        a_idx = names.index('a')
        b_idx = names.index('b')
        c_idx = names.index('c')
        self.assertLess(a_idx, c_idx)
        self.assertLess(b_idx, c_idx)

    def test_postorder_exclude_self(self):
        """Test postorder() with include_self=False."""
        names = [n.name for n in self.tree.postorder(include_self=False)]
        self.assertNotIn('root', names)
        self.assertEqual(len(names), 6)

    def test_levelorder(self):
        """Test levelorder() traversal."""
        names = [n.name for n in self.tree.levelorder()]
        # Level 0: root
        # Level 1: c, f
        # Level 2: a, b, d, e
        self.assertEqual(names[0], 'root')
        self.assertIn('c', names[1:3])
        self.assertIn('f', names[1:3])

    def test_tips(self):
        """Test tips() method."""
        tip_names = [n.name for n in self.tree.tips()]
        self.assertEqual(set(tip_names), {'a', 'b', 'd', 'e'})

    def test_non_tips(self):
        """Test non_tips() method."""
        non_tip_names = [n.name for n in self.tree.non_tips()]
        self.assertEqual(set(non_tip_names), {'c', 'f'})

    def test_non_tips_include_self(self):
        """Test non_tips() with include_self=True."""
        non_tip_names = [n.name for n in self.tree.non_tips(include_self=True)]
        self.assertEqual(set(non_tip_names), {'c', 'f', 'root'})

    def test_traverse_default(self):
        """Test traverse() with default parameters."""
        names = [n.name for n in self.tree.traverse()]
        # Default is preorder
        self.assertEqual(names[0], 'root')


class TestFastTreeNodeSearch(unittest.TestCase):
    """Tests for FastTreeNode search methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.tree = FastTreeNode.read(io.StringIO("((a,b)c,(d,e)c)root;"))

    def test_find_tip(self):
        """Test find() for a tip."""
        node = self.tree.find('a')
        self.assertEqual(node.name, 'a')
        self.assertTrue(node.is_tip())

    def test_find_internal(self):
        """Test find() for an internal node."""
        node = self.tree.find('c')
        self.assertEqual(node.name, 'c')
        self.assertFalse(node.is_tip())

    def test_find_missing_raises(self):
        """Test find() raises MissingNodeError for missing node."""
        with self.assertRaises(MissingNodeError):
            self.tree.find('nonexistent')

    def test_find_with_node(self):
        """Test find() with FastTreeNode returns the node."""
        tip_a = self.tree.find('a')
        result = self.tree.find(tip_a)
        self.assertEqual(result, tip_a)

    def test_find_all(self):
        """Test find_all() for nodes with same name."""
        nodes = self.tree.find_all('c')
        self.assertEqual(len(nodes), 2)
        for node in nodes:
            self.assertEqual(node.name, 'c')

    def test_find_all_single(self):
        """Test find_all() for unique node."""
        nodes = self.tree.find_all('a')
        self.assertEqual(len(nodes), 1)

    def test_find_all_missing_raises(self):
        """Test find_all() raises MissingNodeError for missing node."""
        with self.assertRaises(MissingNodeError):
            self.tree.find_all('nonexistent')

    def test_find_by_func(self):
        """Test find_by_func() method."""
        tips = list(self.tree.find_by_func(lambda x: x.is_tip()))
        self.assertEqual(len(tips), 4)
        for tip in tips:
            self.assertTrue(tip.is_tip())


class TestFastTreeNodeLCA(unittest.TestCase):
    """Tests for FastTreeNode lowest common ancestor methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.tree = FastTreeNode.read(
            io.StringIO("((a,b)c,(d,e)f)root;"))

    def test_lca_same_parent(self):
        """Test LCA of siblings."""
        lca = self.tree.lowest_common_ancestor(['a', 'b'])
        self.assertEqual(lca.name, 'c')

    def test_lca_different_subtrees(self):
        """Test LCA of nodes in different subtrees."""
        lca = self.tree.lowest_common_ancestor(['a', 'd'])
        self.assertEqual(lca.name, 'root')

    def test_lca_single_node(self):
        """Test LCA of a single node."""
        lca = self.tree.lowest_common_ancestor(['a'])
        self.assertEqual(lca.name, 'a')

    def test_lca_alias(self):
        """Test that lca is an alias for lowest_common_ancestor."""
        lca1 = self.tree.lowest_common_ancestor(['a', 'b'])
        lca2 = self.tree.lca(['a', 'b'])
        self.assertEqual(lca1, lca2)


class TestFastTreeNodeDistance(unittest.TestCase):
    """Tests for FastTreeNode distance methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.tree = FastTreeNode.read(
            io.StringIO("((a:1,b:2)c:3,(d:4,e:5)f:6)root;"))

    def test_distance_same_node(self):
        """Test distance to self is 0."""
        tip_a = self.tree.find('a')
        self.assertEqual(tip_a.distance(tip_a), 0.0)

    def test_distance_siblings(self):
        """Test distance between siblings."""
        tip_a = self.tree.find('a')
        tip_b = self.tree.find('b')
        # a:1 + b:2 = 3
        self.assertEqual(tip_a.distance(tip_b), 3.0)

    def test_distance_different_subtrees(self):
        """Test distance between nodes in different subtrees."""
        tip_a = self.tree.find('a')
        tip_d = self.tree.find('d')
        # a:1 + c:3 + f:6 + d:4 = 14
        self.assertEqual(tip_a.distance(tip_d), 14.0)

    def test_accumulate_to_ancestor(self):
        """Test accumulate_to_ancestor() method."""
        tip_a = self.tree.find('a')
        root = self.tree
        # a:1 + c:3 = 4
        self.assertEqual(tip_a.accumulate_to_ancestor(root), 4.0)

    def test_accumulate_to_ancestor_not_ancestor_raises(self):
        """Test accumulate_to_ancestor() raises for non-ancestor."""
        tip_a = self.tree.find('a')
        tip_b = self.tree.find('b')
        with self.assertRaises(NoParentError):
            tip_a.accumulate_to_ancestor(tip_b)


class TestFastTreeNodeSubset(unittest.TestCase):
    """Tests for FastTreeNode subset methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.tree = FastTreeNode.read(
            io.StringIO("((a,b)c,(d,e)f)root;"))

    def test_subset(self):
        """Test subset() method."""
        node_c = self.tree.find('c')
        subset = node_c.subset()
        self.assertEqual(subset, frozenset({'a', 'b'}))

    def test_subset_root(self):
        """Test subset() at root."""
        subset = self.tree.subset()
        self.assertEqual(subset, frozenset({'a', 'b', 'd', 'e'}))

    def test_subset_tip(self):
        """Test subset() at tip."""
        tip_a = self.tree.find('a')
        subset = tip_a.subset()
        self.assertEqual(subset, frozenset({'a'}))


class TestFastTreeNodeShear(unittest.TestCase):
    """Tests for FastTreeNode shear method."""

    def setUp(self):
        """Set up test fixtures."""
        self.tree = FastTreeNode.read(
            io.StringIO("((a,b)c,(d,e)f)root;"))

    def test_shear_subset(self):
        """Test shearing to a subset of tips."""
        sheared = self.tree.shear(['a', 'd'])
        tip_names = {t.name for t in sheared.tips()}
        self.assertEqual(tip_names, {'a', 'd'})

    def test_shear_all_tips(self):
        """Test shearing to all tips (should be same structure)."""
        sheared = self.tree.shear(['a', 'b', 'd', 'e'])
        tip_names = {t.name for t in sheared.tips()}
        self.assertEqual(tip_names, {'a', 'b', 'd', 'e'})

    def test_shear_invalid_tips_raises(self):
        """Test shearing with invalid tips raises ValueError."""
        with self.assertRaises(ValueError):
            self.tree.shear(['a', 'nonexistent'])


class TestFastTreeNodeCopy(unittest.TestCase):
    """Tests for FastTreeNode copy method."""

    def setUp(self):
        """Set up test fixtures."""
        self.tree = FastTreeNode.read(
            io.StringIO("((a,b)c,(d,e)f)root;"))

    def test_copy(self):
        """Test copy() creates independent tree."""
        copy = self.tree.copy()
        self.assertIsNot(copy, self.tree)
        self.assertEqual(copy.count(), self.tree.count())
        self.assertEqual(copy.name, self.tree.name)


class TestFastTreeNodeConversion(unittest.TestCase):
    """Tests for FastTreeNode conversion methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.newick = "((a:1,b:2)c:3,(d:4,e:5)f:6)root;"
        self.tree = FastTreeNode.read(io.StringIO(self.newick))

    def test_from_tree_node(self):
        """Test from_tree_node() class method."""
        tn = TreeNode.read(io.StringIO(self.newick))
        ftn = FastTreeNode.from_tree_node(tn)

        self.assertIsInstance(ftn, FastTreeNode)
        self.assertEqual(ftn.count(), tn.count())
        self.assertEqual(ftn.count(tips=True), tn.count(tips=True))

    def test_to_tree_node(self):
        """Test to_tree_node() method."""
        tn = self.tree.to_tree_node()

        self.assertIsInstance(tn, TreeNode)
        self.assertEqual(tn.count(), self.tree.count())
        self.assertEqual(tn.count(tips=True), self.tree.count(tips=True))

    def test_roundtrip_conversion(self):
        """Test that conversions preserve structure."""
        tn1 = TreeNode.read(io.StringIO(self.newick))
        ftn = FastTreeNode.from_tree_node(tn1)
        tn2 = ftn.to_tree_node()

        # Compare tip names
        tips1 = {t.name for t in tn1.tips()}
        tips2 = {t.name for t in tn2.tips()}
        self.assertEqual(tips1, tips2)


class TestFastTreeNodeIO(unittest.TestCase):
    """Tests for FastTreeNode I/O methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.newick = "((a:1,b:2)c:3,(d:4,e:5)f:6)root;"

    def test_read_from_stringio(self):
        """Test read() from StringIO."""
        tree = FastTreeNode.read(io.StringIO(self.newick))
        self.assertEqual(tree.name, 'root')

    def test_write_to_stringio(self):
        """Test write() to StringIO."""
        tree = FastTreeNode.read(io.StringIO(self.newick))
        out = io.StringIO()
        tree.write(out)
        result = out.getvalue()
        self.assertTrue(result.endswith(';'))

    def test_read_write_roundtrip(self):
        """Test that read/write preserves structure."""
        tree1 = FastTreeNode.read(io.StringIO(self.newick))
        out = io.StringIO()
        tree1.write(out)
        out.seek(0)
        tree2 = FastTreeNode.read(out)

        self.assertEqual(tree1.count(), tree2.count())
        tips1 = {t.name for t in tree1.tips()}
        tips2 = {t.name for t in tree2.tips()}
        self.assertEqual(tips1, tips2)


class TestFastTreeNodeASCIIArt(unittest.TestCase):
    """Tests for FastTreeNode ASCII art method."""

    def test_ascii_art(self):
        """Test ascii_art() returns string."""
        tree = FastTreeNode.read(io.StringIO("((a,b)c);"))
        art = tree.ascii_art()
        self.assertIsInstance(art, str)
        self.assertIn('a', art)
        self.assertIn('b', art)


class TestFastTreeNodeEdgeCases(unittest.TestCase):
    """Tests for edge cases in FastTreeNode."""

    def test_single_node_tree(self):
        """Test a tree with a single node."""
        tree = FastTreeNode.read(io.StringIO("a;"))
        self.assertEqual(tree.count(), 1)
        self.assertTrue(tree.is_tip())
        self.assertTrue(tree.is_root())
        self.assertEqual(tree.name, 'a')

    def test_two_node_tree(self):
        """Test a tree with two nodes."""
        tree = FastTreeNode.read(io.StringIO("(a)root;"))
        self.assertEqual(tree.count(), 2)
        self.assertFalse(tree.is_tip())
        self.assertEqual(len(tree.children), 1)

    def test_multifurcating_tree(self):
        """Test a multifurcating tree."""
        tree = FastTreeNode.read(io.StringIO("(a,b,c,d,e)root;"))
        self.assertEqual(tree.count(), 6)
        self.assertEqual(len(tree.children), 5)

    def test_tree_with_unnamed_nodes(self):
        """Test a tree with unnamed nodes."""
        tree = FastTreeNode.read(io.StringIO("((a,b),(c,d));"))
        self.assertEqual(tree.count(), 7)
        self.assertIsNone(tree.name)

    def test_empty_init(self):
        """Test FastTreeNode with default init."""
        tree = FastTreeNode(name='test', length=1.0)
        self.assertEqual(tree.name, 'test')
        self.assertEqual(tree.length, 1.0)
        self.assertTrue(tree.is_tip())
        self.assertTrue(tree.is_root())


if __name__ == '__main__':
    unittest.main()
