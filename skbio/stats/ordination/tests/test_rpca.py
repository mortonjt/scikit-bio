# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

from skbio import OrdinationResults
from skbio.stats.ordination._rpca import rpca, _filter_table


class TestFilterTable(unittest.TestCase):
    """Tests for table filtering function."""

    def setUp(self):
        """Set up test fixtures."""
        self.table = pd.DataFrame(
            [[10, 0, 5, 2],
             [20, 3, 0, 8],
             [5, 1, 1, 1],
             [0, 0, 0, 1],
             [15, 2, 3, 4]],
            index=['s1', 's2', 's3', 's4', 's5'],
            columns=['f1', 'f2', 'f3', 'f4']
        )

    def test_no_filtering(self):
        """Test that no filtering preserves table."""
        result = _filter_table(self.table)
        pd.testing.assert_frame_equal(result, self.table)

    def test_filter_by_sample_count(self):
        """Test filtering by minimum sample count."""
        result = _filter_table(self.table, min_sample_count=10)

        # s4 has sum=1, should be removed
        self.assertNotIn('s4', result.index)
        self.assertEqual(len(result), 4)

    def test_filter_by_feature_count(self):
        """Test filtering by minimum feature count."""
        result = _filter_table(self.table, min_feature_count=10)

        # Features with sum >= 10: f1=50, f4=16
        self.assertIn('f1', result.columns)
        self.assertIn('f4', result.columns)
        self.assertEqual(len(result.columns), 2)

    def test_filter_by_feature_frequency(self):
        """Test filtering by minimum feature frequency."""
        # f3 appears in 4/5 = 0.8 of samples
        # f2 appears in 3/5 = 0.6 of samples
        result = _filter_table(self.table, min_feature_frequency=0.7)

        # f1 appears in 4/5, f3 in 4/5, f4 in 5/5
        self.assertIn('f1', result.columns)
        self.assertIn('f3', result.columns)
        self.assertIn('f4', result.columns)


class TestRPCA(unittest.TestCase):
    """Tests for Robust PCA ordination."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create a simple count table
        n_samples, n_features = 15, 20

        # Generate counts with some zeros
        counts = np.random.poisson(5, size=(n_samples, n_features))
        counts[counts < 2] = 0

        self.table = pd.DataFrame(
            counts,
            index=['sample_%d' % i for i in range(n_samples)],
            columns=['feature_%d' % i for i in range(n_features)]
        )

    def test_basic_rpca(self):
        """Test basic RPCA analysis."""
        ordination = rpca(self.table, n_components=3)

        # Check ordination results
        self.assertIsInstance(ordination, OrdinationResults)
        self.assertEqual(ordination.short_method_name, 'RPCA')

        # Check samples shape
        self.assertEqual(ordination.samples.shape[0], self.table.shape[0])
        self.assertEqual(ordination.samples.shape[1], 3)

        # Check features shape
        self.assertEqual(ordination.features.shape[0], self.table.shape[1])
        self.assertEqual(ordination.features.shape[1], 3)

    def test_rpca_preserves_sample_ids(self):
        """Test that sample IDs are preserved."""
        ordination = rpca(self.table, n_components=2)

        self.assertListEqual(
            list(ordination.samples.index),
            list(self.table.index)
        )

    def test_rpca_preserves_feature_ids(self):
        """Test that feature IDs are preserved."""
        ordination = rpca(self.table, n_components=2)

        self.assertListEqual(
            list(ordination.features.index),
            list(self.table.columns)
        )

    def test_rpca_proportion_explained(self):
        """Test that proportion explained sums to <= 1."""
        ordination = rpca(self.table, n_components=3)

        # Proportion should sum to approximately 1 or less
        total_prop = ordination.proportion_explained.sum()
        self.assertLessEqual(total_prop, 1.01)  # Allow small numerical error

        # Each proportion should be non-negative
        self.assertTrue(all(ordination.proportion_explained >= 0))

    def test_rpca_eigvals_decreasing(self):
        """Test that eigenvalues are in decreasing order."""
        ordination = rpca(self.table, n_components=3)

        eigvals = ordination.eigvals.values
        for i in range(len(eigvals) - 1):
            self.assertGreaterEqual(eigvals[i], eigvals[i + 1])

    def test_rpca_with_filtering(self):
        """Test RPCA with filtering parameters."""
        ordination = rpca(
            self.table,
            n_components=2,
            min_sample_count=5,
            min_feature_count=5,
            min_feature_frequency=0.1
        )

        # Results should still be valid
        self.assertIsInstance(ordination, OrdinationResults)

    def test_rpca_non_dataframe_error(self):
        """Test error on non-DataFrame input."""
        with self.assertRaises(ValueError) as context:
            rpca(self.table.values, n_components=2)

        self.assertIn("DataFrame", str(context.exception))

    def test_rpca_negative_values_error(self):
        """Test error on negative values."""
        table_neg = self.table.copy()
        table_neg.iloc[0, 0] = -5

        with self.assertRaises(ValueError) as context:
            rpca(table_neg, n_components=2)

        self.assertIn("negative", str(context.exception))

    def test_rpca_insufficient_samples_error(self):
        """Test error when too few samples after filtering."""
        small_table = self.table.iloc[:2, :]

        with self.assertRaises(ValueError) as context:
            rpca(small_table, n_components=2)

        self.assertIn("samples", str(context.exception))

    def test_rpca_insufficient_features_error(self):
        """Test error when n_components exceeds features."""
        small_table = self.table.iloc[:, :2]

        with self.assertRaises(ValueError) as context:
            rpca(small_table, n_components=5)

        self.assertIn("features", str(context.exception))


class TestRPCAReproducibility(unittest.TestCase):
    """Tests for RPCA reproducibility."""

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with same random seed."""
        np.random.seed(42)
        counts = np.random.poisson(5, size=(10, 15))
        counts[counts < 2] = 0

        table = pd.DataFrame(
            counts,
            index=['s%d' % i for i in range(10)],
            columns=['f%d' % i for i in range(15)]
        )

        # Run twice with same seed
        np.random.seed(123)
        ord1 = rpca(table, n_components=2)

        np.random.seed(123)
        ord2 = rpca(table, n_components=2)

        # Results should be identical
        npt.assert_almost_equal(
            ord1.samples.values, ord2.samples.values
        )


class TestRPCAAxisLabels(unittest.TestCase):
    """Tests for RPCA axis labeling."""

    def test_axis_labels(self):
        """Test that axis labels are correct."""
        np.random.seed(42)
        counts = np.random.poisson(5, size=(10, 15))
        counts[counts < 2] = 0

        table = pd.DataFrame(
            counts,
            index=['s%d' % i for i in range(10)],
            columns=['f%d' % i for i in range(15)]
        )

        ordination = rpca(table, n_components=3)

        expected_labels = ['PC1', 'PC2', 'PC3']
        self.assertListEqual(list(ordination.eigvals.index), expected_labels)
        self.assertListEqual(list(ordination.samples.columns), expected_labels)
        self.assertListEqual(list(ordination.features.columns), expected_labels)
        self.assertListEqual(
            list(ordination.proportion_explained.index), expected_labels
        )


if __name__ == '__main__':
    unittest.main()
