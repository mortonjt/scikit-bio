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
from skbio.stats.ordination._ctf import ctf, _build_tensor, _filter_tensor


class TestBuildTensor(unittest.TestCase):
    """Tests for tensor construction from table and metadata."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a feature table
        self.table = pd.DataFrame(
            [[10, 5, 3],
             [8, 6, 2],
             [12, 4, 5],
             [9, 7, 4],
             [11, 3, 6],
             [7, 8, 3]],
            index=['s1', 's2', 's3', 's4', 's5', 's6'],
            columns=['f1', 'f2', 'f3']
        )

        # Create metadata with 3 subjects and 2 time points
        self.metadata = pd.DataFrame({
            'subject': ['subj_0', 'subj_0', 'subj_1', 'subj_1', 'subj_2', 'subj_2'],
            'timepoint': ['t0', 't1', 't0', 't1', 't0', 't1']
        }, index=['s1', 's2', 's3', 's4', 's5', 's6'])

    def test_basic_tensor_construction(self):
        """Test basic tensor construction."""
        tensor, ind_ids, state_ids, feat_ids = _build_tensor(
            self.table, self.metadata, 'subject', 'timepoint'
        )

        # Shape should be (3 subjects, 2 timepoints, 3 features)
        self.assertEqual(tensor.shape, (3, 2, 3))

        # Check IDs
        self.assertEqual(len(ind_ids), 3)
        self.assertEqual(len(state_ids), 2)
        self.assertEqual(len(feat_ids), 3)

    def test_tensor_values(self):
        """Test that tensor contains correct values."""
        tensor, ind_ids, state_ids, feat_ids = _build_tensor(
            self.table, self.metadata, 'subject', 'timepoint'
        )

        # Find indices for subj_0 at t0 (which is sample s1)
        subj_idx = ind_ids.index('subj_0')
        time_idx = state_ids.index('t0')

        # Values should match s1
        npt.assert_array_equal(tensor[subj_idx, time_idx, :], [10, 5, 3])

    def test_missing_samples_in_metadata(self):
        """Test handling of samples not in metadata."""
        # Add a sample to table that's not in metadata
        table_extra = self.table.copy()
        table_extra.loc['s7'] = [1, 2, 3]

        tensor, _, _, _ = _build_tensor(
            table_extra, self.metadata, 'subject', 'timepoint'
        )

        # Extra sample should be ignored
        self.assertEqual(tensor.shape[0], 3)

    def test_no_common_samples_error(self):
        """Test error when no common samples."""
        metadata_diff = pd.DataFrame({
            'subject': ['subj_0'],
            'timepoint': ['t0']
        }, index=['other_sample'])

        with self.assertRaises(ValueError) as context:
            _build_tensor(self.table, metadata_diff, 'subject', 'timepoint')

        self.assertIn("No common samples", str(context.exception))


class TestFilterTensor(unittest.TestCase):
    """Tests for tensor filtering."""

    def setUp(self):
        """Set up test fixtures."""
        self.tensor = np.array([
            [[10, 5, 0], [8, 6, 2]],
            [[12, 4, 5], [9, 7, 4]],
            [[11, 3, 6], [7, 8, 3]]
        ], dtype=float)
        self.ind_ids = ['subj_0', 'subj_1', 'subj_2']
        self.state_ids = ['t0', 't1']
        self.feat_ids = ['f1', 'f2', 'f3']

    def test_no_filtering(self):
        """Test that no filtering preserves data."""
        result = _filter_tensor(
            self.tensor, self.ind_ids, self.state_ids, self.feat_ids
        )
        tensor_out, ind_out, state_out, feat_out = result

        npt.assert_array_equal(tensor_out, self.tensor)
        self.assertEqual(ind_out, self.ind_ids)
        self.assertEqual(state_out, self.state_ids)
        self.assertEqual(feat_out, self.feat_ids)

    def test_filter_by_feature_count(self):
        """Test filtering features by count."""
        result = _filter_tensor(
            self.tensor, self.ind_ids, self.state_ids, self.feat_ids,
            min_feature_count=30
        )
        tensor_out, _, _, feat_out = result

        # f1 has sum=57, f2 has sum=33, f3 has sum=20
        self.assertIn('f1', feat_out)
        self.assertIn('f2', feat_out)
        self.assertNotIn('f3', feat_out)


class TestCTF(unittest.TestCase):
    """Tests for Compositional Tensor Factorization."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create a feature table for longitudinal data
        # 6 subjects, 4 time points, 15 features
        n_subjects = 6
        n_timepoints = 4
        n_features = 15
        n_samples = n_subjects * n_timepoints

        counts = np.random.poisson(10, size=(n_samples, n_features))
        counts[counts < 2] = 0

        sample_ids = ['s%d' % i for i in range(n_samples)]
        self.table = pd.DataFrame(
            counts,
            index=sample_ids,
            columns=['f%d' % i for i in range(n_features)]
        )

        # Create metadata
        subjects = []
        timepoints = []
        for i in range(n_samples):
            subjects.append('subj_%d' % (i // n_timepoints))
            timepoints.append('t%d' % (i % n_timepoints))

        self.metadata = pd.DataFrame({
            'subject_id': subjects,
            'time': timepoints
        }, index=sample_ids)

    def test_basic_ctf(self):
        """Test basic CTF analysis."""
        subject_ord, state_ord = ctf(
            self.table, self.metadata,
            individual_id_column='subject_id',
            state_column='time',
            n_components=2,
            max_als_iterations=10,
            n_initializations=2
        )

        # Check subject ordination
        self.assertIsInstance(subject_ord, OrdinationResults)
        self.assertEqual(subject_ord.short_method_name, 'CTF')

        # Check state ordination
        self.assertIsInstance(state_ord, OrdinationResults)

    def test_ctf_subject_ordination_shape(self):
        """Test subject ordination has correct shape."""
        subject_ord, _ = ctf(
            self.table, self.metadata,
            'subject_id', 'time',
            n_components=2,
            max_als_iterations=10,
            n_initializations=2
        )

        # 6 subjects, 2 components
        self.assertEqual(subject_ord.samples.shape[0], 6)
        self.assertEqual(subject_ord.samples.shape[1], 2)

    def test_ctf_state_ordination_shape(self):
        """Test state ordination has correct shape."""
        _, state_ord = ctf(
            self.table, self.metadata,
            'subject_id', 'time',
            n_components=2,
            max_als_iterations=10,
            n_initializations=2
        )

        # 4 time points, 2 components
        self.assertEqual(state_ord.samples.shape[0], 4)
        self.assertEqual(state_ord.samples.shape[1], 2)

    def test_ctf_non_dataframe_table_error(self):
        """Test error on non-DataFrame table."""
        with self.assertRaises(ValueError) as context:
            ctf(self.table.values, self.metadata, 'subject_id', 'time')

        self.assertIn("DataFrame", str(context.exception))

    def test_ctf_non_dataframe_metadata_error(self):
        """Test error on non-DataFrame metadata."""
        with self.assertRaises(ValueError) as context:
            ctf(self.table, self.metadata.values, 'subject_id', 'time')

        self.assertIn("DataFrame", str(context.exception))

    def test_ctf_missing_column_error(self):
        """Test error on missing metadata column."""
        with self.assertRaises(ValueError) as context:
            ctf(self.table, self.metadata, 'nonexistent_column', 'time')

        self.assertIn("not found", str(context.exception))

    def test_ctf_negative_values_error(self):
        """Test error on negative values in table."""
        table_neg = self.table.copy()
        table_neg.iloc[0, 0] = -5

        with self.assertRaises(ValueError) as context:
            ctf(table_neg, self.metadata, 'subject_id', 'time')

        self.assertIn("negative", str(context.exception))


class TestCTFWithFiltering(unittest.TestCase):
    """Tests for CTF with filtering parameters."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        n_subjects = 5
        n_timepoints = 3
        n_features = 12
        n_samples = n_subjects * n_timepoints

        counts = np.random.poisson(8, size=(n_samples, n_features))
        counts[counts < 3] = 0

        sample_ids = ['s%d' % i for i in range(n_samples)]
        self.table = pd.DataFrame(
            counts,
            index=sample_ids,
            columns=['f%d' % i for i in range(n_features)]
        )

        subjects = ['subj_%d' % (i // n_timepoints) for i in range(n_samples)]
        timepoints = ['t%d' % (i % n_timepoints) for i in range(n_samples)]

        self.metadata = pd.DataFrame({
            'subject': subjects,
            'time': timepoints
        }, index=sample_ids)

    def test_ctf_with_filtering(self):
        """Test CTF with various filtering parameters."""
        subject_ord, state_ord = ctf(
            self.table, self.metadata,
            'subject', 'time',
            n_components=2,
            min_feature_count=10,
            min_feature_frequency=0.1,
            max_als_iterations=10,
            n_initializations=2
        )

        # Should still produce valid results
        self.assertIsInstance(subject_ord, OrdinationResults)
        self.assertIsInstance(state_ord, OrdinationResults)


class TestCTFAxisLabels(unittest.TestCase):
    """Tests for CTF axis labeling."""

    def test_axis_labels(self):
        """Test that axis labels are correct."""
        np.random.seed(42)

        n_subjects = 4
        n_timepoints = 3
        n_features = 10
        n_samples = n_subjects * n_timepoints

        counts = np.random.poisson(10, size=(n_samples, n_features))
        counts[counts < 2] = 0

        sample_ids = ['s%d' % i for i in range(n_samples)]
        table = pd.DataFrame(
            counts, index=sample_ids,
            columns=['f%d' % i for i in range(n_features)]
        )

        subjects = ['subj_%d' % (i // n_timepoints) for i in range(n_samples)]
        timepoints = ['t%d' % (i % n_timepoints) for i in range(n_samples)]
        metadata = pd.DataFrame({
            'subject': subjects, 'time': timepoints
        }, index=sample_ids)

        subject_ord, state_ord = ctf(
            table, metadata, 'subject', 'time',
            n_components=2,
            max_als_iterations=10,
            n_initializations=2
        )

        expected_labels = ['PC1', 'PC2']

        self.assertListEqual(
            list(subject_ord.samples.columns), expected_labels
        )
        self.assertListEqual(
            list(state_ord.samples.columns), expected_labels
        )


if __name__ == '__main__':
    unittest.main()
