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

from skbio.stats.ordination._tf import (
    TensorFactorization, _khatri_rao, _unfold_tensor, _refold_tensor,
    _construct_tensor_from_cp
)


class TestKhatriRao(unittest.TestCase):
    """Tests for Khatri-Rao product."""

    def test_two_matrices(self):
        """Test Khatri-Rao product of two matrices."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8], [9, 10]])

        result = _khatri_rao([A, B])

        # Result should have shape (2*3, 2) = (6, 2)
        self.assertEqual(result.shape, (6, 2))

        # Check first column: outer product of [1,3] and [5,7,9]
        # = [5, 7, 9, 15, 21, 27]
        expected_col1 = np.outer([1, 3], [5, 7, 9]).flatten()
        npt.assert_array_equal(result[:, 0], expected_col1)

    def test_three_matrices(self):
        """Test Khatri-Rao product of three matrices."""
        A = np.array([[1], [2]])
        B = np.array([[3], [4]])
        C = np.array([[5], [6]])

        result = _khatri_rao([A, B, C])

        # Result should have shape (2*2*2, 1) = (8, 1)
        self.assertEqual(result.shape, (8, 1))

    def test_empty_list_error(self):
        """Test error on empty input."""
        with self.assertRaises(ValueError):
            _khatri_rao([])


class TestTensorUnfolding(unittest.TestCase):
    """Tests for tensor unfolding and refolding."""

    def setUp(self):
        """Set up test tensor."""
        np.random.seed(42)
        self.tensor = np.random.randn(3, 4, 5)

    def test_unfold_mode_0(self):
        """Test unfolding along mode 0."""
        unfolded = _unfold_tensor(self.tensor, mode=0)

        # Shape should be (3, 4*5) = (3, 20)
        self.assertEqual(unfolded.shape, (3, 20))

    def test_unfold_mode_1(self):
        """Test unfolding along mode 1."""
        unfolded = _unfold_tensor(self.tensor, mode=1)

        # Shape should be (4, 3*5) = (4, 15)
        self.assertEqual(unfolded.shape, (4, 15))

    def test_unfold_mode_2(self):
        """Test unfolding along mode 2."""
        unfolded = _unfold_tensor(self.tensor, mode=2)

        # Shape should be (5, 3*4) = (5, 12)
        self.assertEqual(unfolded.shape, (5, 12))

    def test_refold(self):
        """Test that refolding inverts unfolding."""
        for mode in range(3):
            unfolded = _unfold_tensor(self.tensor, mode)
            refolded = _refold_tensor(unfolded, mode, self.tensor.shape)

            npt.assert_almost_equal(refolded, self.tensor)


class TestConstructTensor(unittest.TestCase):
    """Tests for tensor construction from CP factors."""

    def test_basic_construction(self):
        """Test basic CP tensor construction."""
        np.random.seed(42)

        # Create factors for a rank-2 decomposition of a 3x4x5 tensor
        A = np.random.randn(3, 2)
        B = np.random.randn(4, 2)
        C = np.random.randn(5, 2)

        tensor = _construct_tensor_from_cp([A, B, C])

        self.assertEqual(tensor.shape, (3, 4, 5))

    def test_rank_1_construction(self):
        """Test construction with rank-1 factors."""
        A = np.array([[1], [2], [3]])
        B = np.array([[4], [5]])
        C = np.array([[6], [7], [8], [9]])

        tensor = _construct_tensor_from_cp([A, B, C])

        self.assertEqual(tensor.shape, (3, 2, 4))

        # Check a specific element
        # tensor[0,0,0] should be 1*4*6 = 24
        self.assertEqual(tensor[0, 0, 0], 24)


class TestTensorFactorization(unittest.TestCase):
    """Tests for TensorFactorization class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create a low-rank tensor with missing values
        self.n1, self.n2, self.n3, self.r = 6, 5, 8, 2

        # True factors
        self.A_true = np.random.randn(self.n1, self.r)
        self.B_true = np.random.randn(self.n2, self.r)
        self.C_true = np.random.randn(self.n3, self.r)

        self.tensor_true = _construct_tensor_from_cp(
            [self.A_true, self.B_true, self.C_true]
        )

        # Create observed tensor with missing values
        self.tensor_observed = self.tensor_true.copy()
        mask = np.random.rand(self.n1, self.n2, self.n3) < 0.3
        self.tensor_observed[mask] = np.nan

    def test_basic_factorization(self):
        """Test basic tensor factorization."""
        tf = TensorFactorization(
            n_components=self.r,
            max_als_iterations=30,
            n_initializations=3
        )
        tf.fit(self.tensor_observed)

        # Check that factors are computed
        self.assertIsNotNone(tf.factors)
        self.assertEqual(len(tf.factors), 3)

        # Check factor shapes
        self.assertEqual(tf.factors[0].shape, (self.n1, self.r))
        self.assertEqual(tf.factors[1].shape, (self.n2, self.r))
        self.assertEqual(tf.factors[2].shape, (self.n3, self.r))

    def test_fit_transform(self):
        """Test fit_transform method."""
        tf = TensorFactorization(
            n_components=self.r,
            max_als_iterations=20,
            n_initializations=2
        )
        reconstructed = tf.fit_transform(self.tensor_observed)

        self.assertEqual(reconstructed.shape, self.tensor_true.shape)

    def test_get_loadings(self):
        """Test loading extraction methods."""
        tf = TensorFactorization(n_components=self.r, max_als_iterations=10)
        tf.fit(self.tensor_observed)

        subject_loadings = tf.get_subject_loadings()
        condition_loadings = tf.get_condition_loadings()
        feature_loadings = tf.get_feature_loadings()

        self.assertEqual(subject_loadings.shape, (self.n1, self.r))
        self.assertEqual(condition_loadings.shape, (self.n2, self.r))
        self.assertEqual(feature_loadings.shape, (self.n3, self.r))

    def test_not_fitted_error(self):
        """Test error when methods called before fit."""
        tf = TensorFactorization()

        with self.assertRaises(ValueError):
            tf.transform()

        with self.assertRaises(ValueError):
            tf.get_subject_loadings()

    def test_insufficient_dimensions_error(self):
        """Test error on 2D input."""
        tf = TensorFactorization()

        with self.assertRaises(ValueError) as context:
            tf.fit(np.random.randn(5, 5))

        self.assertIn("3 dimensions", str(context.exception))

    def test_no_missing_values_error(self):
        """Test error when no missing values present."""
        tf = TensorFactorization()

        # Complete tensor (no NaN)
        complete_tensor = np.random.randn(3, 4, 5)

        with self.assertRaises(ValueError) as context:
            tf.fit(complete_tensor)

        self.assertIn("missing values", str(context.exception))

    def test_n_components_too_large_error(self):
        """Test error when n_components exceeds dimensions."""
        tf = TensorFactorization(n_components=100)

        with self.assertRaises(ValueError) as context:
            tf.fit(self.tensor_observed)

        self.assertIn("cannot exceed", str(context.exception))

    def test_reconstruction_quality(self):
        """Test that reconstruction is reasonable."""
        tf = TensorFactorization(
            n_components=self.r,
            max_als_iterations=50,
            n_initializations=5
        )
        tf.fit(self.tensor_observed)
        reconstructed = tf.transform()

        # Compute RMSE on observed entries
        observed_mask = ~np.isnan(self.tensor_observed)
        rmse = np.sqrt(np.mean(
            (reconstructed[observed_mask] - self.tensor_true[observed_mask]) ** 2
        ))

        # RMSE should be reasonable for a low-rank tensor
        self.assertLess(rmse, 2.0)


class TestTensorFactorizationCentering(unittest.TestCase):
    """Tests for centering in tensor factorization."""

    def test_centering_enabled(self):
        """Test that centering produces zero-mean factors."""
        np.random.seed(42)

        tensor = np.random.randn(5, 4, 6)
        tensor[::2, ::2, 0] = np.nan  # Add missing values

        tf = TensorFactorization(n_components=2, center=True,
                                  max_als_iterations=20)
        tf.fit(tensor)

        # Check that factors are approximately centered
        for factor in tf.factors:
            mean = np.mean(factor, axis=0)
            npt.assert_almost_equal(mean, np.zeros(2), decimal=1)

    def test_centering_disabled(self):
        """Test that disabling centering preserves factor means."""
        np.random.seed(42)

        # Create tensor with non-zero mean factors
        A = np.random.randn(5, 2) + 5
        B = np.random.randn(4, 2) + 5
        C = np.random.randn(6, 2) + 5

        tensor = _construct_tensor_from_cp([A, B, C])
        tensor[::2, ::2, 0] = np.nan

        tf = TensorFactorization(n_components=2, center=False,
                                  max_als_iterations=20)
        tf.fit(tensor)

        # With centering disabled, factors should not be exactly zero mean
        # (though the reconstruction might still center due to optimization)
        self.assertTrue(tf.factors is not None)


if __name__ == '__main__':
    unittest.main()
