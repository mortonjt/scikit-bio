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

from skbio.stats.ordination._rclr import matrix_rclr, tensor_rclr, _matrix_closure


class TestMatrixClosure(unittest.TestCase):
    """Tests for the matrix closure function."""

    def test_basic_closure(self):
        """Test basic normalization to sum 1."""
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        result = _matrix_closure(mat)

        # Each row should sum to 1
        npt.assert_almost_equal(result.sum(axis=1), [1.0, 1.0])

    def test_zero_row(self):
        """Test handling of zero rows."""
        mat = np.array([[0, 0, 0], [1, 2, 3]])
        result = _matrix_closure(mat)

        # First row should remain zeros
        npt.assert_almost_equal(result[0], [0.0, 0.0, 0.0])
        # Second row should sum to 1
        npt.assert_almost_equal(result[1].sum(), 1.0)

    def test_1d_input(self):
        """Test that 1D input is handled correctly."""
        vec = np.array([1, 2, 3])
        result = _matrix_closure(vec)

        self.assertEqual(result.ndim, 2)
        npt.assert_almost_equal(result.sum(), 1.0)


class TestMatrixRclr(unittest.TestCase):
    """Tests for the robust centered log-ratio transformation."""

    def test_basic_rclr(self):
        """Test basic rclr transformation."""
        # Simple matrix with no zeros
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        result = matrix_rclr(mat)

        # For each row, the mean of transformed values should be ~0
        # (only over observed values)
        for i in range(mat.shape[0]):
            observed = ~np.isnan(result[i])
            npt.assert_almost_equal(result[i, observed].mean(), 0.0)

    def test_rclr_with_zeros(self):
        """Test rclr handles zeros by producing NaN."""
        mat = np.array([[1, 0, 3], [4, 5, 0]])
        result = matrix_rclr(mat)

        # Zeros should become NaN
        self.assertTrue(np.isnan(result[0, 1]))
        self.assertTrue(np.isnan(result[1, 2]))

        # Non-zero positions should not be NaN
        self.assertFalse(np.isnan(result[0, 0]))
        self.assertFalse(np.isnan(result[0, 2]))

    def test_rclr_centering(self):
        """Test that rclr centers each row correctly."""
        mat = np.array([[1, 2, 0, 4], [0, 3, 3, 0], [2, 2, 2, 2]])
        result = matrix_rclr(mat)

        # For rows with observed values, mean should be 0
        for i in range(mat.shape[0]):
            observed = ~np.isnan(result[i])
            if np.any(observed):
                npt.assert_almost_equal(result[i, observed].mean(), 0.0)

    def test_rclr_negative_values_error(self):
        """Test that negative values raise an error."""
        mat = np.array([[1, -2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError) as context:
            matrix_rclr(mat)

        self.assertIn("negative", str(context.exception))

    def test_rclr_non_finite_error(self):
        """Test that non-finite values raise an error."""
        mat = np.array([[1, np.inf, 3], [4, 5, 6]])

        with self.assertRaises(ValueError) as context:
            matrix_rclr(mat)

        self.assertIn("non-finite", str(context.exception))

    def test_rclr_1d_input(self):
        """Test that 1D input is promoted to 2D."""
        vec = np.array([1, 2, 3])
        result = matrix_rclr(vec)

        self.assertEqual(result.ndim, 2)

    def test_rclr_uniform_row(self):
        """Test rclr on uniform row (all same values)."""
        mat = np.array([[2, 2, 2, 2]])
        result = matrix_rclr(mat)

        # Uniform row should have all zeros (log-ratio of equal values)
        npt.assert_almost_equal(result[0], [0.0, 0.0, 0.0, 0.0])

    def test_rclr_preserves_ratios(self):
        """Test that rclr preserves log-ratios between features."""
        mat = np.array([[1, 2, 4]])
        result = matrix_rclr(mat)

        # log(2) - log(1) = log(2)
        expected_ratio = np.log(2)
        observed_ratio = result[0, 1] - result[0, 0]
        npt.assert_almost_equal(observed_ratio, expected_ratio)


class TestTensorRclr(unittest.TestCase):
    """Tests for tensor rclr transformation."""

    def test_basic_tensor_rclr(self):
        """Test basic 3D tensor rclr."""
        tensor = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ])
        result = tensor_rclr(tensor)

        # Shape should be preserved
        self.assertEqual(result.shape, tensor.shape)

        # Each sample's mean should be approximately 0
        result_2d = result.reshape(-1, tensor.shape[-1])
        for i in range(result_2d.shape[0]):
            observed = ~np.isnan(result_2d[i])
            if np.any(observed):
                npt.assert_almost_equal(result_2d[i, observed].mean(), 0.0,
                                         decimal=5)

    def test_tensor_rclr_with_zeros(self):
        """Test tensor rclr handles zeros."""
        tensor = np.array([
            [[1, 0, 3], [0, 5, 6]],
            [[7, 8, 0], [10, 0, 12]]
        ])
        result = tensor_rclr(tensor)

        # Zeros should become NaN
        self.assertTrue(np.isnan(result[0, 0, 1]))
        self.assertTrue(np.isnan(result[0, 1, 0]))
        self.assertTrue(np.isnan(result[1, 0, 2]))
        self.assertTrue(np.isnan(result[1, 1, 1]))

    def test_tensor_rclr_preserves_shape(self):
        """Test that various tensor shapes are preserved."""
        shapes = [(2, 3, 4), (5, 2, 6), (3, 3, 3)]

        for shape in shapes:
            tensor = np.random.rand(*shape) + 0.1  # Avoid zeros
            result = tensor_rclr(tensor)
            self.assertEqual(result.shape, shape)

    def test_tensor_rclr_negative_error(self):
        """Test that negative values raise an error."""
        tensor = np.array([[[1, -2, 3]]])

        with self.assertRaises(ValueError) as context:
            tensor_rclr(tensor)

        self.assertIn("negative", str(context.exception))


if __name__ == '__main__':
    unittest.main()
