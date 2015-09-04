# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

from unittest import TestCase, main
import numpy as np
import numpy.testing as npt
from numpy.random import normal
import pandas as pd
from skbio.stats.composition import (closure, multiplicative_replacement,
                                     perturb, perturb_inv, power,
                                     clr, centralize, ancom, _holm)
import scipy.stats


class CompositionTests(TestCase):

    def setUp(self):
        self.data1 = np.array([[2, 2, 6],
                               [4, 4, 2]])
        self.data2 = np.array([2, 2, 6])

        self.data3 = np.array([[1, 2, 3, 0, 5],
                               [1, 0, 0, 4, 5],
                               [1, 2, 3, 4, 5]])
        self.data4 = np.array([1, 2, 3, 0, 5])
        self.data5 = [[2, 2, 6], [4, 4, 2]]
        self.data6 = [[1, 2, 3, 0, 5],
                      [1, 0, 0, 4, 5],
                      [1, 2, 3, 4, 5]]

        self.table1 = np.array([
            [10, 10.5, 10, 20, 20.5, 20.3],
            [11, 11.5, 11, 21, 21.5, 21.3],
            [10, 10.5, 10, 10, 10.5, 10.2],
            [10, 10.5, 10, 10, 10.5, 10.3],
            [10, 10.5, 10, 10, 10.5, 10.1],
            [10, 10.5, 10, 10, 10.5, 10.6],
            [10, 10.5, 10, 10, 10.5, 10.4]]).T
        self.cats1 = [0, 0, 0, 1, 1, 1]

        D, L = 40, 80
        np.random.seed(0)
        self.table2 = np.vstack((np.concatenate((normal(10, 1, D),
                                                 normal(200, 1, D))),
                                 np.concatenate((normal(20, 1, D),
                                                 normal(100000, 1, D))),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 np.concatenate((normal(20, 1, D),
                                                 normal(100000, 1, D))),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 normal(10, 1, L),
                                 normal(10, 1, L)))
        self.table2 = np.absolute(self.table2)
        self.table2 = self.table2.astype(np.int).T
        self.cats2 = np.array([0]*D + [1]*D)

        # Bad datasets
        self.bad1 = np.array([1, 2, -1])
        self.bad2 = np.array([[[1, 2, 3, 0, 5]]])
        self.bad3 = np.array([
            [10, 10, 10, 20, 20, 0],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10]]).T

        self.bad4 = np.array([
            [10, 10, 10, 20, 20, 1],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, -1],
            [10, 10, 10, 10, 10, 10]]).T

        self.bad5 = np.array([[
            [10, 10, 10, 20, 20, 1],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, -1],
            [10, 10, 10, 10, 10, 10]]]).T

    def test_closure(self):

        npt.assert_allclose(closure(self.data1),
                            np.array([[.2, .2, .6],
                                      [.4, .4, .2]]))
        npt.assert_allclose(closure(self.data2),
                            np.array([.2, .2, .6]))
        npt.assert_allclose(closure(self.data5),
                            np.array([[.2, .2, .6],
                                      [.4, .4, .2]]))
        with self.assertRaises(ValueError):
            closure(self.bad1)

        with self.assertRaises(ValueError):
            closure(self.bad2)

        # make sure that inplace modification is not occurring
        closure(self.data2)
        npt.assert_allclose(self.data2, np.array([2, 2, 6]))

    def test_perturb(self):
        pmat = perturb(closure(self.data1),
                       closure(np.array([1, 1, 1])))
        npt.assert_allclose(pmat,
                            np.array([[.2, .2, .6],
                                      [.4, .4, .2]]))

        pmat = perturb(closure(self.data1),
                       closure(np.array([10, 10, 20])))
        npt.assert_allclose(pmat,
                            np.array([[.125, .125, .75],
                                      [1./3, 1./3, 1./3]]))

        pmat = perturb(closure(self.data1),
                       closure(np.array([10, 10, 20])))
        npt.assert_allclose(pmat,
                            np.array([[.125, .125, .75],
                                      [1./3, 1./3, 1./3]]))

        pmat = perturb(closure(self.data2),
                       closure([1, 2, 1]))
        npt.assert_allclose(pmat, np.array([1./6, 2./6, 3./6]))

        pmat = perturb(closure(self.data5),
                       closure(np.array([1, 1, 1])))
        npt.assert_allclose(pmat,
                            np.array([[.2, .2, .6],
                                      [.4, .4, .2]]))

        with self.assertRaises(ValueError):
            perturb(closure(self.data5), self.bad1)

        # make sure that inplace modification is not occurring
        perturb(self.data2, [1, 2, 3])
        npt.assert_allclose(self.data2, np.array([2, 2, 6]))

    def test_power(self):
        pmat = power(closure(self.data1), 2)
        npt.assert_allclose(pmat,
                            np.array([[.04/.44, .04/.44, .36/.44],
                                      [.16/.36, .16/.36, .04/.36]]))

        pmat = power(closure(self.data2), 2)
        npt.assert_allclose(pmat, np.array([.04, .04, .36])/.44)

        pmat = power(closure(self.data5), 2)
        npt.assert_allclose(pmat,
                            np.array([[.04/.44, .04/.44, .36/.44],
                                      [.16/.36, .16/.36, .04/.36]]))

        with self.assertRaises(ValueError):
            power(self.bad1, 2)

        # make sure that inplace modification is not occurring
        power(self.data2, 4)
        npt.assert_allclose(self.data2, np.array([2, 2, 6]))

    def test_perturb_inv(self):
        pmat = perturb_inv(closure(self.data1),
                           closure([.1, .1, .1]))
        imat = perturb(closure(self.data1),
                       closure([10, 10, 10]))
        npt.assert_allclose(pmat, imat)
        pmat = perturb_inv(closure(self.data1),
                           closure([1, 1, 1]))
        npt.assert_allclose(pmat,
                            closure([[.2, .2, .6],
                                     [.4, .4, .2]]))
        pmat = perturb_inv(closure(self.data5),
                           closure([.1, .1, .1]))
        imat = perturb(closure(self.data1), closure([10, 10, 10]))
        npt.assert_allclose(pmat, imat)

        with self.assertRaises(ValueError):
            perturb_inv(closure(self.data1), self.bad1)

        # make sure that inplace modification is not occurring
        perturb_inv(self.data2, [1, 2, 3])
        npt.assert_allclose(self.data2, np.array([2, 2, 6]))

    def test_multiplicative_replacement(self):
        amat = multiplicative_replacement(closure(self.data3))
        npt.assert_allclose(amat,
                            np.array([[0.087273, 0.174545, 0.261818,
                                       0.04, 0.436364],
                                      [0.092, 0.04, 0.04, 0.368, 0.46],
                                      [0.066667, 0.133333, 0.2,
                                       0.266667, 0.333333]]),
                            rtol=1e-5, atol=1e-5)

        amat = multiplicative_replacement(closure(self.data4))
        npt.assert_allclose(amat,
                            np.array([0.087273, 0.174545, 0.261818,
                                      0.04, 0.436364]),
                            rtol=1e-5, atol=1e-5)

        amat = multiplicative_replacement(closure(self.data6))
        npt.assert_allclose(amat,
                            np.array([[0.087273, 0.174545, 0.261818,
                                       0.04, 0.436364],
                                      [0.092, 0.04, 0.04, 0.368, 0.46],
                                      [0.066667, 0.133333, 0.2,
                                       0.266667, 0.333333]]),
                            rtol=1e-5, atol=1e-5)

        with self.assertRaises(ValueError):
            multiplicative_replacement(self.bad1)
        with self.assertRaises(ValueError):
            multiplicative_replacement(self.bad2)

        # make sure that inplace modification is not occurring
        multiplicative_replacement(self.data4)
        npt.assert_allclose(self.data4, np.array([1, 2, 3, 0, 5]))

    def test_clr(self):
        cmat = clr(closure(self.data1))
        A = np.array([.2, .2, .6])
        B = np.array([.4, .4, .2])

        npt.assert_allclose(cmat,
                            [np.log(A / np.exp(np.log(A).mean())),
                             np.log(B / np.exp(np.log(B).mean()))])
        cmat = clr(closure(self.data2))
        A = np.array([.2, .2, .6])
        npt.assert_allclose(cmat,
                            np.log(A / np.exp(np.log(A).mean())))

        cmat = clr(closure(self.data5))
        A = np.array([.2, .2, .6])
        B = np.array([.4, .4, .2])

        npt.assert_allclose(cmat,
                            [np.log(A / np.exp(np.log(A).mean())),
                             np.log(B / np.exp(np.log(B).mean()))])
        with self.assertRaises(ValueError):
            clr(self.bad1)
        with self.assertRaises(ValueError):
            clr(self.bad2)

        # make sure that inplace modification is not occurring
        clr(self.data2)
        npt.assert_allclose(self.data2, np.array([2, 2, 6]))

    def test_centralize(self):
        cmat = centralize(closure(self.data1))
        npt.assert_allclose(cmat,
                            np.array([[0.22474487, 0.22474487, 0.55051026],
                                      [0.41523958, 0.41523958, 0.16952085]]))
        cmat = centralize(closure(self.data5))
        npt.assert_allclose(cmat,
                            np.array([[0.22474487, 0.22474487, 0.55051026],
                                      [0.41523958, 0.41523958, 0.16952085]]))

        with self.assertRaises(ValueError):
            centralize(self.bad1)
        with self.assertRaises(ValueError):
            centralize(self.bad2)

        centralize(self.data1)
        npt.assert_allclose(self.data1,
                            np.array([[2, 2, 6],
                                      [4, 4, 2]]))

    def test_ancom(self):
        W, reject = ancom(pd.DataFrame(self.table1),
                          pd.Series(self.cats1),
                          multicorr=False)
        npt.assert_allclose(W.values,
                            np.array([7., 7., 3., 3., 3., 3., 3.]))
        npt.assert_allclose(reject.values,
                            np.array([True, True, False, False,
                                      False, False, False], dtype=bool))
        W, reject = ancom(self.table1, self.cats1, multicorr=False)
        npt.assert_allclose(W.values,
                            np.array([7., 7., 3., 3., 3., 3., 3.]))
        npt.assert_allclose(reject.values,
                            np.array([True, True, False, False,
                                      False, False, False], dtype=bool))

        W, reject = ancom(self.table1,
                          self.cats1,
                          multicorr=True,
                          func=scipy.stats.mannwhitneyu)
        npt.assert_allclose(W.values,
                            np.array([1., 1., 1., 1., 1., 1., 1.]))
        npt.assert_allclose(reject.values,
                            np.array([True,  True,  True,  True,
                                      True,  True,  True], dtype=bool))

        W, reject = ancom(self.table1,
                          self.cats1,
                          multicorr=False,
                          func=scipy.stats.mannwhitneyu)
        npt.assert_allclose(W.values,
                            np.array([7., 7., 3., 3., 3., 3., 3.]))
        npt.assert_allclose(reject.values,
                            np.array([True,  True, False, False,
                                      False, False, False], dtype=bool))

        W, reject = ancom(self.table2,
                          self.cats2,
                          multicorr=False,
                          func=scipy.stats.mannwhitneyu)

        npt.assert_allclose(W.values,
                            np.array([9.,  9.,  4.,  4.,  9.,
                                      4.,  4.,  4.,  4.]))
        npt.assert_allclose(reject.values,
                            np.array([True, True, False, False,
                                      True, False, False, False, False],
                                     dtype=bool))

        with self.assertRaises(ValueError):
            ancom(self.bad3, self.cats2, multicorr=False)

        with self.assertRaises(ValueError):
            ancom(self.bad4, self.cats2, multicorr=False)

    def test_holm(self):
        p = [0.005, 0.011, 0.02, 0.04, 0.13]
        corrected_p = p * np.arange(1, 6)[::-1]
        guessed_p = _holm(p)
        for a, b in zip(corrected_p, guessed_p):
            self.assertAlmostEqual(a, b)

if __name__ == "__main__":
    main()
