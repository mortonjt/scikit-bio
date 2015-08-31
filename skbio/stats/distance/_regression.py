# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import copy
import numpy as np
from skbio.util._decorator import experimental
from skbio.util._misc import check_random_state
from skbio.stats.distance._mantel import _order_dms



@experimental(as_of="0.4.0")
def linregress(y, *args, **kwargs):
    """
    Performs a linear regression on distance matrices using
    permutation of residuals

    Parameters
    ----------
    x1, x2, ... : skbio.DistanceMatrix
        Predictor distance matrices
    y : skbio.DistanceMatrix
        Response distance matrix
    permutations : int, optional
        Number of permutations to perform.  If no permutations
       are specified, then the residuals are assumed to be normally
        distributed
    random_state: int or np.RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    B : np.array
        Array of coefficients
    T : np.array
        Array of t-statistics for each coefficient
    pvals : np.array
        Array of p-values for each coefficient
    F : float
        pseudo F-statistic for overall model
    model_pval : float
        pvalue for overal model
    R2: float
        Coefficient of determination squared

    See Also
    --------
    DistanceMatrix

    References
    ----------
    .. [1] Legendre, P. and Legendre, L. (2012) Numerical Ecology. 3rd English
       Edition. Elsevier.
    .. [2] Legendre, P. Lapointe, F., Casgrain P. (1994) Modeling Brain
       Evolution from Behavior: A Permutational Regression Approach

    """

    # Unpackage kwargs
    params = {'permutations':10000,
              'random_state':0,
              'strict':True}
    for key in ('permutations', 'random_state'):
        params[key] = kwargs.get(key, params[key])
    permutations = params['permutations']
    random_state = params['random_state']
    strict = params['strict']

    random_state = check_random_state(random_state)

    # Conform all of the ids in the distance matrices to the same order
    xargs = copy.deepcopy(args)
    if strict:
        for i in range(len(xargs)):
            y, xargs[i] = _order_dms(y, xargs[i])

    # Linearize all predictor distance matrices into
    # a single matrix
    n = len(y.data)
    X = np.vstack([np.ones((1, n*(n-1)/2))] + \
                  [x.data[np.triu_indices(n, 1)] for x in xargs]).T
    Y = np.atleast_2d(y[np.triu_indices(n, 1)]).T
    n, p = X.shape
    J = np.ones((n, n))
    I = np.identity(n)

    # Permutation on residuals
    def regress(Y, X, computeR=False):
        XX1 = np.linalg.pinv(X.T.dot(X))
        B = XX1.dot(X.T.dot(Y))
        H = X.dot(XX1).dot(X.T)
        Yhat = H.dot(Y)

        SSE = Y.T.dot(I - H).dot(Y)
        SSR = Y.T.dot(H - (1./n)*J).dot(Y)
        dfe = n - p
        dfr = p - 1
        MSR = SSR / dfr
        MSE = SSE / dfe

        T = np.ravel(B) / np.sqrt(np.diag(XX1) * MSE)
        F = MSR / MSE
        if computeR:
            SST = Y.T.dot(I - (1./n)*J).dot(Y)
            R2 = SSR / SST
        else:
            R2 = None
        return Yhat, B, T, F, R2

    Yhat, B, T, F, R2 = regress(Y, X, computeR=True)
    E = Y - Yhat
    Fs = np.zeros(permutations)
    Ts = np.zeros((permutations, p))
    for i in range(permutations):
        np.random.shuffle(E)
        Ynew = Yhat + E
        Yhat_, B_, T_, F_, _ = regress(Ynew, X, computeR=False)
        Ts[i, :] = T_
        Fs[i] = F_

    pvals = ((abs(T) >= abs(Ts)).sum(axis=0) + 1) / (permutations + 1)
    model_pval = ((F >= Fs).sum() + 1) / (permutations + 1)

    return (np.ravel(B),
            np.ravel(T),
            pvals,
            np.asscalar(F),
            np.asscalar(model_pval),
            np.asscalar(R2))
