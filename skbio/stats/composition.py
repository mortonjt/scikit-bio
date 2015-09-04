r"""
Composition Statistics (:mod:`skbio.stats.composition`)
=======================================================

.. currentmodule:: skbio.stats.composition

This module provides functions for compositional data analysis.

Many 'omics datasets are inherently compositional - meaning that they
are best interpreted as proportions or percentages rather than
absolute counts.

Formally, :math:`x` is a composition if :math:`\sum_{i=0}^D x_{i} = c`
and :math:`x_{i} > 0`, :math:`1 \leq i \leq D` and :math:`c` is a real
valued constant and there are :math:`D` components for each
composition. In this module :math:`c=1`. Compositional data can be
analyzed using Aitchison geometry. [1]_

However, in this framework, standard real Euclidean operations such as
addition and multiplication no longer apply. Only operations such as
perturbation and power can be used to manipulate this data. [1]_

This module allows two styles of manipulation of compositional data.
Compositional data can be analyzed using perturbation and power
operations, which can be useful for simulation studies. The
alternative strategy is to transform compositional data into the real
space.  Right now, the centre log ratio transform (clr) [1]_ can be
used to accomplish this.  This transform can be useful for performing
standard statistical tools such as parametric hypothesis testing,
regressions and more.

The major caveat of using this framework is dealing with zeros.  In
the Aitchison geometry, only compositions with nonzero components can
be considered. The multiplicative replacement technique [2]_ can be
used to substitute these zeros with small pseudocounts without
introducing major distortions to the data.

Functions
---------

.. autosummary::
   :toctree: generated/

   closure
   multiplicative_replacement
   perturb
   perturb_inv
   power
   clr
   centralize
   ancom

References
----------
.. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"
.. [2] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
       Compositional Data Sets Using Nonparametric Imputation"


Examples
--------

>>> import numpy as np

Consider a very simple environment with only 3 species. The species
in the environment are equally distributed and their proportions are
equivalent:

>>> otus = np.array([1./3, 1./3., 1./3])

Suppose that an antibiotic kills off half of the population for the
first two species, but doesn't harm the third species. Then the
perturbation vector would be as follows

>>> antibiotic = np.array([1./2, 1./2, 1])

And the resulting perturbation would be

>>> perturb(otus, antibiotic)
array([ 0.25,  0.25,  0.5 ])

"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import scipy.stats

from skbio.util._decorator import experimental


@experimental(as_of="0.4.0")
def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1.

    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components

    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])

    """
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()


@experimental(as_of="0.4.0")
def multiplicative_replacement(mat, delta=None):
    r"""Replace all zeros with small non-zero values

    It uses the multiplicative replacement strategy [1]_ ,
    replacing zeros with a small positive :math:`\delta`
    and ensuring that the compositions still add up to 1.


    Parameters
    ----------
    mat: array_like
       a matrix of proportions where
       rows = compositions and
       columns = components
    delta: float, optional
       a small number to be used to replace zeros
       If delta is not specified, then the default delta is
       :math:`\delta = \frac{1}{N^2}` where :math:`N`
       is the number of components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    References
    ----------
    .. [1] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
           Compositional Data Sets Using Nonparametric Imputation"


    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import multiplicative_replacement
    >>> X = np.array([[.2,.4,.4, 0],[0,.5,.5,0]])
    >>> multiplicative_replacement(X)
    array([[ 0.1875,  0.375 ,  0.375 ,  0.0625],
           [ 0.0625,  0.4375,  0.4375,  0.0625]])

    """
    mat = closure(mat)
    z_mat = (mat == 0)

    num_feats = mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1. / num_feats)**2

    zcnts = 1 - tot * delta
    mat = np.where(z_mat, delta, zcnts * mat)
    return mat.squeeze()


@experimental(as_of="0.4.0")
def perturb(x, y):
    r"""
    Performs the perturbation operation.

    This operation is defined as
    :math:`x \oplus y = C[x_1 y_1, ..., x_D y_D]`

    :math:`C[x]` is the closure operation defined as
    :math:`C[x] = [\frac{x_1}{\sum x},...,\frac{x_D}{\sum x}]`
    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import perturb
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> perturb(x,y)
    array([ 0.0625,  0.1875,  0.5   ,  0.25  ])

    """
    x, y = closure(x), closure(y)
    return closure(x * y)


@experimental(as_of="0.4.0")
def perturb_inv(x, y):
    r"""
    Performs the inverse perturbation operation.

    This operation is defined as
    :math:`x \ominus y = C[x_1 y_1^{-1}, ..., x_D y_D^{-1}]`

    :math:`C[x]` is the closure operation defined as
    :math:`C[x] = [\frac{x_1}{\sum x},...,\frac{x_D}{\sum x}]`
    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import perturb_inv
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> perturb_inv(x,y)
    array([ 0.14285714,  0.42857143,  0.28571429,  0.14285714])

    """
    x, y = closure(x), closure(y)
    return closure(x / y)


@experimental(as_of="0.4.0")
def power(x, a):
    r"""
    Performs the power operation.

    This operation is defined as follows
    :math:`x \odot a = C[x_1^a, ..., x_D^a]`

    :math:`C[x]` is the closure operation defined as
    :math:`C[x] = [\frac{x_1}{\sum x},...,\frac{x_D}{\sum x}]`
    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    a : float
        a scalar float

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import power
    >>> x = np.array([.1,.3,.4, .2])
    >>> power(x, .1)
    array([ 0.23059566,  0.25737316,  0.26488486,  0.24714631])

    """
    x = closure(x)
    return closure(x**a).squeeze()


@experimental(as_of="0.4.0")
def clr(mat):
    r"""
    Performs centre log ratio transformation.

    This function transforms compositions from Aitchison geometry to
    the real space. This transformation is an isometry, but not an
    isomorphism. It is defined for a composition :math:`x` as follows:

    :math:`clr(x) = ln[\frac{x_1}{g_m(x)}, ..., \frac{x_D}{g_m(x)}]`
    where :math:`g_m(x) = (\prod_{i=1}^{D} x_i)^{1/D}` is the geometric
    mean of :math:`x`.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import clr
    >>> x = np.array([.1,.3,.4, .2])
    >>> clr(x)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])

    """
    mat = closure(mat)
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()


@experimental(as_of="0.4.0")
def centralize(mat):
    """Center data around its geometric average.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         centered composition matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import centralize
    >>> X = np.array([[.1,.3,.4, .2],[.2,.2,.2,.4]])
    >>> centralize(X)
    array([[ 0.17445763,  0.30216948,  0.34891526,  0.17445763],
           [ 0.32495488,  0.18761279,  0.16247744,  0.32495488]])

    """
    mat = closure(mat)
    cen = scipy.stats.gmean(mat, axis=0)
    return perturb_inv(mat, cen)


@experimental(as_of="0.4.0")
def ancom(mat, cats,
          alpha=0.05,
          multicorr=False,
          tau=0.02,
          func=scipy.stats.ttest_ind):
    r"""
    Calculates pairwise log ratios between all otus
    and performs a signficance test to determine if there is a
    significant difference in feature ratios with respect to the
    variable of interest

    In an experiment with only two treatments, this test tests the
    following hypothesis for feature :math:`i`
    :math:`H_{0i}:\E[\ln(u_i^{(1)})] = \E[\ln(u_i^{(2)})]`

    where :math:`u_i^{(1)}` is the mean abundance for feature
    :math:`i` in the first group and :math:`u_i^{(2)}` is the
    mean abundance for feature :math:`i` in the second group.

    This method can be extended two an arbitrary number of classes
    by passing in a different statistical function.


    Parameters
    ----------
    mat: pd.DataFrame or array_like
       A 2D matrix where
       rows = samples
       columns = features
    cat: pd.Series or array_like
       Vector of categories
    multicorr: bool
       Runs multiple comparisons correction or not.
       If specified, this will run Holm-Boniferroni
       correction
    tau: float
       A constant used to determine an appropriate
       cutoff (@wdwvt1 can you comment here plz?)
    func: function
       A statistical signficance function to test for
       signficance between classes.
       The default is scipy.stats.ttest_ind

    Returns:
    --------
    W : pd.Series
        List of W statistics
    reject : pd.Series
        Indicates if the null hypothesis has been rejected

    References
    ----------
    ..[1] S. Mandal, 'Analysis of composition of microbiomes:
          a novel method for studying microbial composition'

    Examples
    --------
    First import all of the necessary modules

    >>> from skbio.stats.composition import ancom
    >>> import pandas as pd

    Now lets load in a pandas dataframe with sample and feature ids
    for our data matrix.
    >>> table = pd.DataFrame(
    ...     [[10., 11., 10., 10., 10., 10., 10.],
    ...      [10.5, 11.5, 10.5, 10.5, 10.5, 10.5, 10.5],
    ...      [10., 11., 10., 10., 10., 10., 10.],
    ...      [20., 21., 10., 10., 10., 10., 10.],
    ...      [20.5, 21.5, 10.5, 10.5, 10.5, 10.5, 10.5],
    ...      [20.3, 21.3, 10.2, 10.3, 10.1, 10.6, 10.4]],
    ...     index=['s1','s2','s3','s4','s5','s6'],
    ...     columns=['b1','b2','b3','b4','b5','b6','b7'])

    Then create a create a category vector.  In this scenerio, there
    are only two class, so the first three samples fall under the first
    class while the last three samples fall under the last class
    >>> cats = pd.Series([0, 0, 0, 1, 1, 1],
    ...                  index=['s1','s2','s3','s4','s5','s6'])

    Now run ancom and see if there are any features that have any
    significant differences
    >>> W, reject = ancom(table, cats)
    >>> print(W)
    b1    7
    b2    7
    b3    3
    b4    3
    b5    3
    b6    3
    b7    3
    dtype: float64
    >>> print(reject)
    b1     True
    b2     True
    b3    False
    b4    False
    b5    False
    b6    False
    b7    False
    dtype: bool
    """
    if len(mat) != len(cats):
        raise ValueError('The number of samples in mat needs'
                         'to be the same as the number of samples'
                         'in cats')
    mat = _check_composition(mat, ignore_zeros=False)
    cats = pd.Series(cats)

    mat = mat.sort_index()
    cats = cats.sort_index()
    labs = mat.columns

    mat = np.atleast_2d(mat.values)
    cats = np.array(cats.values)

    _logratio_mat = _log_compare(mat, cats, func)
    logratio_mat = _logratio_mat + _logratio_mat.T

    n_samp, n_feat = mat.shape
    # Multiple comparisons
    if multicorr:
        for i in range(n_feat):
            pvalues = _holm(logratio_mat[i, :])
            logratio_mat[i, :] = pvalues

    W = np.zeros(n_feat)
    for i in range(n_feat):
        W[i] = (logratio_mat[i, :] < alpha).sum()
    c_start = max(W)/n_feat
    cutoff = c_start - np.linspace(0.05, 0.25, 5)
    dels = np.zeros(len(cutoff))
    prop_cut = np.zeros(len(cutoff), dtype=np.float32)
    for cut in range(len(cutoff)):
        prop_cut[cut] = sum(W > n_feat*cutoff[cut])/len(W)
    for i in range(len(cutoff)-1):
        dels[i] = abs(prop_cut[i]-prop_cut[i+1])

    if (dels[1] < tau) and (dels[2] < tau) and (dels[3] < tau):
        nu = cutoff[1]
    elif (dels[1] >= tau) and (dels[2] < tau) and (dels[3] < tau):
        nu = cutoff[2]
    elif (dels[2] >= tau) and (dels[3] < tau) and (dels[4] < tau):
        nu = cutoff[3]
    else:
        nu = cutoff[4]
    reject = W >= nu*n_feat
    return pd.Series(W, index=labs), pd.Series(reject, index=labs)


def _check_composition(x, ignore_zeros=True):
    """
    Checks to make sure that composition meets the mininum criteria
    Also casts composition into a pandas dataframe

    Parameters
    ----------
    x : array_like or pd.DataFrame
       Input composition matrix
       where rows=samples and columns=features

    Returns
    -------
    pd.DataFrame
       Validated composition matrix
    """
    if x.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if isinstance(x, pd.DataFrame):
        mat = closure(x)
        samp_labs = x.index
        feat_labs = x.columns
    else:
        mat = np.atleast_2d(x)
        r, c = mat.shape
        samp_labs = range(r)
        feat_labs = range(c)
    if np.any(mat == 0) and not ignore_zeros:
        raise ValueError('Cannot handle zeros in compositions. '
                         'Make sure to run a zero replacement method')
    return pd.DataFrame(mat, index=samp_labs, columns=feat_labs)


def _holm(p):
    """
    Performs Holm-Boniferroni correction for pvalues
    to account for multiple comparisons

    Parameters
    ---------
    p: numpy.array
        array of pvalues

    Returns
    -------
    numpy.arrayy
        corrected pvalues
    """
    K = len(p)
    sort_index = -np.ones(K, dtype=np.int64)
    sorted_p = np.sort(p)
    sorted_p_adj = sorted_p*(K-np.arange(K))
    for j in range(K):
        idx = (p == sorted_p[j]) & (sort_index < 0)
        num_ties = len(sort_index[idx])
        sort_index[idx] = np.arange(j, (j+num_ties), dtype=np.int64)

    sorted_holm_p = [min([max(sorted_p_adj[:k]), 1])
                     for k in range(1, K+1)]
    holm_p = [sorted_holm_p[sort_index[k]] for k in range(K)]
    return holm_p


def _log_compare(mat, cats,
                 stat_func=scipy.stats.ttest_ind):
    """
    Calculates pairwise log ratios between all otus
    and performs a permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest

    Parameters
    ----------
    mat: np.array
       rows = samples
       columns = features (i.e. OTUs)
    cat: np.array, float
       Vector of categories
    stat_func: function
        statistical test to run

    Returns:
    --------
    log ratio pvalue matrix
    """
    r, c = mat.shape
    log_ratio = np.zeros((c, c))
    log_mat = np.log(mat)
    cs = np.unique(cats)
    for i in range(c-1):
        ratio = (log_mat[:, i].T - log_mat[:, i+1:].T).T
        def func(x):
            return stat_func(*[x[cats == k] for k in cs])
        m, p = np.apply_along_axis(func,
                                   axis=0,
                                   arr=ratio)
        log_ratio[i, i+1:] = np.squeeze(np.array(p.T))
    return log_ratio
