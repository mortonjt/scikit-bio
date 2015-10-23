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
perturbation and power can be used to manipulate this data.

This module allows two styles of manipulation of compositional data.
Compositional data can be analyzed using perturbation and power
operations, which can be useful for simulation studies. The
alternative strategy is to transform compositional data into the real
space.  Right now, the centre log ratio transform (clr) and
the isometric log ratio transform (ilr) [2]_ can be used to accomplish
this. This transform can be useful for performing standard statistical
tools such as parametric hypothesis testing, regressions and more.

The major caveat of using this framework is dealing with zeros.  In
the Aitchison geometry, only compositions with nonzero components can
be considered. The multiplicative replacement technique [3]_ can be
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
   inner
   clr
   clr_inv
   ilr
   ilr_inv
   centralize
   ancom

References
----------
.. [1] V. Pawlowsky-Glahn, "Lecture Notes on Compositional Data Analysis"
   (2007)

.. [2] J. J. Egozcue.,  "Isometric Logratio Transformations for
   Compositional Data Analysis" Mathematical Geology, 35.3 (2003)

.. [3] J. A. Martin-Fernandez,  "Dealing With Zeros and Missing Values in
   Compositional Data Sets Using Nonparametric Imputation",
   Mathematical Geology, 35.3 (2003)


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

    .. math::
        x \oplus y = C[x_1 y_1, \ldots, x_D y_D]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

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

    .. math::
        x \ominus y = C[x_1 y_1^{-1}, \ldots, x_D y_D^{-1}]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]


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

    .. math::
        `x \odot a = C[x_1^a, \ldots, x_D^a]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

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
def inner(x, y):
    r"""
    Calculates the Aitchson inner product.

    This inner product is defined as follows

    .. math::
        \langle x, y \rangle_a =
        \frac{1}{2D} \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{D}
        \ln\left(\frac{x_i}{x_j}\right) \ln\left(\frac{y_i}{y_j}\right)

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
    numpy.ndarray
         inner product result

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import inner
    >>> x = np.array([.1, .3, .4, .2])
    >>> y = np.array([.2, .4, .2, .2])
    >>> inner(x, y)
    0.21078524737545556
    """
    x = closure(x)
    y = closure(y)
    a, b = clr(x), clr(y)
    return a.dot(b.T)


@experimental(as_of="0.4.0")
def clr(mat):
    r"""
    Performs centre log ratio transformation.

    This function transforms compositions from Aitchison geometry to
    the real space. The :math:`clr` transform is both an isometry and an
    isomorphism defined on the following spaces

    :math:`clr: S^D \rightarrow U`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    It is defined for a composition :math:`x` as follows:

    .. math::
        clr(x) = \ln\left[\frac{x_1}{g_m(x)}, \ldots, \frac{x_D}{g_m(x)}\right]

    where :math:`g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D}` is the geometric
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
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr(x)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])

    """
    mat = closure(mat)
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()


@experimental(as_of="0.4.0")
def clr_inv(mat):
    r"""
    Performs inverse centre log ratio transformation.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`clr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`clr^{-1}: U \rightarrow S^D`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    This transformation is defined as follows

    .. math::
        clr^{-1}(x) = C[\exp( x_1, \ldots, x_D)]

    Parameters
    ----------
    mat : array_like, float
       a matrix of real values where
       rows = transformed compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         inverse clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import clr_inv
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr_inv(x)
    array([ 0.21383822,  0.26118259,  0.28865141,  0.23632778])

    """
    return closure(np.exp(mat))


@experimental(as_of="0.4.0")
def ilr(mat, basis=None, check=True):
    r"""
    Performs isometric log ratio transformation.

    This function transforms compositions from Aitchison simplex to
    the real space. The :math: ilr` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr: S^D \rightarrow \mathbb{R}^{D-1}`

    The ilr transformation is defined as follows

    .. math::
        ilr(x) =
        [\langle x, e_1 \rangle_a, \ldots, \langle x, e_{D-1} \rangle_a]

    where :math:`[e_1,\ldots,e_{D-1}]` is an orthonormal basis in the simplex.

    If an orthornormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.

    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components

    basis: numpy.ndarray, float, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import ilr
    >>> x = np.array([.1, .3, .4, .2])
    >>> ilr(x)
    array([-0.7768362 , -0.68339802,  0.11704769])

    """
    mat = closure(mat)
    if basis is None:
        basis = clr_inv(_gram_schmidt_basis(mat.shape[-1]))
    elif check:
        _check_orthogonality(basis)
    return inner(mat, basis)


@experimental(as_of="0.4.0")
def ilr_inv(mat, basis=None, check=True):
    r"""
    Performs inverse isometric log ratio transform.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`ilr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr^{-1}: \mathbb{R}^{D-1} \rightarrow S^D`

    The inverse ilr transformation is defined as follows

    .. math::
        ilr^{-1}(x) = \bigoplus\limits_{i=1}^{D-1} x \odot e_i

    where :math:`[e_1,\ldots, e_{D-1}]` is an orthonormal basis in the simplex.

    If an orthornormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.


    Parameters
    ----------
    mat: numpy.ndarray, float
       a matrix of transformed proportions where
       rows = compositions and
       columns = components

    basis: numpy.ndarray, float, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import ilr
    >>> x = np.array([.1, .3, .6,])
    >>> ilr_inv(x)
    array([ 0.34180297,  0.29672718,  0.22054469,  0.14092516])

    """

    if basis is None:
        basis = _gram_schmidt_basis(mat.shape[-1] + 1)
    elif check:
        _check_orthogonality(basis)
    return clr_inv(np.dot(mat, basis))


@experimental(as_of="0.4.0")
def centralize(mat):
    r"""Center data around its geometric average.

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


@experimental(as_of="0.4.0-dev")
def ancom(table, grouping,
          alpha=0.05,
          tau=0.02,
          theta=0.1,
          multiple_comparisons_correction=None,
          significance_test=None):
    r""" Performs a differential abundance test using ANCOM

    This is done by calculating pairwise log ratios between all features
    and performing a signficance test to determine if there is a significant
    difference in feature ratios with respect to the variable of interest.

    In an experiment with only two treatments, this test tests the following
    hypothesis for feature :math:`i`

    .. math::

        H_{0i}: \mathbb{E}[\ln(u_i^{(1)})] = \mathbb{E}[\ln(u_i^{(2)})]

    where :math:`u_i^{(1)}` is the mean abundance for feature :math:`i` in the
    first group and :math:`u_i^{(2)}` is the mean abundance for feature
    :math:`i` in the second group.

    This method can be extended to an arbitrary number of classes by passing
    in a different statistical function.

    Parameters
    ----------
    table : pd.DataFrame
       A 2D matrix of counts or proportions where the rows correspond to
       samples and the columns correspond to features
    grouping : pd.Series
       Vector of group categories
    alpha : float, optional
       Significance level for each of the statistical tests
    tau : float, optional
       A constant used to determine an appropriate cutoff
    theta : float, optional
       Lower bound for the proportion of differential features.
       This can can be anywhere between 0 and 1.
    multiple_comparisons_correction : {None, 'holm-bonferroni'}, optional
       The multiple comparison correction procedure to run.  By default
       `scipy.stats.f_oneway` is used.
    significance_test : function, optional
       A statistical signficance function to test for signficance between
       classes.

    Returns
    -------
    pd.DataFrame
        A table of features, their W-statistics and whether the null hypothesis
        is rejected.

        `"W"` is the W-statistic, or number of features that a single feature
        is tested to be significantly different against.

        `"reject"` indicates if feature is significantly different or not

    See Also
    --------
    multiplicative_replacement

    Notes
    -----
    This method cannot handle any zero counts as input, since the logarithm
    of zero cannot be computed.  While this is an unsolved problem, many
    studies have shown promising results by replacing the zeros with pseudo
    counts.

    References
    ----------
    .. [1] Mandal et al. "Analysis of composition of microbiomes: a novel
       method for studying microbial composition", Microbial Ecology in Health
       & Disease, (2015), 26

    Examples
    --------
    First import all of the necessary modules:

    >>> from skbio.stats.composition import ancom
    >>> import pandas as pd

    Now let's load in a pd.DataFrame with sample and feature ids
    for our data matrix:

    >>> table = pd.DataFrame([[10., 11., 10., 10., 10., 10., 10.],
    ...                       [10.5, 11.5, 10.5, 10.5, 10.5, 10.5, 10.5],
    ...                       [10., 11., 10., 10., 10., 10., 10.],
    ...                       [20., 21., 10., 10., 10., 10., 10.],
    ...                       [20.5, 21.5, 10.5, 10.5, 10.5, 10.5, 10.5],
    ...                       [20.3, 21.3, 10.2, 10.3, 10.1, 10.6, 10.4]],
    ...                      index=['s1','s2','s3','s4','s5','s6'],
    ...                      columns=['b1','b2','b3','b4','b5','b6','b7'])

    Then create a grouping vector.  In this scenario, there
    are only two classes, so the first three samples fall under the first
    class while the last three samples fall under the second class:

    >>> grouping = pd.Series([0, 0, 0, 1, 1, 1],
    ...                      index=['s1','s2','s3','s4','s5','s6'])

    Now run `ancom` and see if there are any features that have any
    significant differences:

    >>> results = ancom(table, grouping)
    >>> results['W']
    b1    6
    b2    6
    b3    2
    b4    2
    b5    2
    b6    2
    b7    2
    Name: W, dtype: int64

    The W-statistic is the number of features that a single feature is
    tested to be significantly different against.  In this scenario
    there are b1 was significantly different from all of the other features:

    >>> results['reject']
    b1     True
    b2     True
    b3    False
    b4    False
    b5    False
    b6    False
    b7    False
    Name: reject, dtype: bool

    Here, there were only two features that estimated to
    significantly change between the two groups.

    """
    if len(table) != len(grouping):
        raise ValueError('The number of samples in table must be equal '
                         '(%d samples) to the number of samples in '
                         'grouping (%d samples).' % (len(table),
                                                     len(grouping)))

    if multiple_comparisons_correction is not None:
        if multiple_comparisons_correction != 'holm-bonferroni':
            raise ValueError('%s is not an available option'
                             % multiple_comparisons_correction)

    if significance_test is None:
        significance_test = scipy.stats.f_oneway

    mat = _check_composition(table, ignore_zeros=False)
    cats = pd.Series(grouping)

    mat.sort_index(inplace=True)
    cats = cats.sort_index()
    labs = mat.columns

    n_samp, n_feat = mat.shape

    _logratio_mat = _log_compare(mat.values, cats.values, significance_test)
    logratio_mat = _logratio_mat + _logratio_mat.T

    # Multiple comparisons
    if multiple_comparisons_correction == 'holm-bonferroni':
        logratio_mat = np.apply_along_axis(_holm_bonferroni,
                                           1, logratio_mat)
    np.fill_diagonal(logratio_mat, 1)
    W = (logratio_mat < alpha).sum(axis=1)
    c_start = W.max() / n_feat
    if c_start < theta:
        reject = np.zeros_like(W, dtype=bool)
    else:
        # Select appropriate cutoff
        cutoff = c_start - np.linspace(0.05, 0.25, 5)
        prop_cut = np.array([(W > n_feat*cut).mean() for cut in cutoff])
        dels = np.abs(prop_cut - np.roll(prop_cut, -1))
        dels[-1] = 0
        if (dels[1] < tau) and (dels[2] < tau) and (dels[3] < tau):
            nu = cutoff[1]
        elif (dels[1] >= tau) and (dels[2] < tau) and (dels[3] < tau):
            nu = cutoff[2]
        elif (dels[2] >= tau) and (dels[3] < tau) and (dels[4] < tau):
            nu = cutoff[3]
        else:
            nu = cutoff[4]
        reject = (W >= nu*n_feat)
    return pd.DataFrame({'W': pd.Series(W, index=labs),
                         'reject': pd.Series(reject, index=labs)})


def _check_composition(x, ignore_zeros=True):
    """ Checks to make sure that composition meets the mininum criteria

    Also casts composition into a pandas dataframe

    Parameters
    ----------
    x : array_like or pd.DataFrame
       Input composition matrix
       where rows=samples and columns=features
    ignore_zeros : bool
       Specifies if compositions are allowed to have zeros

    Returns
    -------
    pd.DataFrame
       Validated composition matrix
    """
    if x.ndim > 2:
        raise ValueError("Input matrix can onlx have two dimensions or less")
    if isinstance(x, pd.DataFrame):
        mat = closure(x)
        samp_labs = x.index
        feat_labs = x.columns
    else:
        mat = np.atleast_2d(x)
        r, c = mat.shape
        samp_labs = np.arange(r)
        feat_labs = np.arange(c)
    if np.any(mat == 0) and not ignore_zeros:
        raise ValueError('Cannot handle zeros in compositions. '
                         'Make sure to run a multiplicative_replacement')
    return pd.DataFrame(mat, index=samp_labs, columns=feat_labs)


def _holm_bonferroni(p):
    """ Performs Holm-Bonferroni correction for pvalues
    to account for multiple comparisons

    Parameters
    ---------
    p: numpy.array
        array of pvalues

    Returns
    -------
    numpy.array
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
                 significance_test=scipy.stats.ttest_ind):
    """ Calculates pairwise log ratios between all features and performs a
    permutation tests to determine if there is a significant difference in
    feature ratios with respect to the variable of interest

    Parameters
    ----------
    mat: np.array
       rows correspond to samples and columns correspond to
       features (i.e. OTUs)
    cats: np.array, float
       Vector of categories
    significance_test: function
        statistical test to run

    Returns:
    --------
    log ratio pvalue matrix
    """
    r, c = mat.shape
    log_ratio = np.zeros((c, c))
    log_mat = np.log(mat)
    cs = np.unique(cats)

    def func(x):
        return significance_test(*[x[cats == k] for k in cs])

    for i in range(c-1):
        ratio = (log_mat[:, i].T - log_mat[:, i+1:].T).T
        m, p = np.apply_along_axis(func,
                                   axis=0,
                                   arr=ratio)
        log_ratio[i, i+1:] = np.squeeze(np.array(p.T))
    return log_ratio


def _gram_schmidt_basis(n):
    """
    Builds clr transformed basis derived from
    gram schmidt orthogonalization

    Parameters
    ----------
    n : int
        Dimension of the Aitchison simplex
    """
    basis = np.zeros((n, n-1))
    for j in range(n-1):
        i = j + 1
        e = np.array([(1/i)]*i + [-1] +
                     [0]*(n-i-1))*np.sqrt(i/(i+1))
        basis[:, j] = e
    return basis.T


def _check_orthogonality(basis):
    """
    Checks to see if basis is truly orthonormal in the
    Aitchison simplex

    Parameters
    ----------
    basis: numpy.ndarray
        basis in the Aitchison simplex
    """
    if not np.allclose(inner(basis, basis), np.identity(len(basis)),
                       rtol=1e-4, atol=1e-6):
        raise ValueError("Aitchison basis is not orthonormal")
