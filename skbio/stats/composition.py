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
space.  Right now, the centre log ratio transform (clr) [1]_ and
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

References
----------
.. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"
.. [2] J. J. Egozcue "Isometric Logratio Transformations for
       Compositional Data Analysis"
.. [3] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
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
import scipy.stats as ss


def closure(mat):
    """
    Performs closure [1]_ to ensure that all elements add up to 1.

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

    References
    ----------
    .. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"

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


def perturb(x, y):
    r"""
    Performs the perturbation operation [1]_.

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

    References
    ----------
    .. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"

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


def perturb_inv(x, y):
    r"""
    Performs the inverse perturbation operation [1]_.

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

    References
    ----------
    .. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"

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


def power(x, a):
    r"""
    Performs the power operation [1]_.

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

    References
    ----------
    .. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"

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


def inner(x, y):
    """
    Calculates the Aitchson inner product [1]_.

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

    References
    ----------
    .. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"

    """
    x = closure(x)
    y = closure(y)
    D1 = x.shape[-1]
    D2 = y.shape[-1]
    if D1 != D2:
        raise ValueError("Compositions must have the same dimensions")
    D = D1
    M = np.ones((D, D))*-1 + np.identity(D)*D
    a = clr(x)
    b = clr(y).T
    return np.dot(np.dot(a, M), b) / D


def clr(mat):
    r"""
    Performs centre log ratio transformation [1]_.

    This function transforms compositions from Aitchison geometry to
    the real space. The :math:`clr` transform is both an isometry and an
    isomorphism defined on the following spaces

    :math:`clr: S^D \rightarrow U`
    where :math:`U=\{x :\sum_{i}^D x = 0 \qquad for x \in \mathbb{R}^D\}`

    It is defined for a composition :math:`x` as follows:

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

    References
    ----------
    .. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"

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


def clr_inv(mat):
    """
    Performs inverse centre log ratio transformation [1]_.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`clr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`clr^{-1}: U \rightarrow S^D`

    where :math:`U=\{x :\sum_{i}^D x = 0 \qquad for x \in \mathbb{R}^D\}`

    This transformation is defined as follows

    :math:`clr^{-1}(x) = C[exp( x_1, ..., x_D)]`

    Parameters
    ----------
    mat : numpy.ndarray, float
       a matrix of real values where
       rows = transformed compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         inverse clr transformed matrix

    References
    ----------
    .. [1] V. Pawlowsky-Glahn. "Lecture Notes on Compositional Data Analysis"

    """
    return closure(np.exp(mat))


def ilr(mat, basis=None, check=True):
    """
    Performs isometric log ratio transformation [1]_.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math: ilr` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr: S^D \rightarrow \mathbb{R}^{D-1}`

    The ilr transformation is defined as follows

    :math:`\[ilr(x) = [ \langle x, e_1 \rangle, ... ,
              \langle x, e_{D-1} \rangle]\]`

    where :math:`[e_1,...,e_{D-1}` is an orthonormal basis in the simplex.

    If an orthornormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.

    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components

    basis: numpy.ndarray, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    References
    ----------
    .. [1] J. J. Egozcue "Isometric Logratio Transformations for
           Compositional Data Analysis"
    """
    mat = closure(mat)
    if basis is None:
        basis = _gram_schmidt_basis(mat.shape[-1])
    elif check:
        _check_orthogonality(basis)
    return np.dot(clr(mat), basis.T)


def ilr_inv(mat, basis=None, check=True):
    """
    Performs inverse isometric log ratio transform [1]_.

    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of transformed proportions where
       rows = compositions and
       columns = components

    basis: numpy.ndarray, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    References
    ----------
    .. [1] J. J. Egozcue "Isometric Logratio Transformations for
           Compositional Data Analysis"
    """

    if basis is None:
        basis = _gram_schmidt_basis(mat.shape[-1] + 1)
    elif check:
        _check_orthogonality(basis)
    return clr_inv(np.dot(mat, basis))


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
    cen = ss.gmean(mat, axis=0)
    return perturb_inv(mat, cen)


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
    for i in range(len(basis)):
        for j in range(i+1, len(basis)):
            if not np.allclose(inner(basis[i, :], basis[j, :]), 0,
                               rtol=1e-4, atol=1e-6):
                raise ValueError("Aitchison basis is not orthonormal")
