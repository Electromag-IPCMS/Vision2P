# Authors: Gael Varoquaux
#          Alexandre Gramfort
#          Alexandre T. Passos
#          Olivier Grisel
#          Lars Buitinck
#          Stefan van der Walt
#          Kyle Kastner
#          Giorgio Patrini
# License: BSD 3 clause

import warnings

import numpy as np
from scipy import linalg, sparse


def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """

    n_samples, n_features = X.shape

    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvda"
        else:
            init = "random"

    # Random initialization
    if init == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.standard_normal(size=(n_components, n_features)).astype(
            X.dtype, copy=False
        )
        W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(
            X.dtype, copy=False
        )
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = np.linalg.norm(x_p), np.linalg.norm(y_p)
        x_n_nrm, y_n_nrm = np.linalg.norm(x_n), np.linalg.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H


def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Faster than norm(x) ** 2.

    Parameters
    ----------
    x : array-like
        The input array which could be either be a vector or a 2 dimensional array.

    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    x = np.ravel(x, order="K")
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn(
            "Array type is integer, np.dot may overflow. "
            "Data should be float type to avoid this issue",
            UserWarning,
        )
    return np.dot(x, x)


def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.

    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret


def randomized_range_finder(
    A, *, size, n_iter, power_iteration_normalizer="auto", random_state=None
):
    """Compute an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D array
        The input data matrix.

    size : int
        Size of the return array.

    n_iter : int
        Number of power iterations used to stabilize the result.

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    Q : ndarray
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    :arxiv:`"Finding structure with randomness:
    Stochastic algorithms for constructing approximate matrix decompositions"
    <0909.4061>`
    Halko, et al. (2009)

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(A.shape[1], size))
    if hasattr(A, "dtype") and A.dtype.kind == "f":
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False)

    # Deal with "auto" mode
    if power_iteration_normalizer == "auto":
        if n_iter <= 2:
            power_iteration_normalizer = "none"
        else:
            power_iteration_normalizer = "LU"

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        if power_iteration_normalizer == "none":
            Q = safe_sparse_dot(A, Q)
            Q = safe_sparse_dot(A.T, Q)
        elif power_iteration_normalizer == "LU":
            Q, _ = linalg.lu(safe_sparse_dot(A, Q), permute_l=True)
            Q, _ = linalg.lu(safe_sparse_dot(A.T, Q), permute_l=True)
        elif power_iteration_normalizer == "QR":
            Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode="economic")
            Q, _ = linalg.qr(safe_sparse_dot(A.T, Q), mode="economic")

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode="economic")

    return Q


def randomized_svd(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose="auto",
    flip_sign=True,
    random_state=None,
    svd_lapack_driver="gesdd",
):
    """Compute a truncated randomized SVD.

    This method solves the fixed-rank approximation problem described in [1]_
    (problem (1.5), p5).

    Parameters
    ----------
    M : {ndarray, sparse matrix}
        Matrix to decompose.

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See [1]_
        (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see [1]_ page 9).

        .. versionchanged:: 0.18

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose : bool or 'auto', default='auto'
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : bool, default=True
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, default='warn'
        The seed of the pseudo random number generator to use when
        shuffling the data, i.e. getting the random vectors to initialize
        the algorithm. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

        .. versionchanged:: 1.2
            The default value changed from 0 to None.

    svd_lapack_driver : {"gesdd", "gesvd"}, default="gesdd"
        Whether to use the more efficient divide-and-conquer approach
        (`"gesdd"`) or more general rectangular approach (`"gesvd"`) to compute
        the SVD of the matrix B, which is the projection of M into a low
        dimensional subspace, as described in [1]_.

        .. versionadded:: 1.2

    Returns
    -------
    u : ndarray of shape (n_samples, n_components)
        Unitary matrix having left singular vectors with signs flipped as columns.
    s : ndarray of shape (n_components,)
        The singular values, sorted in non-increasing order.
    vh : ndarray of shape (n_components, n_features)
        Unitary matrix having right singular vectors with signs flipped as rows.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision). To increase the precision it is recommended to
    increase `n_oversamples`, up to `2*k-n_components` where k is the
    effective rank. Usually, `n_components` is chosen to be greater than k
    so increasing `n_oversamples` up to `n_components` should be enough.

    References
    ----------
    .. [1] :arxiv:`"Finding structure with randomness:
      Stochastic algorithms for constructing approximate matrix decompositions"
      <0909.4061>`
      Halko, et al. (2009)

    .. [2] A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    .. [3] An implementation of a randomized algorithm for principal component
      analysis A. Szlam et al. 2014

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import randomized_svd
    >>> a = np.array([[1, 2, 3, 5],
    ...               [3, 4, 5, 6],
    ...               [7, 8, 9, 10]])
    >>> U, s, Vh = randomized_svd(a, n_components=2, random_state=0)
    >>> U.shape, s.shape, Vh.shape
    ((3, 2), (2,), (2, 4))
    """
    if isinstance(M, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(M).__name__),
            sparse.SparseEfficiencyWarning,
        )

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == "auto":
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4

    if transpose == "auto":
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(
        M,
        size=n_random,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
    )

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, Vt = linalg.svd(B, full_matrices=False, lapack_driver=svd_lapack_driver)

    del B
    U = np.dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], Vt[:n_components, :]


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )