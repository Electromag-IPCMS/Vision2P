Non-Negative Matrix Factorization (NMF)
======================================

.. py:function:: Vision2P.nmf.nmf(X, W=None, H=None, n_components=4, init='random', update_W=True, update_H=True, solver='mu', tol=0.0001, max_iter=50, random_state=None, polarimetry_reg=0, constrained_components=None)

   Perform Non-negative Matrix Factorization (NMF) on the input matrix X.

   The cost function is:

   .. math::

      X = \|X - WH\|^2 + \alpha \|H - FH\|^2

   with :math:`\alpha` corresponding to the parameter *polarimetry_reg* and:

   .. math::

      F = \begin{bmatrix}
         0 & I_n \\
         I_n & 0
      \end{bmatrix}

   where *n* is half the number of features. The number of features must be even if *polarimetry_reg* is not zero.

   :param X: The input data matrix to be factorized.
   :type X: array-like, shape (n_samples, n_features)

   :param W: Initial guess for the matrix W. If None, it will be initialized based on *init*.
   :type W: array-like, shape (n_samples, n_components), optional

   :param H: Initial guess for the matrix H. If None, it will be initialized based on *init*.
   :type H: array-like, shape (n_components, n_features), optional

   :param n_components: The number of components (latent features) to extract. Default is 4.
   :type n_components: int

   :param init: Initialization method for W and H. Options:
      - 'random'
      - 'nndsvd'
      - 'nndsvda'
      - 'nndsvdar'
   :type init: {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}

   :param update_W: Whether to update matrix W during optimization. Default is True.
   :type update_W: bool

   :param update_H: Whether to update matrix H during optimization. Default is True.
   :type update_H: bool

   :param solver: Solver to use for optimization:
      - 'mu': Multiplicative update
      - 'cd': Coordinate descent
   :type solver: {'mu', 'cd'}

   :param tol: Tolerance for the stopping condition. Default is 1e-4.
   :type tol: float

   :param max_iter: Maximum number of iterations. Default is 50.
   :type max_iter: int

   :param random_state: Random seed or RandomState instance for reproducibility. Default is None.
   :type random_state: int or RandomState or None

   :param polarimetry_reg: Regularization parameter for polarimetry with linearly polarized light. Default is 0.
   :type polarimetry_reg: float

   :param constrained_components: List of components in H that should not be updated. Default is None.
   :type constrained_components: array-like or None

   :returns: 
      - **H** (*array-like, shape (n_components, n_features)*): The learned coefficient matrix.
      - **W** (*array-like, shape (n_samples, n_components)*): The learned basis matrix.
      - **reconstruction_err** (*float*): Final reconstruction error.
