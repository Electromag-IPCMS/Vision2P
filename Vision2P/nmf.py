"""
Vision2P - A data vision software package that enables advanced AI and machine learning based analysis. It combines clustering and unmixing through a modified constrained Non-Negative Matrix Factorization (NMF) algorithm with advanced features tailored for spectromicroscopy applications.

Copyright © 2025 CNRS and Université de Strasbourg

Authors: Boris Croes and Salia Cherifi-Hertel

This program is free software: you can redistribute it and/or modify it under the terms of the 3-Clause BSD License. You should have received a copy of the 3-Clause BSD License along with this program.  If not, see <https://opensource.org/license/BSD-3-Clause>.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Contacts: 
Boris Croes, IPCMS Strasbourg, 23 rue du Loess,67034 Strasbourg, France. boris.croes@ipcms.unistra.fr
Salia Cherifi-Hertel, IPCMS Strasbourg, 23 rue du Loess,67034 Strasbourg, France. salia.cherifi@ipcms.unistra.fr
"""

import warnings
import numpy as np
from ._cdnmf_fast import _update_cdnmf_fast
from ._initialize_nmf import _initialize_nmf, squared_norm


def nmf(X,
            W = None,
            H = None,        
            n_components = 4,
            init = "random",
            update_W = True,
            update_H = True,
            solver = "mu",
            tol = 1e-4,
            max_iter = 50,
            random_state = None,
            polarimetry_reg = 0,
            constrained_components = None
            ):
    """
    Perform Non-negative Matrix Factorization (NMF) on the input matrix X.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data matrix to be factorized.

    W : array-like, shape (n_samples, n_components), optional
        Initial guess for the matrix W. If None, it will be initialized based on `init`.

    H : array-like, shape (n_components, n_features), optional
        Initial guess for the matrix H. If None, it will be initialized based on `init`.

    n_components : int, default=4
        The number of components (latent features) to extract.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default='random'
        Initialization method for W and H. Options include:
        - 'random': Random initialization.
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD).
        - 'nndsvda': NNDSVD with zeros filled with the average of X (NNDSVDA).
        - 'nndsvdar': NNDSVD with zeros filled with small random values (NNDSVDAR).

    update_W : bool, default=True
        Whether to update matrix W during the optimization process.

    update_H : bool, default=True
        Whether to update matrix H during the optimization process.

    solver : {'mu', 'cd'}, default='mu'
        The solver to use for optimization:
        - 'mu': Multiplicative update rule.
        - 'cd': Coordinate descent.

    tol : float, default=1e-4
        Tolerance for stopping condition.

    max_iter : int, default=50
        Maximum number of iterations to run the algorithm.

    random_state : int, RandomState instance, or None, default=None
        Seed or random state for reproducibility.

    polarimetry_reg : float, default=0
        Regularization parameter for polarimetry with linearly polarized light.

    constrained_components : array-like or None, default=None
        List of the components of the H matrix that will not be updated.

    Returns:
    --------
    W : array-like, shape (n_samples, n_components)
        The learned basis matrix.

    H : array-like, shape (n_components, n_features)
        The learned coefficient matrix.

    reconstruction_err : float
        The final reconstruction error.

    """
    np.random.seed(random_state) if random_state is not None else np.random.seed()

    if H is None or W is None or W.shape[1] < n_components or H.shape[0] < n_components:
        Wi, Hi = _initialize_nmf(X, n_components=n_components, init=init, eps=1e-6)
        W = W if W is not None else Wi
        H = H if H is not None else Hi
        if W.shape[1] < n_components or H.shape[0] < n_components:
            H = update_with_closest_components(H, Hi)
        del Wi, Hi
        
    if polarimetry_reg != 0:
        half_features = int(X.shape[1]/2)
        if half_features % 2 !=0: raise ValueError('Regularization for SHG only available for even numbers of features (measurements)')
        F = create_F_matrix(half_features)
        if solver=="cd": F = np.eye(*F.shape)-F
    else:
        F = None

    if constrained_components is not None:
        if H is None: warnings.warn("No input components (H matrix) from the user, the constrained components will be the initialized one (random or SVD)")
        if max(constrained_components) > n_components-1: warnings.warn("Highest constrained component is higher than the number of components, it will have no effect")
        components_to_update = [x for x in np.arange(n_components) if x not in constrained_components]
        components_to_update = np.asarray(components_to_update, dtype=np.intp)
    else: components_to_update = None

    if solver=="mu":
        H, W = update_mu(X, H, W, update_H, update_W, tol, max_iter, polarimetry_reg, F, components_to_update)
        
    if solver=="cd":
        Ht = np.array(H.T, order="C")
        H, W = update_cd(X, Ht, W, update_H, update_W, tol, max_iter, polarimetry_reg, n_components, F, components_to_update)

    return H, W


def update_mu(X, H, W, update_H, update_W, tol, max_iter, polarimetry_reg, F, components_to_update):

    error_at_init = frobenius_norm(X - np.dot(W, H))
    previous_error = error_at_init

    for i in range(max_iter):

        if update_H: H = update_mu_H(X, H, W, polarimetry_reg, F, components_to_update)
        if update_W: W*= np.divide(np.dot(X, H.T), np.linalg.multi_dot([W, H, H.T]))

        if tol > 0 and i % 10 == 0:
            error = frobenius_norm(X - np.dot(W, H))
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    return H, W


def update_mu_H(X, H, W, polarimetry_reg, F, components_to_update):

    H_numerator = np.dot(W.T, X)
    H_denominator = np.linalg.multi_dot([W.T, W, H])

    if polarimetry_reg != 0:
        H_numerator += polarimetry_reg * np.dot(H, F)
        H_denominator += polarimetry_reg * H

    H[components_to_update, :] *= np.divide(H_numerator[components_to_update, :], H_denominator[components_to_update, :])

    return H


def update_cd(X, Ht, W, update_H, update_W, tol, max_iter, polarimetry_reg, n_components, F, components_to_update):

    all_components = np.arange(n_components)
    components_to_update_W = np.asarray(all_components, dtype=np.intp)
    if components_to_update is None:
        components_to_update_H = components_to_update_W
    else:
        components_to_update_H = components_to_update
    components_to_update_H = np.asarray(components_to_update_H, dtype=np.intp)

    for i in range(max_iter):

        violation = 0.0      
        if update_W: violation += update_cd_H_W(X, W, Ht, polarimetry_reg, components_to_update_W)
        if update_H: violation += update_cd_H_W(X.T, Ht, W, polarimetry_reg, components_to_update_H, F=F)

        if i == 0:
            violation_init = violation

        if violation / violation_init <= tol or violation_init == 0:
            break

    return Ht.T, W


def update_cd_H_W(X, W, Ht, polarimetry_reg, components_to_update, F=None):

    HHt = np.dot(Ht.T, Ht)
    XHt = np.dot(X, Ht)
    FW = None if F is None else np.dot(F, W)

    return _update_cdnmf_fast(W, HHt, XHt, components_to_update, FW, F, polarimetry_reg)



def update_with_closest_components(H, Hi):
    """
    Updates the matrix `H` by aligning its rows with the closest matching rows from the matrix `Hi`, based on cosine similarity.

    Parameters:
    -----------
    H : numpy.ndarray
        A 2D numpy array of shape `(m, n)` where `m` is the number of rows and `n` is the number of columns. This matrix is the one to be updated with the closest rows from `Hi`.
    
    Hi : numpy.ndarray
        A 2D numpy array of shape `(p, n)` where `p` is the number of rows and `n` is the number of columns. This matrix provides the potential rows to replace or fill in `H` based on similarity.

    Returns:
    --------
    H_updated : numpy.ndarray
        A 2D numpy array of shape `(p, n)` where `p` is the number of rows from `Hi`. The first `m` rows correspond to the original `H`, updated with the closest rows from `Hi`. Any remaining rows are filled with unmatched rows from `Hi`.

    Notes:
    ------
    - The function normalizes both `H` and `Hi` rows before computing similarity, ensuring that comparisons are based on cosine similarity.
    - The function guarantees that each row in `H` is replaced or aligned with a unique row from `Hi`.
    - If `Hi` has more rows than `H`, the additional rows from `Hi` are appended to the updated matrix.
    - If `H` has more rows than `Hi`, the function will raise an error or behave unexpectedly.
    """

    H = H/np.linalg.norm(H, axis=1, keepdims=True)
    Hi = Hi/np.linalg.norm(Hi, axis=1, keepdims=True)
    similarity_matrix = np.dot(H, Hi.T).squeeze()
    closest_idx_list = []
    H_updated = np.zeros(Hi.shape)
    H_updated[:H.shape[0], :] = H

    for i in range(H.shape[0]):

        closest_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        closest_idx_list.append(closest_idx[1])
        similarity_matrix[:, closest_idx[1]] = -np.inf
        similarity_matrix[closest_idx[0], :] = -np.inf

    remaining_rows = Hi.shape[0] - H.shape[0]
    Hi = np.delete(Hi, (closest_idx_list), axis=0)
    if remaining_rows > 0 and Hi.shape[0] > 0:
        H_updated[H.shape[0]:, :] = Hi[:remaining_rows, :]

    return H_updated


def create_F_matrix(half_features):
    """Create F matrix 

    Args:
        half_features (integer): number of features of the input data divided by two

    Returns:
        F (np.ndarray): matrix with shape (2*half_features, 2*half_features)
        the top right and bottom left part of F are identity matrices
    """
    F = np.zeros((2*half_features, 2*half_features))
    F[:half_features, half_features:] = np.eye(half_features)
    F[half_features:, :half_features] = np.eye(half_features)
    return F


def frobenius_norm(X):
    return np.sqrt(squared_norm(X))


