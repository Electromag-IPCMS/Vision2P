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

import numpy as np

def _initialize_kmeans_plusplus(data, k, random_state=None):
    '''
    Centroid initialization for K-means++

    Inputs:
        data : numpy array of data points with shape (n_samples, n_features)
        k : number of clusters
    
    Returns:
        centroids : list of k centroids selected from data
    '''
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = data.shape[0]
    centroids = [data[np.random.randint(n_samples)]]

    for _ in range(1, k):
        dist = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in centroids], axis=0)
        next_centroid = data[np.argmax(dist)]
        centroids.append(next_centroid)

    return centroids


def kmeans(data, k, max_iters=100, tol=1e-4, random_state=None):
    '''
    K-means clustering algorithm

    Inputs:
        data : numpy array of data points with shape (n_samples, n_features)
        k : number of clusters
        max_iters : maximum number of iterations
        tol : tolerance to declare convergence based on centroid movement
    
    Returns:
        centroids : numpy array of shape (k, n_features) with final centroids
        labels : numpy array of shape (n_samples,) with the cluster index for each data point
    '''

    centroids = _initialize_kmeans_plusplus(data, k, random_state=random_state)
    
    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

    return centroids, labels