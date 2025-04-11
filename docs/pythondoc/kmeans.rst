K-Means
==========

.. py:function:: Vision2P.kmeans.kmeans(data, k, max_iters=100, tol=0.0001, random_state=None)

   K-means clustering algorithm.

   :param data: array-like, shape (n_samples, n_features)  
                The input data matrix.
   :type data: array-like

   :param k: The number of clusters.
   :type k: int

   :param max_iters: Maximum number of iterations to run the algorithm. Default is 100.
   :type max_iters: int

   :param tol: Tolerance to declare convergence based on centroid movement. Default is 1e-4.
   :type tol: float

   :param random_state: Seed for random number generator (for reproducibility). Default is None.
   :type random_state: int or None

   :returns: 
      - **centroids** (*array-like, shape (k, n_features)*): K-means centroids.
      - **labels** (*array-like, shape (n_samples,)*): The cluster index for each data point.
