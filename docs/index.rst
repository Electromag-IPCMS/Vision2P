Vision2P documentation
=====================


This package is designed specifically for spectro-microscopy measurements, providing tools for clustering and unmixing spectral data. It includes a K-means clustering algorithm for segmentation and a modified Non-Negative Matrix Factorization (NMF) algorithm with enhanced features tailored for spectro-microscopy applications.

The K-means clustering algorithm enables automatic segmentation of spectro-microscopy measurements and helps to identify distinct regions or features within a dataset.

A regularization term has been added to the NMF algorithm to improve the results from spectro-microscopy with linearly polarized light. It is also possible to constrain some components, using simulated or experimental components, to obtain more accurate results. It is useful for spectral unmixing, to identify pure spectral components within a complex mixture.

It is intended to be compatible with data formats used in some of the most popular Python machine learning packages such as scikit-learn. It retains the same option names to simplify its usage.

The source code is available under BSD-3-Clause license at https://github.com/Electromag-IPCMS/Vision2P


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: Vision2P
   :members:
