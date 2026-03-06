from setuptools import find_packages, setup, Extension
import os

class get_numpy_include:
    def __str__(self):
        import numpy
        return numpy.get_include()

setup(
    name='Vision2P',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx.ext.napoleon',
        ]
    },
    version='0.4.0',
    description='Spectro-microscopy and polarimetry machine learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='CNRS, Croes Boris, Cherifi-Hertel Salia',
    url="https://github.com/Electromag-IPCMS/Vision2P",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

