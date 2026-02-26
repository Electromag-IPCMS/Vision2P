from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="Vision2P._cdnmf_fast",
        sources=["Vision2P/_cdnmf_fast.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name='Vision2P',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'cython'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx.ext.napoleon',
        ]
    },
    ext_modules=cythonize(extensions, language_level="3"),
    version='0.1.4',
    description='Spectro-microscopy and polarimetry machine learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='CNRS, Croes Boris, Cherifi-Hertel Salia',
    url="https://github.com/Electromag-IPCMS/Vision2P",
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
