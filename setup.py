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
    version='0.1.2',
    description='Spectro-microscopy and polarimetry machine learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='CNRS, Croes Boris, Cherifi-Hertel Salia',
)
