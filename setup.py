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
    #packages=find_packages(include=['Vision2P']),
    install_requires=[
        'numpy',
        'scipy',
        'cython'
    ],
    ext_modules=cythonize(extensions, language_level="3"),
    version='0.1.0',
    description='Spectro-microscopy and polarimetry machine learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='CNRS, CroesBoris, Cherifi-Hertel Salia',
)