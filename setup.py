import setuptools
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

__name__ = "pymoo"
__author__ = "Julian Blank"
__version__ = '0.0.1'
__url__ = "https://github.com/julesy89/pyGaussianProcess"



setup(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>3.3.0',
    author_email="blankjul@egr.msu.edu",
    description="Gaussian Process using pytorch",
    long_description=readme(),
    license='Apache License 2.0',
    keywords="machine learning",
    packages=setuptools.find_packages(exclude=['tests', 'docs']),
    install_requires=['pytorch', 'numpy']
)
