################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################

import setuptools
import os
import sys
import warnings
import skforecast

VERSION = skforecast.__version__

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

if sys.version_info[:2] > (3, 10):
    fmt = "Skforecast {} may not yet support Python {}.{}."
    warnings.warn(
        fmt.format(VERSION, *sys.version_info[:2]),
        RuntimeWarning)
    del fmt
        
setuptools.setup(
    name="skforecast",
    version=VERSION,
    author="Joaquin Amat Rodrigo and Javier Escobar Ortiz",
    author_email="j.amatrodrigo@gmail.com, javier.escobar.ortiz@gmail.com",
    description="Forecasting time series with scikit-learn regressors. It also works with any regressor compatible with the scikit-learn API (pipelines, CatBoost, LightGBM, XGBoost, Ranger...).",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/JoaquinAmatRodrigo/skforecast",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires=requirements
)
