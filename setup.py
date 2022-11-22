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
    requirements_base = f.read().splitlines()

with open('requirements_optional.txt') as f:
    requirements_optional = f.read()

with open("requirements_test.txt") as f:
    requirements_test = f.read().splitlines()

extras_require = {
    "statsmodels": requirements_optional.split("\n\n")[0].splitlines(),
    "bayesian": requirements_optional.split("\n\n")[1].splitlines(),
    "plotting": requirements_optional.split("\n\n")[2].splitlines(),
    "test": requirements_test
}

extras_require["full"] = (
    extras_require["statsmodels"]
    + extras_require["bayesian"]
    + extras_require["plotting"]
    + extras_require["test"]
)


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
    install_requires=requirements_base,
    extras_require=extras_require,
    tests_require=requirements_test,
)
