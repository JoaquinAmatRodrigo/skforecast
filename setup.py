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
    "sarimax": requirements_optional.split("\n\n")[0].splitlines(),
    "plotting": requirements_optional.split("\n\n")[1].splitlines(),
    "test": requirements_test
}

extras_require["full"] = (
    extras_require["sarimax"]
    + extras_require["plotting"]
    + extras_require["test"]
)

extras_require["all"] = extras_require["full"]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")

if sys.version_info[:2] > (3, 11):
    fmt = "skforecast {} may not yet support Python {}.{}."
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License"
    ],
    install_requires=requirements_base,
    extras_require=extras_require,
    tests_require=requirements_test,
)
