################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################

import setuptools

import skforecast

VERSION = skforecast.__version__
        
        
setuptools.setup(
    name="skforecast",
    version=VERSION,
    author="Joaquin Amat Rodrigo",
    author_email="j.amatrodrigo@gmail.com",
    description="Forecasting time series with scikitlearn regressors",
    url="https://github.com/JoaquinAmatRodrigo/skforecast",
    packages=setuptools.find_packages(),
    classifiers=[]
)
