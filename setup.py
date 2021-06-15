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
    classifiers=[],
    install_requires=[
          'numpy>=1.20.1',
          'pandas>=1.2.2',
          'tqdm>=4.57.0',
          'scikit-learn>=0.24'
    ]
)
