################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################

import setuptools
import sys
import warnings

if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")

if sys.version_info[:2] > (3, 11):
    fmt = "skforecast {} may not yet support Python {}.{}."
    warnings.warn(
        fmt.format(VERSION, *sys.version_info[:2]),
        RuntimeWarning)
    del fmt


setuptools.setup()
