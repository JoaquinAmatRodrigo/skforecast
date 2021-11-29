################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8

import warnings
from .ForecasterAutoregCustom import ForecasterAutoregCustom


################################################################################
#                              ForecasterCustom                                #
################################################################################

class ForecasterCustom(ForecasterAutoregCustom):
    '''
    Deprecated. This class is equialent to ForecasterAutoregCustom. It exists for backward
    compatibility. Please use ForecasterAutoregCustom.     
    '''

    def __init__(self, regressor, fun_predictors, window_size):
        warnings.warn(
            'Deprecated. This class is equialent to ForecasterAutoregCustom. It exists for backward compatibility. Use ForecasterAutoregCustom instead.',
            DeprecationWarning
        )
        super().__init__(regressor, fun_predictors, window_size)

    