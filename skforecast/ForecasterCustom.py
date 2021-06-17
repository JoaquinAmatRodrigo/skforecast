################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8

from .ForecasterAutoregCustom import ForecasterAutoregCustom


################################################################################
#                              ForecasterCustom                                #
################################################################################

class ForecasterCustom(ForecasterAutoregCustom):
    '''
    This class is equialent to ForecasterAutoregCustom. It exists for backward
    compatibility. Please use ForecasterAutoregCustom.     
    '''