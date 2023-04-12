################################################################################
#                            skforecast.exceptions                             #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under a Creative Commons Attribution 4.0 International License.              #
################################################################################
# coding=utf-8

'''
The skforecast.exceptions module includes all custom warnings and error classes used
across skforecast.
'''

class MissingValuesExogWarning(UserWarning):
    """
    Warning used to notify there are missing values in the exogenous data.
    This warning occurs when the input data passed to the `exog` arguments contains
    missing values. Most machine learning models do not accept missing values, therefore
    the forecaster `fit` and `predict` may fail.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingValuesExogWarning)"
        )
        return self.message + " " + extra_message
    

class ValueTypeWarning(UserWarning):
    """
    Warning used to notify there are values in the exogenous data that are not
    'int', 'float', 'bool' or 'category'. Most machine learning models do not
    accept other data types, therefore the forecaster `fit` and `predict` may fail.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=ValueTypesExogWarning)"
        )
        return self.message + " " + extra_message
    

class LongTrainingWarning(UserWarning):
    """
    Warning used to notify that a large number of models will be trained and the
    the process may take a while to run.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=LongTrainingWarning)"
        )
        return self.message + " " + extra_message