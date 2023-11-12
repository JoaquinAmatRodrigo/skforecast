################################################################################
#                            skforecast.exceptions                             #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

"""
The skforecast.exceptions module contains all the custom warnings and error 
classes used across skforecast.
"""

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


class DataTypeWarning(UserWarning):
    """
    Warning used to notify there are dtypes in the exogenous data that are not
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


class IgnoredArgumentWarning(UserWarning):
    """
    Warning used to notify that an argument is ignored when using a method 
    or a function.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=IgnoredArgumentWarning)"
        )
        return self.message + " " + extra_message


class SkforecastVersionWarning(UserWarning):
    """
    Warning used to notify that the skforecast version installed in the 
    environment differs from the version used to initialize the forecaster.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SkforecastVersionWarning)"
        )
        return self.message + " " + extra_message