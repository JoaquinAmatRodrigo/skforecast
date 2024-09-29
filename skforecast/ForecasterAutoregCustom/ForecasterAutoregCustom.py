################################################################################
#                        ForecasterAutoregCustom                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Optional, Callable
import warnings

# TODO: Update warning with user guide link
class ForecasterAutoregCustom():
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive (multi-step) forecaster with a custom function to create predictors.
    
    """
    
    def __init__(
        self, 
        regressor: object = None, 
        fun_predictors: Callable = None, 
        window_size: int = None,
        name_predictors: Optional[list] = None,
        transformer_y: Optional[object] = None,
        transformer_exog: Optional[object] = None,
        weight_func: Optional[Callable] = None,
        differentiation: Optional[int] = None,
        fit_kwargs: Optional[dict] = None,
        forecaster_id: Optional[Union[str, int]] = None
    ) -> None:
        
        warnings.warn(
            ("This class is deprecated since skforecast 0.14.0. Use "
             "ForecasterAutoreg instead.")
        )
