# Unit test prepare_residuals_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from skforecast.utils import prepare_residuals_multiseries
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

# Fixtures
from skforecast.ForecasterAutoregMultiSeries.tests.fixtures_ForecasterAutoregMultiSeries import series


def test_prepare_residuals_multiseries_ValueError_when_not_in_sample_residuals_for_any_level():
    """
    Test ValueError is raised when there is no in_sample_residuals for any level.
    """
    levels = ['1', '2']
    use_in_sample = True
    in_sample_residuals = {2: np.array([1, 2, 3])}

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    forecaster.in_sample_residuals = in_sample_residuals

    err_msg = re.escape(
        (f"Not `forecaster.in_sample_residuals` for levels: "
         f"{set(['1', '2']) - set(forecaster.in_sample_residuals.keys())}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_residuals_multiseries(
            levels              = levels,
            use_in_sample       = use_in_sample, 
            in_sample_residuals = forecaster.in_sample_residuals
        )


def test_prepare_residuals_multiseries_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when `use_in_sample` is False and 
    `out_sample_residuals` is None.
    """
    levels = ['1', '2']
    use_in_sample = False

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals` is `None`. Use "
         "`in_sample_residuals=True` or method "
         "`set_out_sample_residuals()` before `predict_interval()`, "
         "`predict_bootstrapping()`,`predict_quantiles()` or "
         "`predict_dist()`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_residuals_multiseries(
            levels               = levels,
            use_in_sample        = use_in_sample, 
            out_sample_residuals = forecaster.out_sample_residuals
        )


def test_prepare_residuals_multiseries_ValueError_when_not_out_sample_residuals_for_all_levels():
    """
    Test ValueError is raised when out_sample_residuals is not 
    available for all levels.
    """
    levels = ['1']
    use_in_sample = False
    out_sample_residuals = {'2': np.array([1, 2, 3, 4, 5]), 
                            '3': np.array([1, 2, 3, 4, 5])}
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    forecaster.out_sample_residuals = out_sample_residuals

    err_msg = re.escape(
        (f"Not `forecaster.out_sample_residuals` for levels: "
         f"{set(levels) - set(forecaster.out_sample_residuals.keys())}. "
         f"Use method `set_out_sample_residuals()`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_residuals_multiseries(
            levels               = levels,
            use_in_sample        = use_in_sample, 
            out_sample_residuals = forecaster.out_sample_residuals
        )


@pytest.mark.parametrize("transformer_series", 
                         [None, StandardScaler()],
                         ids = lambda tr : f'transformer_series type: {type(tr)}')
def test_prepare_residuals_multiseries_ValueError_when_level_out_sample_residuals_value_is_None(transformer_series):
    """
    Test ValueError is raised when use_in_sample=False and
    forecaster.out_sample_residuals has a level with a None.
    """
    levels = ['1', '2']
    use_in_sample = False
    out_sample_residuals = {'1': np.array([1, 2, 3, 4, 5])}

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=transformer_series)
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(residuals=out_sample_residuals)

    err_msg = re.escape(
        ("Not available residuals for level '2'. "
         "Check `forecaster.out_sample_residuals`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_residuals_multiseries(
            levels               = levels,
            use_in_sample        = use_in_sample, 
            out_sample_residuals = forecaster.out_sample_residuals
        )


@pytest.mark.parametrize("transformer_series", 
                         [None, StandardScaler()],
                         ids = lambda tr : f'transformer_series type: {type(tr)}')
def test_prepare_residuals_multiseries_ValueError_when_level_out_sample_residuals_value_contains_None_or_NaNs(transformer_series):
    """
    Test ValueError is raised when in_sample_residuals=False and
    forecaster.out_sample_residuals has a level with a None or NaN value.
    """
    levels = ['1', '2']
    use_in_sample = False
    out_sample_residuals = {'1': np.array([1, 2, 3, 4, 5]),
                            '2': np.array([1, 2, 3, 4, None])}  # StandardScaler() transforms None to NaN
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=transformer_series)
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(residuals = out_sample_residuals)

    err_msg = re.escape(
        ("forecaster residuals for level '2' contains `None` "
         "or `NaNs` values. Check `forecaster.out_sample_residuals`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_residuals_multiseries(
            levels               = levels,
            use_in_sample        = use_in_sample, 
            out_sample_residuals = forecaster.out_sample_residuals
        )


@pytest.mark.parametrize("use_in_sample", 
                         [True, False],
                         ids = lambda use_in_sample : f'use_in_sample: {use_in_sample}')
def test_output_prepare_residuals_multiseries(use_in_sample):
    """
    Test output of prepare_residuals_multiseries when use_in_sample=True and False.
    """
    levels = ['1', '2']
    residuals = {'1': np.array([1, 2, 3, 4, 5]),
                 '2': np.array([1, 2, 3, 4, 5])}
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series)
    forecaster.in_sample_residuals = residuals
    forecaster.set_out_sample_residuals(residuals=residuals)

    residuals_levels = prepare_residuals_multiseries(
                           levels               = levels,
                           use_in_sample        = use_in_sample,
                           in_sample_residuals  = forecaster.in_sample_residuals,
                           out_sample_residuals = forecaster.out_sample_residuals
                       )
    
    for level in residuals.keys():
        np.testing.assert_array_equal(residuals_levels[level], residuals[level])