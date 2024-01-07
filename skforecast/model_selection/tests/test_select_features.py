# Unit test select_features
# ==============================================================================
import re
import pytest
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import select_features

# Fixtures
exog, y = make_regression(n_samples=500, n_features=5, n_informative=2, random_state=123)
exog = pd.DataFrame(
           data    = exog,
           index   = pd.date_range(start='2020-01-01', periods=len(exog), freq='H'),
           columns = [f"exog_{i}" for i in range(exog.shape[1])]
       )
y = pd.Series(y, index=exog.index, name="y")


def test_TypeError_select_features_raise_when_forecaster_is_not_supported():
    """
    Test TypeError is raised in select_features when forecaster is not supported.
    """
    forecaster = ForecasterAutoregDirect(
                     regressor = LinearRegression(),
                     lags      = 5,
                     steps     = 3
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    err_msg = re.escape(
        "`forecaster` must be one of the following classes: ['ForecasterAutoreg', "
        "'ForecasterAutoregCustom']."
    )
    with pytest.raises(TypeError, match = err_msg):
        select_features(
            selector   = selector,
            forecaster = forecaster,
            y          = y,
            exog       = exog,
        )


@pytest.mark.parametrize("select_only", 
                         ['not_exog_or_autoreg', 1, False], 
                         ids=lambda so: f'select_only: {so}')
def test_ValueError_select_features_raise_when_select_only_is_not_autoreg_exog_None(select_only):
    """
    Test ValueError is raised in select_features when `select_only` is not 'autoreg',
    'exog' or None.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    err_msg = re.escape(
        "`select_only` must be one of the following values: 'autoreg', 'exog', None."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = select_only,
        )


@pytest.mark.parametrize("subsample", 
                         [-1, -0.5, 0, 0., 1.1, 2], 
                         ids=lambda ss: f'subsample: {ss}')
def test_ValueError_select_features_raise_when_subsample_is_not_greater_0_less_equal_1(subsample):
    """
    Test ValueError is raised in select_features when `subsample` is not in (0, 1].
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    err_msg = re.escape(
        "`subsample` must be a number greater than 0 and less than or equal to 1."
    )
    with pytest.raises(ValueError, match = err_msg):
        select_features(
            selector   = selector,
            forecaster = forecaster,
            y          = y,
            exog       = exog,
            subsample  = subsample,
        )


def test_select_features_when_selector_is_RFE_and_select_only_is_exog():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog'.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = True,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5]
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']


def test_select_features_when_selector_is_RFE_and_select_only_is_exog_regressor():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'exog' and regressor is passed to the selector instead
    of forecaster.regressor.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=LinearRegression(), n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = False,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5]
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']


def test_select_features_when_selector_is_RFE_and_select_only_is_exog_ForecasterAutoregCustom():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only is 'exog' and forecaster is ForecasterAutoregCustom.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = lambda y: y[-1:-6:-1],
                     window_size    = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'exog',
        verbose     = False,
    )

    assert selected_autoreg == ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2',
                                'custom_predictor_3', 'custom_predictor_4']
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']


def test_select_features_when_selector_is_RFE_and_select_only_is_autoreg():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is 'autoreg'.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_autoreg == [1, 3, 4]
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']


def test_select_features_when_selector_is_RFE_and_select_only_is_autoreg_ForecasterAutoregCustom():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only is 'autoreg' and forecaster is ForecasterAutoregCustom.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = lambda y: y[-1:-6:-1],
                     window_size    = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector    = selector,
        forecaster  = forecaster,
        y           = y,
        exog        = exog,
        select_only = 'autoreg',
        verbose     = False,
    )

    assert selected_autoreg == ['custom_predictor_0', 'custom_predictor_2', 'custom_predictor_3']
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']


def test_select_features_when_selector_is_RFE_and_select_only_is_None():
    """
    Test that select_features returns the expected values when selector is RFE
    and select_only is None.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=5)

    warn_msg = re.escape(
        ("No autoregressive features has been selected. Since a Forecaster "
         "cannot be created without them, be sure to include at least one "
         "to ensure the autoregressive component of the forecast model "
         "using the `force_inclusion` parameter.")
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_autoreg, selected_exog = select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_autoreg == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']


def test_select_features_when_selector_is_RFE_and_select_only_is_None_ForecasterAutoregCustom():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only is None and forecaster is ForecasterAutoregCustom.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = lambda y: y[-1:-6:-1],
                     window_size    = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=5)

    warn_msg = re.escape(
        ("No autoregressive features has been selected. Since a Forecaster "
         "cannot be created without them, be sure to include at least one "
         "to ensure the autoregressive component of the forecast model "
         "using the `force_inclusion` parameter.")
    )
    with pytest.warns(UserWarning, match = warn_msg):
        selected_autoreg, selected_exog = select_features(
            selector    = selector,
            forecaster  = forecaster,
            y           = y,
            exog        = exog,
            select_only = None,
            verbose     = False,
        )

    assert selected_autoreg == []
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']


def test_select_features_when_selector_is_RFE_select_only_exog_is_True_and_force_inclusion_is_regex():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only_exog is True and force_inclusion is regex.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = 'exog',
        force_inclusion = "^exog_4",
        verbose         = False,
    )

    assert selected_autoreg == [1, 2, 3, 4, 5]
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_4']


def test_select_features_when_selector_is_RFE_select_only_exog_is_False_and_force_inclusion_is_regex_ForecasterAutoregCustom():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only_exog is False and force_inclusion is regex and forecaster is
    ForecasterAutoregCustom.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = lambda y: y[-1:-6:-1],
                     window_size    = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = None,
        force_inclusion = "^custom_predictor_",
        verbose         = False,
    )

    assert selected_autoreg == ['custom_predictor_0', 'custom_predictor_1', 'custom_predictor_2',
                                'custom_predictor_3', 'custom_predictor_4']
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']


def test_select_features_when_selector_is_RFE_select_only_exog_is_False_and_force_inclusion_is_list():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only_exog is False and force_inclusion is list.
    """
    forecaster = ForecasterAutoreg(
                     regressor = LinearRegression(),
                     lags      = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = None,
        force_inclusion = ['lag_1'],
        verbose         = False,
    )

    assert selected_autoreg == [1]
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2']


def test_select_features_when_selector_is_RFE_select_only_exog_is_True_and_force_inclusion_is_list_ForecasterAutoregCustom():
    """
    Test that select_features returns the expected values when selector is RFE
    select_only_exog is True and force_inclusion is list and forecaster is
    ForecasterAutoregCustom.
    """
    forecaster = ForecasterAutoregCustom(
                     regressor      = LinearRegression(),
                     fun_predictors = lambda y: y[-1:-6:-1],
                     window_size    = 5,
                 )
    selector = RFE(estimator=forecaster.regressor, n_features_to_select=3)

    selected_autoreg, selected_exog = select_features(
        selector        = selector,
        forecaster      = forecaster,
        y               = y,
        exog            = exog,
        select_only     = 'autoreg',
        force_inclusion = ['custom_predictor_4', 'exog_4'],
        verbose         = False,
    )

    assert selected_autoreg == ['custom_predictor_0', 'custom_predictor_2', 
                                'custom_predictor_3', 'custom_predictor_4']
    assert selected_exog == ['exog_0', 'exog_1', 'exog_2', 'exog_3', 'exog_4']