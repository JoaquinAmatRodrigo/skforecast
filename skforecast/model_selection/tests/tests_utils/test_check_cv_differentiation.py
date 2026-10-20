# Unit test _check_cv_differentiation
# ==============================================================================
import re
import pytest
from sklearn.linear_model import Ridge, LogisticRegression
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.recursive import ForecasterEquivalentDate
from skforecast.recursive import ForecasterRecursiveClassifier
from skforecast.direct import ForecasterDirect
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from skforecast.model_selection._utils import _check_cv_differentiation


@pytest.mark.parametrize(
    "cv_class", [TimeSeriesFold, OneStepAheadFold], ids=lambda cv: cv.__name__
)
@pytest.mark.parametrize(
    "forecaster, cv_differentiation",
    [
        (ForecasterRecursive(estimator=Ridge(), lags=2, differentiation=1), None),
        (ForecasterRecursive(estimator=Ridge(), lags=2, differentiation=1), 1),
        (ForecasterDirect(estimator=Ridge(), lags=2, steps=3, differentiation=2), None),
        (ForecasterRecursive(estimator=Ridge(), lags=2), None),
        (ForecasterRecursive(estimator=Ridge(), lags=2), 0),
        (
            ForecasterRecursiveMultiSeries(
                estimator=Ridge(),
                lags=2,
                differentiation={"l1": 1, "l2": 2, "_unknown_level": 1},
            ),
            None,
        ),
        (
            ForecasterRecursiveMultiSeries(
                estimator=Ridge(),
                lags=2,
                differentiation={"l1": 1, "l2": 2, "_unknown_level": 1},
            ),
            2,
        ),
        (ForecasterEquivalentDate(offset=1), None),
        (ForecasterRecursiveClassifier(estimator=LogisticRegression(), lags=2), 0),
    ],
    ids=lambda x: type(x).__name__ if not isinstance(x, (int, type(None))) else f"cv: {x}",
)
def test_check_cv_differentiation_no_error_when_cv_differentiation_None_or_equal(
    forecaster, cv_differentiation, cv_class
):
    """
    Test no error is raised when `differentiation` of the cv is `None` (the
    value of the forecaster is used), equal to the one of the forecaster
    (the maximum when it is a dict), or 0 for a forecaster without
    differentiation.
    """
    kwargs = {"steps": 3} if cv_class is TimeSeriesFold else {}
    cv = cv_class(
        initial_train_size=20, differentiation=cv_differentiation, **kwargs
    )

    _check_cv_differentiation(forecaster=forecaster, cv=cv)

    assert cv.differentiation == cv_differentiation


@pytest.mark.parametrize(
    "cv_class", [TimeSeriesFold, OneStepAheadFold], ids=lambda cv: cv.__name__
)
@pytest.mark.parametrize(
    "forecaster, cv_differentiation, forecaster_differentiation",
    [
        (ForecasterRecursive(estimator=Ridge(), lags=2, differentiation=2), 1, 2),
        (ForecasterRecursive(estimator=Ridge(), lags=2), 1, None),
        (ForecasterEquivalentDate(offset=1), 1, None),
    ],
    ids=lambda x: type(x).__name__ if not isinstance(x, (int, type(None))) else f"{x}",
)
def test_check_cv_differentiation_ValueError_when_cv_differentiation_differs(
    forecaster, cv_differentiation, forecaster_differentiation, cv_class
):
    """
    Test ValueError is raised when `differentiation` of the cv is set by the
    user to a value different from the `differentiation` of the forecaster.
    """
    kwargs = {"steps": 3} if cv_class is TimeSeriesFold else {}
    cv = cv_class(
        initial_train_size=20, differentiation=cv_differentiation, **kwargs
    )

    err_msg = re.escape(
        f"The differentiation included in the forecaster "
        f"({forecaster_differentiation}) differs from the differentiation "
        f"included in the cv ({cv_differentiation}). Set `differentiation=None` "
        f"in the cv to use the value of the forecaster."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_cv_differentiation(forecaster=forecaster, cv=cv)


def test_check_cv_differentiation_ValueError_when_dict_differentiation_differs():
    """
    Test ValueError is raised when `differentiation` of the cv differs from the
    maximum `differentiation` of a ForecasterRecursiveMultiSeries with a dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=Ridge(),
        lags=2,
        differentiation={"l1": 1, "l2": 2, "_unknown_level": 1},
    )
    cv = TimeSeriesFold(steps=3, initial_train_size=20, differentiation=1)

    err_msg = re.escape(
        "When using a dict as `differentiation` in ForecasterRecursiveMultiSeries, "
        "the `differentiation` included in the cv (1) must be "
        "the same as the maximum `differentiation` included in the forecaster "
        "(2). Set `differentiation=None` in the cv "
        "to use the value of the forecaster."
    )
    with pytest.raises(ValueError, match=err_msg):
        _check_cv_differentiation(forecaster=forecaster, cv=cv)
