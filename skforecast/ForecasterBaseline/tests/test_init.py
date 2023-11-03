# Unit test __init__ ForecasterEquivalentDate
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterEquivalentDate import ForecasterEquivalentDate


def test_init_offset_not_int_or_DateOffset():
    """
    Test Exception is raised when offset is not an int or DateOffset.
    """
    with pytest.raises(
        Exception, match="`offset` must be an integer or a pandas.tseries.offsets."
    ):
        ForecasterEquivalentDate(
            offset="not_int_or_DateOffset",
            n_offsets=1,
            agg_func=np.mean,
            forecaster_id=None,
        )


def test_init_window_size_calculation():
    """
    Test window_size is calculated correctly.
    """
    forecaster = ForecasterEquivalentDate(
        offset=5, n_offsets=2, agg_func=np.mean, forecaster_id=None
    )
    assert forecaster.window_size == forecaster.offset * forecaster.n_offsets


def test_init_creation_date_format():
    """
    Test creation_date is in correct format.
    """
    forecaster = ForecasterEquivalentDate(
        offset=5, n_offsets=2, agg_func=np.mean, forecaster_id=None
    )
    assert (
        re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", forecaster.creation_date)
        is not None
    )


def test_init_skforecast_version():
    """
    Test skforecast version is set correctly.
    """
    forecaster = ForecasterEquivalentDate(
        offset=5, n_offsets=2, agg_func=np.mean, forecaster_id=None
    )
    assert forecaster.skforcast_version == skforecast.__version__


def test_init_python_version():
    """
    Test python version is set correctly.
    """
    forecaster = ForecasterEquivalentDate(
        offset=5, n_offsets=2, agg_func=np.mean, forecaster_id=None
    )
    assert forecaster.python_version == sys.version.split(" ")[0]