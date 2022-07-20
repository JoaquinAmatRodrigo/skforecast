# Unit test model_selection.utils
# ==============================================================================

from skforecast.utils.utils import check_interval
import pytest


def test_check_interval():
    """Test provided interval fits all constraints."""

    # Good interval sequence
    assert check_interval([5, 95]) is None

    # Wrong interval sequences
    err_msg = "Interval sequence should contain exactly 2 values, respectively lower and upper interval bounds"
    with pytest.raises(ValueError, match=err_msg):
        check_interval([5, 1, 95])

    err_msg = r"Lower interval bound \(95\) has to be strictly smaller than upper interval bound \(5\)"
    with pytest.raises(ValueError, match=err_msg):
        check_interval([95, 5])

    err_msg = r"Lower interval bound \(-1\) has to be >= 0"
    with pytest.raises(ValueError, match=err_msg):
        check_interval([-1, 95])

    err_msg = r"Upper interval bound \(105\) has to be <= 100"
    with pytest.raises(ValueError, match=err_msg):
        check_interval([5, 105])
