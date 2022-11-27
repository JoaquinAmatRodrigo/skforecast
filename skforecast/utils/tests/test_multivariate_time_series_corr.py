# Unit test multivariate_time_series_corr
# ==============================================================================
import pandas as pd
from skforecast.utils.utils import multivariate_time_series_corr

def test_output_multivariate_time_series_corr():
    """
    Test the output of the correlation matrix
    """

    time_series = pd.Series(range(10), name='series_1')
    other = pd.DataFrame({
                'series_2': range(10),
                'series_3': [9, 2, 3, 8, 2, 6, 0, 8, 6, 1]
            })

    expected = pd.DataFrame(
                data = {
                    'series_2': [1., 1., 1., 1., 1.],
                    'series_3': [-0.21854656, -0.02836, -0.11964501, -0.45360921, -0.17251639],
                },
                index = range(5)
            )
    expected.index.name = 'lag'

    results = multivariate_time_series_corr(
                  time_series = time_series,
                  other       = other,
                  lags        = 5
              )

    pd.testing.assert_frame_equal(results, expected)
