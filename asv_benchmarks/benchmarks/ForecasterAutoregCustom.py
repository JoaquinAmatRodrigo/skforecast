# ASV Benchmarks ForecasterAutoregCustom
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression


class Suite:

    def setup(self):
        # Create forecaster and data. 
        # setup is excluded from the timing

        def create_predictors(y):
            """
            Create first 10 lags of a time series.
            Calculate moving average with window 20.
            """

            lags = y[-1:-11:-1]
            mean = np.mean(y[-20:])
            predictors = np.hstack([lags, mean])

            return predictors

        forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 20
             )
        time_series = pd.Series(np.arange(10000))

        self.forecaster = forecaster
        self.time_series = time_series

    def time_create_train_X_y(self):
        """
        Benchmark method create_train_X_y
        """
        X, y = self.forecaster.create_train_X_y(self.time_series)