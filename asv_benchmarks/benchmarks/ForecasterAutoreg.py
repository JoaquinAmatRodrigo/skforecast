import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression

class Suite:

    def setup(self):
        # Create forecaster and data. 
        # setup is excluded from the timing
        forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=24)
        time_series = pd.Series(np.arange(10000))

        self.forecaster = forecaster
        self.time_series = time_series

    def time_create_train_X_y(self):
        '''
        Benchmark method create_train_X_y
        '''
        X, y = self.forecaster.create_train_X_y(self.time_series)