# Unit test create_train_X_y
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression



def create_predictors(y):
    '''
    Create first 5 lags of a time series.
    '''
    
    lags = y[-1:-6:-1]
    
    return lags 


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_None():
    '''
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is None.
    '''
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    results = forecaster.create_train_X_y(y=pd.Series(np.arange(10)))
    expected = (pd.DataFrame(
                    data = np.array([[4, 3, 2, 1, 0],
                                    [5, 4, 3, 2, 1],
                                    [6, 5, 4, 3, 2],
                                    [7, 6, 5, 4, 3],
                                    [8, 7, 6, 5, 4]]),
                    index   = np.array([5, 6, 7, 8, 9]),
                    columns = ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4']
                ),
                pd.Series(
                    np.array([5, 6, 7, 8, 9]),
                    index = np.array([5, 6, 7, 8, 9]))
               )     

    assert (results[0] == expected[0]).all().all()
    assert (results[1] == expected[1]).all()


def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_series():
    '''
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas series
    '''
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    results = forecaster.create_train_X_y(
                y = pd.Series(np.arange(10)),
                exog =  pd.Series(np.arange(100, 110), name='exog')
              )
    expected = (pd.DataFrame(
                    data = np.array([[4, 3, 2, 1, 0, 105],
                                    [5, 4, 3, 2, 1, 106],
                                    [6, 5, 4, 3, 2, 107],
                                    [7, 6, 5, 4, 3, 108],
                                    [8, 7, 6, 5, 4, 109]]),
                    index   = np.array([5, 6, 7, 8, 9]),
                    columns = ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4', 'exog']
                ),
                pd.Series(
                    np.array([5, 6, 7, 8, 9]),
                    index = np.array([5, 6, 7, 8, 9]))
               )       

    assert (results[0] == expected[0]).all().all()
    assert (results[1] == expected[1]).all()

def test_create_train_X_y_output_when_y_is_series_10_and_exog_is_dataframe():
    '''
    Test the output of create_train_X_y when y=pd.Series(np.arange(10)) and 
    exog is a pandas dataframe with two columns.
    '''
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    results = forecaster.create_train_X_y(
                y = pd.Series(np.arange(10)),
                exog = pd.DataFrame({
                            'exog_1' : np.arange(100, 110),
                            'exog_2' : np.arange(1000, 1010)
                })
              )
    expected = (pd.DataFrame(
                    data = np.array([[4, 3, 2, 1, 0, 105, 1005],
                                  [5, 4, 3, 2, 1, 106, 1006],
                                  [6, 5, 4, 3, 2, 107, 1007],
                                  [7, 6, 5, 4, 3, 108, 1008],
                                  [8, 7, 6, 5, 4, 109, 1009]]),
                    index   = np.array([5, 6, 7, 8, 9]),
                    columns = ['custom_predictor_0', 'custom_predictor_1',
                               'custom_predictor_2', 'custom_predictor_3',
                               'custom_predictor_4', 'exog_1', 'exog_2']
                ),
                pd.Series(
                    np.array([5, 6, 7, 8, 9]),
                    index = np.array([5, 6, 7, 8, 9])
                )
               )        

    assert (results[0] == expected[0]).all().all()
    assert (results[1] == expected[1]).all()

    
def test_create_train_X_y_exception_when_y_and_exog_have_different_length():
    '''
    Test exception is raised when length of y and length of exog are different.
    '''
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    with pytest.raises(Exception):
        forecaster.fit(y=pd.Series(np.arange(50)), exog=pd.Series(np.arange(10)))
    with pytest.raises(Exception):
        forecaster.fit(y=pd.Series(np.arange(10)), exog=pd.Series(np.arange(50)))
    with pytest.raises(Exception):
        forecaster.fit(
            y=pd.Series(np.arange(10)),
            exog=pd.DataFrame(np.arange(50).reshape(25,2))
        )
        
def test_create_train_X_y_exception_when_y_and_exog_have_different_index():
    '''
    Test exception is raised when y and exog have different index.
    '''
    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    with pytest.raises(Exception):
        forecaster.fit(
            y=pd.Series(np.arange(50)),
            exog=pd.Series(np.arange(10), index=np.arange(100, 110))
        )  
        