import sys
sys.path.insert(1, '/home/ximo/Documents/GitHub/skforecast')

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast import __version__
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def create_predictors(y):
    '''
    Create first 5 lags of a time series.
    '''
    X_train = pd.DataFrame({'y':y.copy()})
    for i in range(0, 5):
        X_train[f'lag_{i+1}'] = X_train['y'].shift(i)
    X_train = X_train.drop(columns='y').tail(1).to_numpy()  
    
    return X_train  
    
    
# Test initializations
#-------------------------------------------------------------------------------
def test_init_exception_when_window_size_argument_is_string():
   
   with pytest.raises(Exception):
        forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = '5'
                    )

def test_init_exception_when_fun_predictors_argument_is_string():
   
   with pytest.raises(Exception):
        forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = 'create_predictors',
                        window_size    = 5
                    )

# Test method create_train_X_y()
#-------------------------------------------------------------------------------
def test_create_train_X_y_output_when_y_is_range_10_and_exog_is_None():

    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    results = forecaster.create_train_X_y(y=np.arange(10))
    expected = (np.array([[4., 3., 2., 1., 0.],
                          [5., 4., 3., 2., 1.],
                          [6., 5., 4., 3., 2.],
                          [7., 6., 5., 4., 3.],
                          [8., 7., 6., 5., 4.]]),
                np.array([5, 6, 7, 8, 9]))     

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_create_train_X_y_output_when_y_is_range_10_and_exog_is_1d_array():

    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    results = forecaster.create_train_X_y(y=np.arange(10), exog=np.arange(100, 110))
    expected = (np.array([[4.,   3.,   2.,   1.,   0., 105.],
                          [5.,   4.,   3.,   2.,   1., 106.],
                          [6.,   5.,   4.,   3.,   2., 107.],
                          [7.,   6.,   5.,   4.,   3., 108.],
                          [8.,   7.,   6.,   5.,   4., 109.]]),
                np.array([5, 6, 7, 8, 9]))     

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()

def test_create_train_X_y_output_when_y_is_range_10_and_exog_is_2d_array():

    forecaster = ForecasterAutoregCustom(
                    regressor      = LinearRegression(),
                    fun_predictors = create_predictors,
                    window_size    = 5
                )
    results = forecaster.create_train_X_y(
                y=np.arange(10),
                exog=np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
              )
    expected = (np.array([[4,    3,    2,    1,    0,  105, 1005],
                          [5,    4,    3,    2,    1,  106, 1006],
                          [6,    5,    4,    3,    2,  107, 1007],
                          [7,    6,    5,    4,    3,  108, 1008],
                          [8,    7,    6,    5,    4,  109, 1009]]),
                np.array([5, 6, 7, 8, 9]))     

    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()
                
# Test method fit()
#-------------------------------------------------------------------------------
def test_fit_exception_when_y_and_exog_have_different_lenght():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(50), exog=np.arange(10))


def test_fit_exception_when_y_lenght_is_less_than_window_size_needed():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    with pytest.raises(Exception):
        forecaster.fit(y=np.arange(5))


def test_last_window_stored_when_fit_forecaster():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    forecaster.fit(y=np.arange(50))
    assert (forecaster.last_window == np.array([45, 46, 47, 48, 49])).all()
    
    
def test_in_sample_residuals_stored_when_fit_forecaster():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    forecaster.fit(y=np.arange(7))
    assert (forecaster.in_sample_residuals == np.array([0., 0.])).all()


# Test method predict()
#-------------------------------------------------------------------------------
def test_predict_exception_when_steps_lower_than_1():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                )
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=0)

def test_predict_exception_when_forecaster_fited_without_exog_and_exog_passed_when_predict():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=np.arange(10))


def test_predict_exception_when_forecaster_fited_with_exog_but_not_exog_passed_when_predict():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(50), exog=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10)
        
        
def test_predict_exception_when_exog_lenght_is_less_than_steps():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(50), exog=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=np.arange(5))


def test_predict_exception_when_exog_passed_in_predict_has_different_columns_than_exog_used_to_fit_nparray():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10), exog=np.arange(30).reshape(-1, 3))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=np.arange(30).reshape(-1, 2))

def test_predict_exception_when_exog_passed_in_predict_has_different_columns_than_exog_used_to_fit_pdDataDrame():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10), exog=pd.DataFrame(np.arange(30).reshape(-1, 3)))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, exog=pd.DataFrame(np.arange(30).reshape(-1, 2)))
        
        
def test_predict_exception_when_last_window_argument_is_not_numpy_array_or_pandas_series():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, last_window=[1,2,3])


def test_predict_exception_when_last_window_lenght_is_less_than_maximum_lag():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(50))
    with pytest.raises(Exception):
        forecaster.predict(steps=10, last_window=pd.Series([1, 2]))
        

def test_predict_output_when_regresor_is_LinearRegression_lags_is_3_ytrain_is_numpy_arange_50_and_steps_is_5():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(50))
    predictions = forecaster.predict(steps=5)
    expected = np.array([50., 51., 52., 53., 54.])
    assert predictions == approx(expected)
    
    
# Test method _check_y()
#-------------------------------------------------------------------------------
def test_check_y_exception_when_y_is_int():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_y(y=10)
        
def test_check_y_exception_when_y_is_list():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_y(y=[1, 2, 3])
        
        
def test_check_y_exception_when_y_is_numpy_array_with_more_than_one_dimension():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_y(y=np.arange(10).reshape(-1, 1))
        
        
        
# Test method _check_last_window()
#-------------------------------------------------------------------------------
def test_check_last_window_exception_when_last_window_is_int():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_y(y=10)
        
def test_check_last_window_exception_when_last_window_is_list():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_y(y=[1, 2, 3])
        
        
def test_check_last_window_exception_when_last_window_is_numpy_array_with_more_than_one_dimension():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_y(y=np.arange(10).reshape(-1, 1))
    
    
# Test method _check_exog()
#-------------------------------------------------------------------------------
def test_check_exog_exception_when_exog_is_int():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(exog=10)
        
        
def test_check_exog_exception_when_exog_is_list():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(exog=[1, 2, 3])
        
        
def test_check_exog_exception_when_exog_is_numpy_array_with_more_than_2_dimensions():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(exog=np.arange(30).reshape(-1, 10, 3))
        
        
def test_check_exog_exception_when_ref_type_is_pandas_series_and_exog_is_numpy_array_with_more_than_1_column():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog     = np.arange(30).reshape(-1, 2),
            ref_type = pd.Series
        )
        
        
def test_check_exog_exception_when_ref_type_is_pandas_series_and_exog_is_numpy_array_with_more_than_2_dimensions():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(exog=np.arange(30).reshape(-1, 10, 3))
        
        
def test_check_exog_exception_when_exog_has_diferent_number_of_columns_than_ref_shape():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = np.arange(30).reshape(-1, 3),
            ref_type  = np.ndarray,
            ref_shape = (1, 2)
        )

def test_check_exog_exception_when_exog_is_1d_numpy_array_and_ref_shape_has_more_than_1_column():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = np.arange(30),
            ref_type  = np.ndarray,
            ref_shape = (1, 5)
        )
        
        
def test_check_exog_exception_when_exog_is_pandas_series_and_ref_shape_has_more_than_1_column():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = pd.Series(np.arange(30)),
            ref_type  = np.ndarray,
            ref_shape = (1, 5)
        )

def test_check_exog_exception_when_exog_is_pandas_dataframe_with_3_columns_and_ref_shape_has_2_columns():
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster._check_exog(
            exog      = pd.DataFrame(np.arange(30).reshape(-1, 3)),
            ref_type  = pd.DataFrame,
            ref_shape = (10, 2)
        )

        
# Test method _preproces_y()
#-------------------------------------------------------------------------------
def test_preproces_y_when_y_is_pandas_series():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    assert (forecaster._preproces_y(y=pd.Series([0, 1, 2])) == np.arange(3)).all()
    

def test_preproces_y_when_y_is_1d_numpy_array():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    assert (forecaster._preproces_y(y=np.arange(3)) == np.arange(3)).all()
    


# Test method _preproces_last_window()
#-------------------------------------------------------------------------------
def test_preproces_last_window_when_last_window_is_pandas_series():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    assert (forecaster._preproces_last_window(last_window=pd.Series([0, 1, 2])) == np.arange(3)).all()
    

def test_preproces_last_window_when_last_window_is_1d_numpy_array():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    assert (forecaster._preproces_last_window(last_window=np.arange(3)) == np.arange(3)).all()


# Test method _preproces_exog()
#-------------------------------------------------------------------------------
def test_preproces_exog_when_exog_is_pandas_series():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    assert (forecaster._preproces_exog(exog=pd.Series([0, 1, 2])) == np.arange(3).reshape(-1, 1)).all()
    

def test_preproces_exog_when_exog_is_1d_numpy_array():

    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    assert (forecaster._preproces_exog(exog=np.arange(3)) == np.arange(3).reshape(-1, 1)).all()
    
    
# Test method test_set_paramns()
#-------------------------------------------------------------------------------
def test_set_paramns():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(fit_intercept=True),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    new_paramns = {'fit_intercept': False}
    forecaster.set_params(**new_paramns)
    expected = {'copy_X': True,
                 'fit_intercept': False,
                 'n_jobs': None,
                 'normalize': False,
                 'positive': False
                }
    assert forecaster.regressor.get_params() == expected

   
    
# Test method set_out_sample_residuals()
#-------------------------------------------------------------------------------
def test_set_out_sample_residuals_exception_when_residuals_argument_is_list():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    with pytest.raises(Exception):
        forecaster.set_out_sample_residuals(residuals=[1,2,3])
        
        
def test_set_out_sample_residuals_when_residuals_lenght_is_less_or_equal_to_1000():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.set_out_sample_residuals(residuals=np.arange(10))
    assert (forecaster.out_sample_residuals == np.arange(10)).all()
    

def test_set_out_sample_residuals_when_residuals_lenght_is_greater_than_1000():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.set_out_sample_residuals(residuals=np.arange(2000))
    assert len(forecaster.out_sample_residuals) == 1000
    
    
# Test method get_coef()
#-------------------------------------------------------------------------------
def test_get_coef_when_regressor_is_LinearRegression():
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(7))
    expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert forecaster.get_coef() == approx(expected)
    
def test_get_coef_when_regressor_is_Ridge():
    forecaster = ForecasterAutoregCustom(
                        regressor      = Ridge(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(7))
    expected = np.array([0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714])
    assert forecaster.get_coef() == approx(expected)
    
def test_get_coef_when_regressor_is_Lasso():
    forecaster = ForecasterAutoregCustom(
                        regressor      = Lasso(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(50))
    expected = np.array([0.99407115, 0., 0., 0., 0.])
    assert forecaster.get_coef() == approx(expected)


def test_get_coef_when_regressor_is_RandomForest():
    forecaster = ForecasterAutoregCustom(
                        regressor      = RandomForestRegressor(n_estimators=1, max_depth=2),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(6))
    expected = None
    assert forecaster.get_coef() is None
    

# Test method get_feature_importances
#-------------------------------------------------------------------------------
def test_get_feature_importances_when_regressor_is_RandomForest():
    forecaster = ForecasterAutoregCustom(
                        regressor      = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=123),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    expected = np.array([0.82142857, 0.        , 0.17857143, 0.        , 0.        ])
    assert forecaster.get_feature_importances() == approx(expected)
    
def test_get_feature_importances_when_regressor_is_GradientBoostingRegressor():
    forecaster = ForecasterAutoregCustom(
                        regressor      = GradientBoostingRegressor(n_estimators=1, max_depth=2, random_state=123),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    expected = np.array([0, 0, 0, 0.94736842, 0.05263158])
    assert forecaster.get_feature_importances() == approx(expected)
    
def test_get_feature_importances_when_regressor_is_linear_model():
    forecaster = ForecasterAutoregCustom(
                        regressor      = Lasso(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    expected = None
    assert forecaster.get_feature_importances() is None
    
    
    
# Test method _estimate_boot_interval()
#-------------------------------------------------------------------------------

def test_estimate_boot_interval_exception_when_steps_argument_is_less_than_1():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=0)
        
        
def test_estimate_boot_interval_exception_when_in_sample_residuals_argument_is_False_and_out_sample_residuals_attribute_is_empty():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, in_sample_residuals=False, n_boot=2)
        
        
def test_estimate_boot_interval_exception_when_forecaster_fitted_with_exog_but_exog_argument_is_None():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10), exog=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, n_boot=2)
        
        
def test_estimate_boot_interval_exception_when_forecaster_fitted_without_exog_but_exog_argument_is_not_None():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, exog=np.arange(10), n_boot=2)
        
        
def test_estimate_boot_interval_exception_when_lenght_exog_argument_is_less_than_steps():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10), exog=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, exog=np.arange(3), n_boot=2)
        

def test_estimate_boot_interval_exception_when_lenght_last_window_argument_is_less_than_max_lag_attribute():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    with pytest.raises(Exception):
        forecaster._estimate_boot_interval(steps=5, last_window=np.array([1,2]), n_boot=2)
        
        
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20., 20.]])
    results = forecaster._estimate_boot_interval(steps=1, in_sample_residuals=True, n_boot=2)  
    assert results == approx(expected)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    forecaster.fit(y=np.arange(10))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20, 20],
                        [23, 23]])
    results = forecaster._estimate_boot_interval(steps=2, in_sample_residuals=True, n_boot=2)  
    assert results == approx(expected)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20., 20.]])
    results = forecaster._estimate_boot_interval(steps=1, in_sample_residuals=False, n_boot=2)  
    assert results == approx(expected)
    
    
def test_estimate_boot_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[20, 20],
                        [23, 23]])
    results = forecaster._estimate_boot_interval(steps=2, in_sample_residuals=False, n_boot=2)  
    assert results == approx(expected)
    
    
    
# Test method predict_interval()
#-------------------------------------------------------------------------------
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[10., 20., 20.]])
    results = forecaster.predict_interval(steps=1, in_sample_residuals=True, n_boot=2)  
    assert results == approx(expected)
    
    
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                    )
    forecaster.fit(y=np.arange(10))
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[10, 20, 20],
                         [11, 23, 23]])
    results = forecaster.predict_interval(steps=2, in_sample_residuals=True, n_boot=2)  
    assert results == approx(expected)
    
    
def testpredict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[10., 20., 20.]])
    results = forecaster.predict_interval(steps=1, in_sample_residuals=False, n_boot=2)  
    assert results == approx(expected)
    
    
def testpredict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 5
                 )
    forecaster.fit(y=np.arange(10))
    forecaster.out_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=10)
    expected = np.array([[10, 20, 20],
                         [11, 23, 23]])
    results = forecaster.predict_interval(steps=2, in_sample_residuals=False, n_boot=2)  
    assert results == approx(expected)