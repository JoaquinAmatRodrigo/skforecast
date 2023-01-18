# Unit test predict_bootstrapping ForecasterAutoregCustom
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# Fixtures
# np.random.seed(123)
# y = np.random.rand(50)
# exog = np.random.rand(50)
y = pd.Series(
        data = np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                         0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                         0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                         0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                         0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                         0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                         0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                         0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                         0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                         0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]
            ),
        name = 'y'
    )

exog = pd.Series(
           data = np.array([0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                            0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
                            0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
                            0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
                            0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
                            0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
                            0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
                            0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
                            0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
                            0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601]
               ),
           name = 'exog'
       )

exog_predict = pd.Series(
                  data = np.array([0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                                   0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234]
                      ),
                  name = 'exog',
                  index = pd.RangeIndex(start=50, stop=60)
              )
        

def create_predictors(y):
    """
    Create first 3 lags of a time series.
    """
    
    lags = y[-1:-4:-1]
    
    return lags 


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step is predicted using in-sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 3
                 )
    forecaster.fit(y=y, exog=exog)
    expected = pd.DataFrame(
                    data    = np.array([[0.39132925, 0.71907802, 0.45297104, 0.52687879]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=51)
                )
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, exog=exog_predict, in_sample_residuals=True)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps are predicted using in-sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 3
                 )
    forecaster.fit(y=y, exog=exog)
    expected = pd.DataFrame(
                    data    = np.array([[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                        [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, exog=exog_predict, in_sample_residuals=True)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step is predicted using out-sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 3
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    expected = pd.DataFrame(
                    data    = np.array([[0.39132925, 0.71907802, 0.45297104, 0.52687879]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=51)
                )
    results = forecaster.predict_bootstrapping(steps=1, n_boot=4, exog=exog_predict, in_sample_residuals=False)

    pd.testing.assert_frame_equal(expected, results)
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps are predicted using out-sample residuals.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor      = LinearRegression(),
                        fun_predictors = create_predictors,
                        window_size    = 3
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals = forecaster.in_sample_residuals
    expected = pd.DataFrame(
                    data    = np.array([[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                        [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    results = forecaster.predict_bootstrapping(steps=2, n_boot=4, exog=exog_predict, in_sample_residuals=False)

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer_fixed_to_zero():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included, both
    inputs are transformed and in-sample residuals are fixed to 0.
    """
    forecaster = ForecasterAutoregCustom(
                        regressor        = LinearRegression(),
                        fun_predictors   = create_predictors,
                        window_size      = 3,
                        transformer_y    = StandardScaler(),
                        transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.in_sample_residuals = np.full_like(forecaster.in_sample_residuals, fill_value=0)
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                    data = np.array([[0.67998879, 0.67998879, 0.67998879, 0.67998879],
                                    [0.46135022, 0.46135022, 0.46135022, 0.46135022]]),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_bootstrappingwhen regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """

    forecaster = ForecasterAutoregCustom(
                        regressor        = LinearRegression(),
                        fun_predictors   = create_predictors,
                        window_size      = 3,
                        transformer_y    = StandardScaler(),
                        transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_bootstrapping(steps=2, exog=exog_predict, n_boot=4, in_sample_residuals=True)
    expected = pd.DataFrame(
                    data = np.array(
                                [[ 0.39132925,  0.71907802,  0.45297104,  0.52687879],
                                [-0.06983175,  0.40010008,  0.06326565,  0.05134409]]
                        ),
                    columns = [f"pred_boot_{i}" for i in range(4)],
                    index   = pd.RangeIndex(start=50, stop=52)
                )
    pd.testing.assert_frame_equal(expected, results)
    