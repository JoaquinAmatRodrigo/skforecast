# Understanding the forecaster parameters

Understanding what can be done when initializing a forecaster with skforecast can have a significant impact on the accuracy and effectiveness of the model. This guide highlights key considerations to keep in mind when initializing a forecaster and how these functionalities can be used to create more powerful and accurate forecasting models in Python.

We will explore the arguments that can be included in a `ForecasterAutoreg`, but this can be extrapolated to any of the skforecast forecasters.

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoreg(
                 regressor        = None,
                 lags             = None,
                 transformer_y    = None,
                 transformer_exog = None,
                 weight_func      = None,
                 differentiation  = None,
                 fit_kwargs       = None,
                 forecaster_id    = None
             )
```

!!! tip

    To be able to create and train a forecaster, at least `regressor` and `lags` must be specified.


## General parameters

### Regressor

Skforecast is a Python library that facilitates using scikit-learn regressors as multi-step forecasters and also works with any regressor compatible with the scikit-learn API. Therefore, any of these regressors can be used to create a forecaster:

+ [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

+ [LGBMRegressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

+ [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn)

+ [CatBoost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoreg(
                 regressor = RandomForestRegressor(),
                 lags      = None
             )
```


### Lags

To apply machine learning models to forecasting problems, the time series needs to be transformed into a matrix where each value is associated with a specific time window (known as lags) that precedes it. In the context of time series, a lag with respect to a time step *t* is defined as the value of the series at previous time steps. For instance, lag 1 represents the value at time step *t-1*, while lag *m* represents the value at time step *t-m*.

This transformation is essential for machine learning models to capture the dependencies and patterns that exist between past and future values in a time series. By using lags as input features, machine learning models can learn from the past and make predictions about future values. The number of lags used as input features in the matrix is an important hyperparameter that needs to be carefully tuned to obtain the best performance of the model.
<br><br>

<p style="text-align: center">
<img src="../img/transform_timeseries.gif" style="width: 500px;">
<br>
<font size="2.5"> <i>Time series transformation into a matrix of 5 lags and a vector with the value of the series that follows each row of the matrix.</i></font>
</p>

```python
# Create a forecaster using 5 lags
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoreg(
                 regressor = RandomForestRegressor(),
                 lags      = 5
             )
```


### Transformers

Skforecast has two arguments in all the forecasters that allow more detailed control over input data transformations. This feature is particularly useful as many machine learning models require specific data pre-processing transformations. For example, linear models may benefit from features being scaled, or categorical features being transformed into numerical values.

Both arguments expect an instance of a transformer (preprocessor) compatible with the scikit-learn preprocessing API with the methods: fit, transform, fit_transform and, inverse_transform.

More information: [Scikit-learn transformers and pipelines](https://skforecast.org/latest/user_guides/sklearn-transformers-and-pipeline.html).

!!! example

    In this example, a scikit-learn `StandardScaler` preprocessor is used for both the time series and the exogenous variables.

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

forecaster = ForecasterAutoreg(
                 regressor        = RandomForestRegressor(),
                 lags             = 5,
                 transformer_y    = StandardScaler(),
                 transformer_exog = StandardScaler()
             )
```


### Custom weights

The weight_func parameter allows you to define custom weights for each observation in your time series. These custom weights can be used to assign different levels of importance to different time periods within your time series data. For example, you might want to assign higher weights to recent data points and lower weights to older data points to emphasize the significance of recent observations in your forecasting model.

To use the weight_func parameter, you need to define a custom weighting function that takes the index of the time series as input and returns a weight for each observation. The function should return a weight of 0 for time periods that you want to assign low importance and a weight of 1 for time periods that you want to assign high importance. The function should be compatible with NumPy's array operations.

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Custom function to create weights
# ==============================================================================
def custom_weights(index):
    """
    Return 0 if index is between 2012-06-01 and 2012-10-21.
    """
    weights = np.where(
                  (index >= '2012-06-01') & (index <= '2012-10-21'),
                   0,
                   1
              )

    return weights

forecaster = ForecasterAutoreg(
                 regressor        = RandomForestRegressor(),
                 lags             = 5,
                 transformer_y    = None,
                 transformer_exog = None,
                 weight_func      = custom_weights,
             )
```


### Differentiation

Time series differentiation involves computing the differences between consecutive observations in the time series. When it comes to training forecasting models, differentiation offers the advantage of focusing on relative rates of change rather than directly attempting to model the absolute values. **Skforecast**, version 0.10.0 or higher, introduces a novel differentiation parameter within its Forecasters. 

More information: [Time series differentiation](https://skforecast.org/latest/faq/time-series-differentiation).

!!! warning

    The `differentiation` parameter is only available for the `ForecasterAutoreg` and `ForecasterAutoregCustom`, in the following versions it will be incorporated to the rest of the Forecasters.

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

forecaster = ForecasterAutoreg(
                 regressor        = RandomForestRegressor(),
                 lags             = 5,
                 transformer_y    = None,
                 transformer_exog = None,
                 weight_func      = None,
                 differentiation  = 1,
             )
```


### Inclusion of kwargs in the regressor fit method

Some regressors include the possibility to add some additional configuration during the fitting method. The predictor parameter `fit_kwargs` allows these arguments to be set when the forecaster is declared.

!!! example

    The following example demonstrates the inclusion of categorical features in an LGBM regressor. This must be done during the `LGBMRegressor` fit method. [Fit parameters lightgbm](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.fit)

More information: [Categorical features](https://skforecast.org/latest/user_guides/categorical-features.html#native-implementation-for-categorical-features).

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from lightgbm import LGBMRegressor

forecaster = ForecasterAutoreg(
                 regressor        = LGBMRegressor(),
                 lags             = 5,
                 transformer_y    = None,
                 transformer_exog = None,
                 weight_func      = None,
                 differentiation  = None,
                 fit_kwargs       = {'categorical_feature': ['exog_1', 'exog_2']}
             )
```


### Forecaster ID

Name used as an identifier of the forecaster. It may be used, for example to identify the time series being modeled.

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoreg(
                 regressor        = RandomForestRegressor(),
                 lags             = 5,
                 transformer_y    = None,
                 transformer_exog = None,
                 weight_func      = None,
                 differentiation  = None,
                 fit_kwargs       = None,
                 forecaster_id    = 'my_forecaster'
             )
```


## Direct multi-step parameters

For the Forecasters that follow a [direct multi-step strategy](https://skforecast.org/latest/introduction-forecasting/introduction-forecasting#direct-multi-step-forecasting) (`ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`), there are two additional parameters in addition to those mentioned above.

### Steps

Direct multi-step forecasting consists of training a different model for each step of the forecast horizon. For example, to predict the next 5 values of a time series, 5 different models are trained, one for each step. As a result, the predictions are independent of each other. 

The number of models to be trained is specified by the `steps` parameter.

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoregDirect(
                 regressor        = RandomForestRegressor(),
                 steps            = 5,
                 lags             = 5,
                 transformer_y    = None,
                 transformer_exog = None,
                 weight_func      = None,
                 fit_kwargs       = None,
                 n_jobs           = 'auto',
                 forecaster_id    = 'my_forecaster'
             )
```


### Number of jobs

The `n_jobs` parameter allows multi-process parallelization to train regressors for all `steps` simultaneously. 

The benefits of parallelization depend on several factors, including the regressor used, the number of fits to be performed, and the volume of data involved. When the `n_jobs` parameter is set to `'auto'`, the level of parallelization is automatically selected based on heuristic rules that aim to choose the best option for each scenario.

For a more detailed look at parallelization, visit [Parallelization in skforecast](https://skforecast.org/latest/faq/parallelization-skforecast).

```python
# Create a forecaster
# ==============================================================================
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoregDirect(
                 regressor        = RandomForestRegressor(),
                 steps            = 5,
                 lags             = 5,
                 transformer_y    = None,
                 transformer_exog = None,
                 weight_func      = None,
                 fit_kwargs       = None,
                 n_jobs           = 'auto',
                 forecaster_id    = 'my_forecaster'
             )
```
