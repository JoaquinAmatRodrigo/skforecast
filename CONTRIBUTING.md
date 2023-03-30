# Contributing to Skforecast

## How to Contribute

Skforecast is an open source project supported by a community that will gratefully and humbly accept any contribution you can make to the project. Big or small, any contribution makes a big difference; and if you have never contributed to an open source project before, we hope you will start with Skforecast!

Primarily, Skforecast development consists of adding and creating new *Forecasters*, new validation strategies or improving the performance of the current code. However, there are many other ways to contribute:


- Submit a bug report or feature request on [GitHub Issues](https://github.com/JoaquinAmatRodrigo/skforecast/issues).
- Contribute a Jupyter notebook to our [examples](https://joaquinamatrodrigo.github.io/skforecast/0.7.0/examples/examples.html).
- Write [unit or integration tests]() for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

As you can see, there are lots of ways to get involved and we would be very happy for you to join us! Before you start, please open an issue with with a brief description of the proposal so we can align all together.


## Testing

To run the test suite, first install the testing dependencies that are located in main folder:

```
$ pip install -r requirements_test.txt
```

All unit test can be run at once as follows from the project root:

```
$ pytest -vv
```

The tests do take a while to run, so during normal development it is recommended that you only run the tests for the test file you are writing:

```
$ pytest new_module/tests/test_module.py
```

This will help a lot to ensure that new code do not affect the already existent functionalities of the library.

## Documentation

Docstring documentation must be included in every class and function. Skforecast uses MkDocs to build documentation, and follow the numpydoc format (similar to scikit-learn). The primary location of your docstring should be right under the class definition, here are two examples:

```python
class ForecasterAutoreg(ForecasterBase):
    """
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster.

    Parameters
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.

    lags : int, list, 1d numpy ndarray, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list`, `numpy ndarray` or `range`: include only lags present in `lags`,
            all elements must be int.

    transformer_y : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster. 

    transformer_exog : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.

    weight_func : Callable, default `None`
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method. The resulting `sample_weight` cannot have negative values.

    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.

    Attributes
    ----------
    regressor : regressor or pipeline compatible with the scikit-learn API
        An instance of a regressor or pipeline compatible with the scikit-learn API.

    lags : numpy ndarray
        Lags used as predictors.

    transformer_y : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API with methods: fit, transform, fit_transform and inverse_transform.
        ColumnTransformers are not allowed since they do not have inverse_transform method.
        The transformation is applied to `y` before training the forecaster.

    transformer_exog : object transformer (preprocessor), default `None`
        An instance of a transformer (preprocessor) compatible with the scikit-learn
        preprocessing API. The transformation is applied to `exog` before training the
        forecaster. `inverse_transform` is not available when using ColumnTransformers.

    weight_func : Callable
        Function that defines the individual weights for each sample based on the
        index. For example, a function that assigns a lower weight to certain dates.
        Ignored if `regressor` does not have the argument `sample_weight` in its `fit`
        method.
        **New in version 0.6.0**

    source_code_weight_func : str
        Source code of the custom function used to create weights.
        **New in version 0.6.0**

    max_lag : int
        Maximum value of lag included in `lags`.

    window_size : int
        Size of the window needed to create the predictors. It is equal to
        `max_lag`.

    last_window : pandas Series
        Last window the forecaster has seen during trained. It stores the
        values needed to predict the next `step` right after the training data.

    index_type : type
        Type of index of the input used in training.

    index_freq : str
        Frequency of Index of the input used in training.

    training_range : pandas Index
        First and last values of index of the data used during training.

    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.

    exog_type : type
        Type of exogenous variable/s used in training.

    exog_col_names : list
        Names of columns of `exog` if `exog` used in training was a pandas
        DataFrame.

    X_train_col_names : list
        Names of columns of the matrix created internally for training.

    in_sample_residuals : numpy ndarray
        Residuals of the model when predicting training data. Only stored up to
        1000 values. If `transformer_y` is not `None`, residuals are stored in the
        transformed scale.

    out_sample_residuals : numpy ndarray
        Residuals of the model when predicting non training data. Only stored
        up to 1000 values. If `transformer_y` is not `None`, residuals
        are assumed to be in the transformed scale. Use `set_out_sample_residuals` to
        set values.

    fitted : bool
        Tag to identify if the regressor has been fitted (trained).

    creation_date : str
        Date of creation.

    fit_date : str
        Date of last fit.

    skforcast_version : str
        Version of skforecast library used to create the forecaster.

    python_version : str
        Version of python used to create the forecaster.

    forecaster_id : str, int default `None`
        Name used as an identifier of the forecaster.

    """
```

```python
def preprocess_y(
    y: pd.Series
) -> Tuple[np.ndarray, pd.Index]:
    """
    Returns values and index of series separately. Index is overwritten 
    according to the next rules:
        If index is of type DatetimeIndex and has frequency, nothing is 
        changed.
        If index is of type RangeIndex, nothing is changed.
        If index is of type DatetimeIndex but has no frequency, a 
        RangeIndex is created.
        If index is not of type DatetimeIndex, a RangeIndex is created.

    Parameters
    ----------        
    y : pandas Series
        Time series.

    Returns 
    -------
    y_values : numpy ndarray
        Numpy array with values of `y`.

    y_index : pandas Index
        Index of `y` modified according to the rules.

    """
```