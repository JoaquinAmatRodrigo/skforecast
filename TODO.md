ForecasterBase:
    [] Create a base class with all common methods
    [] Add name of predictor in get_coef and get_feature_importances

Backtesting `backtesting_forecaster()`:
    [] Include an argument `refit` to decide if the forecaster is retrained in each iteration.
        [] __backtesting_forecaster_refit()
        [] __backtesting_forecaster_no_refit()

Cross validation `cv_forecaster()`:
    [] Deprecated in favour to `backtesting_forecaster(refit=True)`

Grid search `grid_search_forecaster`:
    [] Remove argument `method`
    [] Add argument `refit`. This will be the strategy applied to `backtesting_forecaster`.

Model_selection_statsmodels:
    [] Change all inputs of `y` and `exog` to pandas Series and pandas DataFrame.

Unit testing:
    [] ForecasterAutoreg
    [] ForecasterAutoregCustom
    [] ForecasterAutoregMultioutput
    [] backtesting_forecaster

