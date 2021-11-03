ForecasterBase:
- [x] Create a base class with all common methods

Forecaster:
- [x] Add name of predictor in get_coef and get_feature_importances

Backtesting `backtesting_forecaster()`:
- [x] Include an argument `refit` to decide if the forecaster is retrained in each iteration.
- [x] __backtesting_forecaster_refit()
- [x] __backtesting_forecaster_no_refit()

Backtesting `backtesting_forecaster_intervals()`:
- [x] Include an argument `refit` to decide if the forecaster is retrained in each iteration.
- [x] __backtesting_forecaster_refit()
- [x] __backtesting_forecaster_no_refit()

Grid search `grid_search_forecaster`:
- [x] Remove argument `method`
- [x] method 'cv' deprecated in favour to `backtesting_forecaster(refit=True)`
- [x] Add argument `refit`. This will be the strategy applied to `backtesting_forecaster`.

Model_selection_statsmodels:
- [ ] Change all inputs of `y` and `exog` to pandas Series and pandas DataFrame.

Unit testing:
- [ ] ForecasterAutoreg
- [ ] ForecasterAutoregCustom
- [ ] ForecasterAutoregMultioutput
- [ ] backtesting_forecaster

