Forecaster Classes
- [x] unified inputs `y` and `exog` to pandas Series and pandas Dataframe
- [x] unified `predict` and `predict_interval`
- [x] ForecasterBase as parent class
- [ ] add examples in docstrings (statsmodels)
- [ ] add references in docstrings (statsmodels)

Module model_selection
- [x] unified cross-validation and backtesting using argument `refit`
- [x] unified `grid_search` methods using argument `refit`

Model_selection_statsmodels:
- [x] Change all inputs of `y` and `exog` to pandas Series and pandas DataFrame
- [x] Combine `backtesting_sarimax_statsmodels` and `cv_sarimax_statsmodels` in a single function

Module utils
- [x] all static methods for checking and preprocessing inputs moved to module utils

Module plot
- [x] function to plot residuals

Unit testing:
- [x] ForecasterAutoreg
- [x] ForecasterAutoregCustom
- [ ] ForecasterAutoregMultioutput
- [ ] model_selection
- [x] utils

Others

- [x] add typing Optional[]
