Backtesting `backtesting_forecaster()`:
    [x] The current implementation of `backtesting_forecaster()` always do an initial training. Allow `backtesting_forecaster()` to backtest forecasters already trained.
    [] Include an argument `refit` to decide if the forecaster is retrained in each iteration.
    [x] Currently, backtesting ForecasterAutoregMultiOutput do not allow for incomplete folds. Allow it including dummy values for the remaining steps of the last fold and removing then the corresponding predictions.
    [x] Create a function to select the metric  
    [] Add option to update out_sample_residuals at the end of the proces of backtesting

Cross validation `cv_forecaster()`:
    [x] Create a function to select the metric  
    [] Add option to update out_sample_residuals at the end of the proces of cv

ForecasterCustom
    [X] Remove class ForecasterCustom



