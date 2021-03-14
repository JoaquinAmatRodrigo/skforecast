
# Change Log
All notable changes to this project will be documented in this file.
 
 
## [0.1.6] - 2021-03-14
 
 
### Added

- Method `get_feature_importances()` to `skforecast.ForecasterAutoreg`.
- Added backtesting strategy in `grid_search_forecaster()`.
- Added `backtesting_forecast()` to `skforecast.model_selection`.
 
### Changed

- Method `create_lags()` return a matrix where the order of columns match the ascending order of lags. For example, column 0 contains the values of the minimum lag used as predictor.
- Renamed argument `X` to `last_window` in method `predict()`.
- Renamed `ts_cv_forecaster()` to `cv_forecaster()`.
 
### Fixed
 
## [0.1.4] - 2021-02-15
  
### Added

- Method `get_coef()` to `skforecast.ForecasterAutoreg`.
 
### Changed

 
### Fixed
