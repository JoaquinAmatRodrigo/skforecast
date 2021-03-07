
# Change Log
All notable changes to this project will be documented in this file.
 
 
## [0.1.6] - unreleased
 
 
### Added

- Method `get_feature_importances()` to `skforecast.ForecasterAutoreg`.
- Added backtesting strategy in `grid_search_forecaster()`.
 
### Changed

- Method `create_lags()` return a matrix where the order of columns match the ascending order of lags. For example, column 0 contains the values of the minimum lag used as predictor.
- Rename argument `X` to last_window in method `predict()`.
 
### Fixed
 
## [0.1.4] - 2017-03-15
  
### Added

- Method `get_coef()` to `skforecast.ForecasterAutoreg`.
 
### Changed

 
### Fixed
