# Backtesting

Backtesting is a type of cross-validation making predictions about the past. It is helpful to test the predictive model on historical data.

**Backtesting with refit**

The model is trained each time before making predictions. With this configuration, the model uses all the information available so far. It is a variation of the standard cross-validation but, instead of making a random distribution of the observations, the training set increases sequentially, maintaining the temporal order of the data.

<p align="center"><img src="../img/diagram-backtesting-refit.png" style="width: 500px;"></p>

<p align="center"><img src="../img/backtesting_refit.gif" style="width: 600px;"></p>

**Backtesting with refit and fixed train size**

A technique similar to backtesting with refit but, in this case, the training set doesn't increase sequentially. This is also known as time series cross-validation or walk-forward validation.

<p align="center"><img src="../img/diagram-backtesting-refit-fixed-train-size.png" style="width: 500px;"></p>

<p align="center"><img src="../img/backtesting_refit_fixed_train_size.gif" style="width: 600px;"></p>


**Backtesting without refit**

After an initial train, the model is used sequentially without updating it and following the temporal order of the data. This strategy has the advantage of being much faster since the model is trained only once. However, the model does not incorporate the latest information available, so it may lose predictive capacity over time.

<p align="center"><img src="../img/diagram-backtesting-no-refit.png" style="width: 500px;"></p>

<p align="center"><img src="../img/backtesting_no_refit.gif" style="width: 600px;"></p>