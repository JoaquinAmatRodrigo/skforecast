################################################################################
#                        ForecasterEquivalentDate                              #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Tuple, Optional, Callable
import warnings
import logging
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from copy import copy

import skforecast
from ..utils import check_predict_input
from ..utils import preprocess_y
from ..utils import preprocess_last_window
from ..utils import expand_index

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


class ForecasterEquivalentDate():
    """
    This forecaster predicts future values based on the most recent equivalent
    date. It also allows to aggregate multiple past values of the equivalent
    date using a function (e.g. mean, median, max, min, etc.). The equivalent
    date is calculated by moving back in time a given number of steps (offset).
    The offset can be defined as an integer or as a pandas DateOffset. This
    approach is useful as a baseline, however, it is a simplistic method and may
    not capture complex underlying patterns.
    
    Parameters
    ----------
    offset : int, pandas.tseries.offsets.DateOffset
        Number of steps to go back in time to find the most recent equivalent
        date to the target period.
        If `offset` is an integer, it represents the number of steps to go back
        in time. For example, if the frequency of the time series is daily, 
        `offset = 7` means that the most recent data similar to the target
        period is the value observed 7 days ago.
        Pandas DateOffsets can also be used to move forward a given number of 
        valid dates. For example, Bday(2) can be used to move two business days
        backward. If the date does not start on a valid date, first it is moved 
        to a valid date. For example, if the date is a Saturday, it is moved to
        the previous Friday. Then, the offset is applied. If the result is a non
        valid date, it is moved to the next valid date. For example, if the date
        is a Sunday, it is moved to the next Monday. Find more information about
        offsets in https://pandas.pydata.org/docs/reference/offset_frequency.html.
    n_offsets : int, default 1
        Number of equivalent dates (multiple of offset) used in the prediction.
        If `offset` is greater than 1, the value at the equivalent dates is
        aggregated using the function `agg_func`. For example, if the frequency
        of the time series is daily, `offset = 7`, `n_offsets = 2` and
        `agg_func = np.mean`, the predicted value is the mean of the values
        observed 7 and 14 days ago.
    agg_func : callable, default `np.mean`
        Function used to aggregate the values of the equivalent dates when the
        number of equivalent dates (`n_offsets`) is greater than 1.
    forecaster_id : str, int, default `None`
        Name used as an identifier of the forecaster.
    
    Attributes
    ----------
    offset : int, pandas.tseries.offsets.DateOffset
        Number of steps to go back in time to find the most recent equivalent
        date to the target period.
        If `offset` is an integer, it represents the number of steps to go back
        in time. For example, if the frequency of the time series is daily, 
        `offset = 7` means that the most recent data similar to the target
        period is the value observed 7 days ago.
        Pandas DateOffsets can also be used to move forward a given number of 
        valid dates. For example, Bday(2) can be used to move two business days
        backward. If the date does not start on a valid date, first it is moved 
        to a valid date. For example, if the date is a Saturday, it is moved to
        the previous Friday. Then, the offset is applied. If the result is a non
        valid date, it is moved to the next valid date. For example, if the date
        is a Sunday, it is moved to the next Monday. Find more information about
        offsets in https://pandas.pydata.org/docs/reference/offset_frequency.html.
    n_offsets : int, default 1
        Number of equivalent dates (multiple of offset) used in the prediction.
        If `offset` is greater than 1, the value at the equivalent dates is
        aggregated using the function `agg_func`. For example, if the frequency
        of the time series is daily, `offset = 7`, `n_offsets = 2` and
        `agg_func = np.mean`, the predicted value is the mean of the values
        observed 7 and 14 days ago.
    agg_func : callable, default `np.mean`
        Function used to aggregate the values of the equivalent dates when the
        number of equivalent dates (`n_offsets`) is greater than 1.
    window_size : int
        Number of past values needed to include the last equivalent dates according
        to the `offset` and `n_offsets`.
    last_window : pandas Series
        This window represents the most recent data observed by the predictor
        during its training phase. It contains the past values needed to include
        the last equivalent date according the `offset` and `n_offsets`.
    index_type : type
        Type of index of the input used in training.
    index_freq : str
        Frequency of Index of the input used in training.
    training_range : pandas Index
        First and last values of index of the data used during training.
    fitted : bool
        Tag to identify if the regressor has been fitted (trained).
    transformer_y : Ignored
            Not used, present here for API consistency by convention.
    transformer_exog : Ignored
            Not used, present here for API consistency by convention.
    weight_func : Ignored
            Not used, present here for API consistency by convention.
    differentiation : Ignored
            Not used, present here for API consistency by convention.
    fit_kwargs : Ignored
            Not used, present here for API consistency by convention.
    creation_date : str
        Date of creation.
    fit_date : str
        Date of last fit.
    skforcast_version : str
        Version of skforecast library used to create the forecaster.
    python_version : str
        Version of python used to create the forecaster.
    forecaster_id : str, int
        Name used as an identifier of the forecaster.
    
    """
    
    def __init__(
        self,
        offset: Union[int, pd.tseries.offsets.DateOffset],
        n_offsets: int=1,
        agg_func: Callable=np.mean,
        forecaster_id: Optional[Union[str, int]]=None
    ) -> None:
        
        self.offset                  = offset
        self.n_offsets               = n_offsets
        self.agg_func                = agg_func
        self.regressor               = None
        self.transformer_y           = None
        self.transformer_exog        = None
        self.weight_func             = None
        self.differentiation         = None
        self.differentiator          = None
        self.source_code_weight_func = None
        self.last_window             = None
        self.index_type              = None
        self.index_freq              = None
        self.training_range          = None
        self.included_exog           = False
        self.exog_type               = None
        self.exog_dtypes             = None
        self.exog_col_names          = None
        self.X_train_col_names       = None
        self.in_sample_residuals     = None
        self.out_sample_residuals    = None
        self.fitted                  = False
        self.creation_date           = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                = None
        self.skforcast_version       = skforecast.__version__
        self.python_version          = sys.version.split(" ")[0]
        self.forecaster_id           = forecaster_id
       
        if not isinstance(self.offset, (int, pd.tseries.offsets.DateOffset)):
            raise Exception(
                "`offset` must be an integer or a pandas.tseries.offsets. "
                "Find more information about offsets in "
                "https://pandas.pydata.org/docs/reference/offset_frequency.html"
            )
        self.window_size = self.offset * self.n_offsets
        

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when a Forecaster object is printed.
        """

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Offset: {self.offset} \n"
            f"Number of offsets: {self.n_offsets} \n"
            f"Aggregation function: {self.agg_func.__name__} \n"
            f"Window size: {self.window_size} \n"
            f"Training range: {self.training_range.to_list() if self.fitted else None} \n"
            f"Training index type: {str(self.index_type).split('.')[-1][:-2] if self.fitted else None} \n"
            f"Training index frequency: {self.index_freq if self.fitted else None} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"Skforecast version: {self.skforcast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

        
    def fit(
        self,
        y: pd.Series,
        exog=None,
        store_in_sample_residuals=None
    ) -> None:
        """
        Training Forecaster.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        exog : Ignored
            Not used, present here for API consistency by convention.
        store_in_sample_residuals : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        None
        
        """

        if isinstance(self.offset, pd.tseries.offsets.DateOffset) and not isinstance(
            y.index, pd.DatetimeIndex
        ):
            raise Exception(
                "If `offset` is a pandas DateOffset, the index of `y` must be a "
                "pandas DatetimeIndex with a frequency."
            )
        
        # Reset values in case the forecaster has already been fitted.
        self.index_type     = None
        self.index_freq     = None
        self.last_window    = None
        self.fitted         = False
        self.training_range = None

        _, y_index = preprocess_y(y=y, return_values=True)
        self.fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range = y_index[[0, -1]]
        self.index_type = type(y_index)
        self.index_freq = (
            y_index.freqstr if isinstance(y_index, pd.DatetimeIndex) else y_index.step
        )
        if isinstance(self.offset, pd.tseries.offsets.DateOffset):
            # Calculate the window_size in steps for compatibility with the
            # check_predict_input function.
            last_window_start = (y_index[-1] - self.offset * self.n_offsets)
            self.window_size = int((y_index[-1] - last_window_start) / y_index.freq)
        
        # The last time window of training data is stored so that equivalent
        # dates are available when calling the `predict` method.
        self.last_window = y.iloc[-self.window_size:].copy()

            
    def predict(
        self,
        steps: int,
        last_window: Optional[pd.Series]=None,
        exog=None
    ) -> pd.Series:
        """
        Predict n steps ahead.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
        last_window : pandas Series, default `None`
            Past values needed to select the last equivalent dates according to 
            the offset. If `last_window = None`, the values stored in 
            `self.last_window` areused and the predictions start right after
            the training data.
        exog : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        
        """

        if last_window is None:
            last_window = copy(self.last_window)

        check_predict_input(
            forecaster_name  = None,
            steps            = steps,
            fitted           = self.fitted,
            included_exog    = None,
            index_type       = self.index_type,
            index_freq       = self.index_freq,
            window_size      = self.window_size,
            last_window      = last_window,
            last_window_exog = None,
            exog             = None,
            exog_type        = None,
            exog_col_names   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_col_names = None
        )

        last_window_values, last_window_index = preprocess_last_window(
                                                    last_window = last_window
                                                )
        
        if isinstance(self.offset, int):
            equivalent_indexes = np.tile(
                                    np.arange(-self.offset, 0),
                                    int(np.ceil(steps/self.offset))
                                )
            equivalent_indexes = equivalent_indexes[:steps]

            if self.n_offsets == 1:
                equivalent_values = last_window_values[equivalent_indexes]
                predictions = equivalent_values.ravel()

            if self.n_offsets > 1:
                equivalent_indexes = [
                    equivalent_indexes - n * self.offset 
                    for n
                    in np.arange(self.n_offsets)
                ]
                equivalent_indexes = np.vstack(equivalent_indexes)
                equivalent_values = last_window_values[equivalent_indexes]
                predictions = np.apply_along_axis(
                                self.agg_func,
                                axis=0,
                                arr=equivalent_values
                            )
            predictions = pd.Series(
                        data  = predictions,
                        index = expand_index(index=last_window_index,steps=steps),
                        name = 'pred'
                    )

        if isinstance(self.offset, pd.tseries.offsets.DateOffset):
            predictions_index = expand_index(index=last_window_index,steps=steps)
            max_allowed_date = last_window_index[-1]
            offset_dates = []
            # For every date in predictions_index, calculate the n offsets
            for date in predictions_index:
                selected_obsets = []
                while len(selected_obsets) < self.n_offsets:
                    offset_date = date - self.offset
                    if offset_date <= max_allowed_date:
                        selected_obsets.append(offset_date)
                    date = offset_date
                offset_dates.append(selected_obsets)
            offset_dates = np.array(offset_dates)
    
            # Select the values of the time series corresponding to the each
            # offset date. If the offset date is not in the time series, the
            # value is set to NaN.
            equivalent_values = (
                last_window.
                reindex(offset_dates.ravel())
                .to_numpy()
                .reshape(-1,self.n_offsets)
            )
            equivalent_values = pd.DataFrame(
                                    data    = equivalent_values,
                                    index   = predictions_index,
                                    columns = [f'offset_{i}' for i in range(self.n_offsets)]
                                )
            # Error if all values are missing
            if equivalent_values.isnull().all().all():
                raise Exception(
                    f"All equivalent values are missing. This is caused by using an offset "
                    f"({self.offset}) larger than the data available. Try decrease the "
                    f"size of the offset ({self.offset}), the number of offsets "
                    f"({self.n_offsets}) or increase the size of `last_window`. In "
                    f"backtesing this error may be caused by using an too small "
                    f"`initial_train_size`."
                )
            # Warning if equivalent values are missing
            incomplete_offsets = equivalent_values.isnull().any(axis=1)
            incomplete_offsets = incomplete_offsets[incomplete_offsets].index
            if not incomplete_offsets.empty:
                warnings.warn(
                    f"Steps: {incomplete_offsets.strftime('%Y-%m-%d').to_list()} "
                    f"are calculated with less than {self.n_offsets} offsets. To avoid "
                    f"this increase `last_window` size or decrease the number "
                    f"of offsets. The current configuration needs a total offset "
                    f"of {self.offset * self.n_offsets}."
                )
            aggregate_values = equivalent_values.apply(self.agg_func, axis=1)
            predictions = aggregate_values.rename('pred')
        
        return predictions