
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from typing import Any
from typing_extensions import Self

class TimeSeriesDifferentiator(BaseEstimator, TransformerMixin):
    """
    Transforms a time series into a differentiated time series of order n.
    It also reverts the differentiation.

    Parameters
    ----------
    order : int
        Order of differentiation.

    Attributes
    ----------
    initial_values : list
        List with the initial value of the time series after each differentiation.
        This is used to revert the differentiation.
    last_values : list
        List with the last value of the time series after each differentiation.
        This is used to revert the differentiation of a new window of data. A new
        window of data is a time series that starts right after the time series
        used to fit the transformer.
    order : int
        Order of differentiation.   

    """
    
    def __init__(self, order: int=1) -> None:

        if not isinstance(order, int):
            raise TypeError(f"Parameter 'order' must be an integer. Found {type(order)}.")
        if order < 1:
            raise ValueError(f"Parameter 'order' must be an integer greater than 0. Found {order}.")

        self.order = order
        self.initial_values = []
        self.last_values = []
    

    def fit(self, X: np.ndarray, y: Any=None) -> Self:
        """
        Fits the transformer. This method only removes the values stored in
        `self.initial_values`.

        Parameters
        ----------
        X : np.ndarray
            Time series to be differentiated.
        y : None
            Ignored.

        Returns
        -------
        self : TimeSeriesDifferentiator
        """

        self.initial_values = []
        self.last_values = []

        return self
    

    def transform(self, X: np.ndarray, y: Any=None) -> np.ndarray:
        """
        Transforms a time series into a differentiated time series of order n and
        stores the values needed to revert the differentiation.

        Parameters
        ----------
        X : numpy.ndarray
            Time series to be differentiated.
        y : None
            Ignored.
        
        Returns
        -------
        X_diff : numpy.ndarray
            Differentiated time series. The length of the array is the same as
            the original time series but the first n=`order` values are nan.

        """
        for i in range(self.order):
            if i == 0:
                self.initial_values.append(X[0])
                self.last_values.append(X[-1])
                X_diff = np.diff(X, n=1)
            else:
                self.initial_values.append(X_diff[0])
                self.last_values.append(X_diff[-1])
                X_diff = np.diff(X_diff, n=1)
                
        X_diff = np.append((np.full(shape=self.order, fill_value=np.nan)), X_diff)

        return X_diff
    
    
    def inverse_transform(self, X: np.ndarray, y: Any=None) -> np.ndarray:
        """
        Reverts the differentiation. To do so, the input array is assumed to be
        a differentiated time series of order n that starts right after the
        the time series used to fit the transformer.

        Parameters
        ----------
        X : numpy.ndarray
            Differentiated time series.
        y : None
            Ignored.
        
        Returns
        -------
        X_diff : numpy.ndarray
            Reverted differentiated time series.
        """

        # Remove initial nan values if present
        X = X[np.argmax(~np.isnan(X)):]
        for i in range(self.order):
            if i == 0:
                X_undiff = np.insert(X, 0, self.initial_values[-1])
                X_undiff = np.cumsum(X_undiff, dtype=float)
            else:
                X_undiff = np.insert(X_undiff, 0, self.initial_values[-(i+1)])
                X_undiff = np.cumsum(X_undiff, dtype=float)

        return X_undiff


    def inverse_transform_next_window(
        self,
        X: np.ndarray,
        y: Any=None
    )  -> np.ndarray:
        """
        Reverts the differentiation. The input array `x` is assumed to be a 
        differentiated time series of order n that starts right after the
        the time series used to fit the transformer.

        Parameters
        ----------
        X : np.ndarray
            Differentiated time series. It is assumed o start right after
            the time series used to fit the transformer.
        y : Any, optional
            Ignored.

        Returns
        -------
        np.ndarray
            Reverted differentiated time series.
        """

        # Remove initial nan values if present
        X = X[np.argmax(~np.isnan(X)):]

        for i in range(self.order):
            if i == 0:
                X_undiff = np.cumsum(X, dtype=float) + self.last_values[-1]
            else:
                X_undiff = np.cumsum(X_undiff, dtype=float) + self.last_values[-(i+1)]
        
        return X_undiff

