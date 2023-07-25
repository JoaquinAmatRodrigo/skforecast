
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from typing import Any, Union

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
        List with the initial value the time series after each differentiation.
        This is used to revert the differentiation.
    order : int
        Order of differentiation.   

    """
    
    def __init__(self, order: int=1):

        self.order = order
        self.initial_values = []
    

    def fit(self, X: np.ndarray, y: Any=None):
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
        self : object
        """

        self.initial_values = []

        return self
    

    def transform(self, X: np.ndarray, y: Any=None):
        """
        Transforms a time series into a differentiated time series of order n.

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
                X_diff = np.diff(X, n=1)
            else:
                self.initial_values.append(X_diff[0])
                X_diff = np.diff(X_diff, n=1)
                
        X_diff = np.append((np.full(shape=self.order, fill_value=np.nan)), X_diff)

        return X_diff
    
    
    def inverse_transform(self, X: np.ndarray, y: Any=None):
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
                X_diff = np.insert(X, 0, self.initial_values[self.order-1])
                X_diff = np.cumsum(X_diff, dtype=float)
            else:
                X_diff = np.insert(X_diff, 0, self.initial_values[self.order-i-1])
                X_diff = np.cumsum(X_diff, dtype=float)

        return X_diff