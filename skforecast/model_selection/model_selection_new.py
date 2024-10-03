################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import re
import os
from copy import deepcopy, copy
import logging
from typing import Union, Tuple, Optional, Callable, Any
import warnings
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

from ..metrics import add_y_train_argument, _get_metric
from ..exceptions import LongTrainingWarning
from ..exceptions import IgnoredArgumentWarning
from ..utils import check_backtesting_input
from ..utils import initialize_lags_grid
from ..utils import initialize_lags
from ..utils import select_n_jobs_backtesting

logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


class TimeSeriesFold():
    """
    Class to split time series data into train and test folds. 
    When used within a backtesting or hiperparameter search, arguments
    'initial_train_size', 'window_size' and 'differentiation' are not required
    since they are set automatically by the backtesting or hyperparameter search
    functions.

    Parameters
    ----------
    steps : int
        Number of observations used to be predicted in each fold. This is also commonly
        referred to as the forecast horizon or test size.
    initial_train_size : int, default=None
        Number of observations used for initial training. If `None` or 0, the initial
        forecaster is not trained in the first fold.
    window_size : int
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int, default=None
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    refit : bool or int, default=False
        Whether to refit the forecaster in each fold.

        - If `True`, the forecaster is refitted in each fold.
        - If `False`, the forecaster is trained only in the first fold.
        - If an integer, the forecaster is trained in the first fold and then refitted
          every `refit` folds.
    fixed_train_size : bool, default=True
        Whether the training size is fixed or increases in each fold.
    gap : int, default=0
        Number of observations between the end of the training set and the start of the
        test set.
    skip_folds : int or list, default=None
        Number of folds to skip.

        - If an integer, every 'skip_folds'-th is returned.
        - If a list, the indexes of the folds to skip.

        For example, if `skip_folds=3` and there are 10 folds, the returned folds are
        0, 3, 6, and 9. If `skip_folds=[1, 2, 3]`, the returned folds are 0, 4, 5, 6, 7,
        8, and 9.
    allow_incomplete_fold : bool, default=True
        Whether to allow the last fold to include fewer observations than `steps`.
        If `False`, the last fold is excluded if it is incomplete.
    return_all_indexes : bool, default=False
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool, default=True
        Whether to print information about generated folds.

    Attributes
    ----------
    steps : int
        Number of observations used to be predicted in each fold. This is also commonly
        referred to as the forecast horizon or test size.
    initial_train_size : int, default=None
        Number of observations used for initial training. If `None` or 0, the initial
        forecaster is not trained in the first fold.
    window_size : int
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int, default=None
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    refit : bool or int, default=False
        Whether to refit the forecaster in each fold.

        - If `True`, the forecaster is refitted in each fold.
        - If `False`, the forecaster is trained only in the first fold.
        - If an integer, the forecaster is trained in the first fold and then refitted
          every `refit` folds.
    fixed_train_size : bool, default=True
        Whether the training size is fixed or increases in each fold.
    gap : int, default=0
        Number of observations between the end of the training set and the start of the
        test set.
    skip_folds : int or list, default=None
        Number of folds to skip.

        - If an integer, every 'skip_folds'-th is returned.
        - If a list, the indexes of the folds to skip.

        For example, if `skip_folds=3` and there are 10 folds, the returned folds are
        0, 3, 6, and 9. If `skip_folds=[1, 2, 3]`, the returned folds are 0, 4, 5, 6, 7,
        8, and 9.
    allow_incomplete_fold : bool, default=True
        Whether to allow the last fold to include fewer observations than `steps`.
        If `False`, the last fold is excluded if it is incomplete.
    return_all_indexes : bool, default=False
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool, default=True
        Whether to print information about generated folds.

    Notes
    -----
    Returned values are the positions of the observations and not the actual values of
    the index, so they can be used to slice the data directly using iloc. For example,
    if the input series is `X = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`, the 
    `initial_train_size = 3`, `window_size = 2`, `steps = 4`, and `gap = 1`,
    the output of the first fold will: [[0, 3], [1, 3], [3, 8], [4, 8], True].

    The first list `[0, 3]` indicates that the training set goes from the first to the
    third observation. The second list `[1, 3]` indicates that the last window seen by
    the forecaster during training goes from the second to the third observation. The
    third list `[3, 8]` indicates that the test set goes from the fourth to the eighth
    observation. The fourth list `[4, 8]` indicates that the test set including the gap
    goes from the fifth to the eighth observation. The boolean `False` indicates that the
    forecaster should not be trained in this fold.

    Following the python convention, the start index is inclusive and the end index is
    exclusive. This means that the last index is not included in the slice.

    """

    def __init__(
            self,
            steps: int,
            initial_train_size: Optional[int] = None,
            window_size: Optional[int] = None,
            differentiation: Optional[int] = None,
            refit: Union[bool, int] = False,
            fixed_train_size: bool = True,
            gap: int = 0,
            skip_folds: Optional[Union[int, list]] = None,
            allow_incomplete_fold: bool = True,
            return_all_indexes: bool = False,
            verbose: bool = True
    ) -> None:

        self._validate_params(
            steps,
            initial_train_size,
            window_size,
            differentiation,
            refit, 
            fixed_train_size,
            gap,
            skip_folds,
            allow_incomplete_fold, 
            return_all_indexes,
        )

        self.initial_train_size = initial_train_size
        self.steps = steps
        self.window_size = window_size
        self.differentiation = differentiation
        self.refit = refit
        self.fixed_train_size = fixed_train_size
        self.gap = gap
        self.skip_folds = skip_folds
        self.allow_incomplete_fold = allow_incomplete_fold
        self.return_all_indexes = return_all_indexes
        self.verbose = verbose

    def _validate_params(
        self,
        steps: int,
        initial_train_size: Optional[int] = None,
        window_size: Optional[int] = None,
        differentiation: Optional[int] = None,
        refit: Union[bool, int] = False,
        fixed_train_size: bool = True,
        gap: int = 0,
        skip_folds: Optional[Union[int, list]] = None,
        allow_incomplete_fold: bool = True,
        return_all_indexes: bool = False,
        verbose: bool = True
    ) -> None:
                
        """
        Validate all input parameters to ensure correctness.
        """

        if (
            not isinstance(window_size, (int, np.integer, type(None)))
            or window_size is not None
            and window_size < 1
        ):
            raise ValueError(
                f"`window_size` must be an integer greater than 0. Got {window_size}."
            )
        if not isinstance(initial_train_size, (int, np.integer)) and initial_train_size is not None:
            raise ValueError(
                f"`initial_train_size` must be an integer or None. Got {initial_train_size}."
            )
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError(
                f"`steps` must be an integer greater than 0. Got {steps}."
            )
        if not isinstance(refit, (bool, int, np.integer)):
            raise ValueError(
                f"`refit` must be a boolean or an integer. Got {refit}."
            )
        if not isinstance(fixed_train_size, bool):
            raise ValueError(
                f"`fixed_train_size` must be a boolean. Got {fixed_train_size}."
            )
        if not isinstance(gap, (int, np.integer)) or gap < 0:
            raise ValueError(
                f"`gap` must be an integer greater than or equal to 0. Got {gap}."
            )
        if skip_folds is not None:
            if not isinstance(skip_folds, (int, np.integer, list, type(None))):
                raise ValueError(
                    f"`skip_folds` must be an integer, list or None. Got {skip_folds}."
                )
            if isinstance(skip_folds, (int, np.integer)) and skip_folds < 1:
                raise ValueError(
                    f"`skip_folds` must be an integer greater than 0. Got {skip_folds}."
                )
            if isinstance(skip_folds, list) and any([x < 1 for x in skip_folds]):
                raise ValueError(
                    f"`skip_folds` list must contain integers greater than or equal to 1. "
                    f"The first fold is always needed to train the forecaster. "
                    f"Got {skip_folds}."
                )
        if not isinstance(allow_incomplete_fold, bool):
            raise ValueError(
                f"`allow_incomplete_fold` must be a boolean. Got {allow_incomplete_fold}."
            )
        if not isinstance(return_all_indexes, bool):
            raise ValueError(
                f"`return_all_indexes` must be a boolean. Got {return_all_indexes}."
            )
        if differentiation is not None:
            if not isinstance(differentiation, (int, np.integer)) or differentiation < 0:
                raise ValueError(
                    f"differentiation must be None or an integer greater than or equal to 0. "
                    f"Got {differentiation}."
                )
        if not isinstance(verbose, bool):
            raise ValueError(
                f"`verbose` must be a boolean. Got {verbose}."
            )

    def __repr__(
        self
    ) -> str:
        """
        Information displayed when printed.
        """
            
        return (
            f"TimeSeriesFold(\n"
            f"    steps={self.steps},\n"
            f"    initial_train_size={self.initial_train_size},\n"
            f"    window_size={self.window_size},\n"
            f"    differentiation={self.differentiation},\n"
            f"    refit={self.refit},\n"
            f"    fixed_train_size={self.fixed_train_size},\n"
            f"    gap={self.gap},\n"
            f"    skip_folds={self.skip_folds},\n"
            f"    allow_incomplete_fold={self.allow_incomplete_fold},\n"
            f"    return_all_indexes={self.return_all_indexes},\n"
            f"    verbose={self.verbose}\n"
            f")"
        )


    def split(
            self,
            X: Union[pd.Series, pd.DataFrame, pd.Index, dict],
            externally_fitted: bool = False,
            as_pandas: bool = False
    ) -> Union[list, pd.DataFrame]:
        """
        Split the time series data into train and test folds.

        Parameters
        ----------
        X : pandas Series, DataFrame, Index, or dictionary
            Time series data or index to split.
        externally_fitted : bool, default=False
            If True, the forecaster is assumed to be already fitted so no training is
            done in the first fold.
        as_pandas : bool, default=False
            If True, the folds are returned as a DataFrame. This is useful to visualize
            the folds in a more interpretable way.

        Returns
        -------
        list, pd.DataFrame
            A list of lists containing the indices (position) for for each fold. Each list
            contains 4 lists and a boolean with the following information:

            - [train_start, train_end]: list with the start and end positions of the
            training set.
            - [last_window_start, last_window_end]: list with the start and end positions
            of the last window seen by the forecaster during training. The last window
            is used to generate the lags use as predictors. If `diferentiation` is
            included, the interval is extended as many observations as the
            differentiation order. If the arguemnt `window_size` is `None`, this list is
            empty.
            - [test_start, test_end]: list with the start and end positions of the test
            set. These are the observations used to evaluate the forecaster.
            - [test_start_with_gap, test_end_with_gap]: list with the start and end
            positions of the test set including the gap. The gap is the number of
            observations bwetween the end of the training set and the start of the test
            set.
            - fit_forecaster: boolean indicating whether the forecaster should be fitted
            in this fold.

            It is important to note that the returned values are the positions of the
            observations and not the actual values of the index, so they can be used to
            slice the data directly using iloc.

            If `as_pandas` is `True`, the folds are returned as a DataFrame with the
            following columns: 'fold', 'train_start', 'train_end', 'last_window_start',
            'last_window_end', 'test_start', 'test_end', 'test_start_with_gap',
            'test_end_with_gap', 'fit_forecaster'.

            Following the python convention, the start index is inclusive and the end
            index is exclusive. This means that the last index is not included in the
            slice.

        """

        if not isinstance(X, (pd.Series, pd.DataFrame, pd.Index, dict)):
            raise ValueError(
                f"X must be a pandas Series, DataFrame, Index or a dictionary. "
                f"Got {type(X)}."
            )
        
        if self.window_size is None:
            warnings.warn(
                "Last window cannot be calculated because `window_size` is None. "
            )

        index = self._extract_index(X)
        idx = range(len(X))
        folds = []
        i = 0
        last_fold_excluded = False

        if self.window_size is not None and self.differentiation is not None:
            window_size = self.window_size + self.differentiation
        else:
            window_size = self.window_size

        if len(index) < self.initial_train_size + self.steps:
            raise ValueError(
                f"The time series must have at least `initial_train_size + steps` "
                f"observations. Got {len(index)} observations."
            )

        while self.initial_train_size + (i * self.steps) + self.gap < len(index):

            if self.refit:
                # If `fixed_train_size` the train size doesn't increase but moves by 
                # `steps` positions in each iteration. If `False`, the train size
                # increases by `steps` in each iteration.
                train_iloc_start = i * (self.steps) if self.fixed_train_size else 0
                train_iloc_end = self.initial_train_size + i * (self.steps)
                test_iloc_start = train_iloc_end
            else:
                # The train size doesn't increase and doesn't move.
                train_iloc_start = 0
                train_iloc_end = self.initial_train_size
                test_iloc_start = self.initial_train_size + i * (self.steps)
            
            if window_size is not None:
                last_window_iloc_start = test_iloc_start - window_size
            test_iloc_end = test_iloc_start + self.gap + self.steps
        
            partitions = [
                idx[train_iloc_start : train_iloc_end],
                idx[last_window_iloc_start : test_iloc_start] if window_size is not None else [],
                idx[test_iloc_start : test_iloc_end],
                idx[test_iloc_start + self.gap : test_iloc_end]
            ]
            folds.append(partitions)
            i += 1

        if not self.allow_incomplete_fold and len(folds[-1][3]) < self.steps:
            folds = folds[:-1]
            last_fold_excluded = True

        # Replace partitions inside folds with length 0 with `None`
        folds = [
            [partition if len(partition) > 0 else None for partition in fold] 
            for fold in folds
        ]

        # Create a flag to know whether to train the forecaster
        if self.refit == 0:
            self.refit = False
            
        if isinstance(self.refit, bool):
            fit_forecaster = [self.refit] * len(folds)
            fit_forecaster[0] = True
        else:
            fit_forecaster = [False] * len(folds)
            for i in range(0, len(fit_forecaster), self.refit): 
                fit_forecaster[i] = True
        
        for i in range(len(folds)): 
            folds[i].append(fit_forecaster[i])
            if fit_forecaster[i] is False:
                folds[i][0] = folds[i - 1][0]

        index_to_skip = []
        if self.skip_folds is not None:
            if isinstance(self.skip_folds, (int, np.integer)) and self.skip_folds > 0:
                index_to_keep = np.arange(0, len(folds), self.skip_folds)
                index_to_skip = np.setdiff1d(np.arange(0, len(folds)), index_to_keep, assume_unique=True)
                index_to_skip = [int(x) for x in index_to_skip] # Required since numpy 2.0
            if isinstance(self.skip_folds, list):
                index_to_skip = [i for i in self.skip_folds if i < len(folds)]        
        
        if self.verbose:
            self._print_info(
                index = index,
                folds = folds,
                externally_fitted = externally_fitted,
                last_fold_excluded = last_fold_excluded,
                index_to_skip = index_to_skip
            )

        folds = [fold for i, fold in enumerate(folds) if i not in index_to_skip]
        if not self.return_all_indexes:
            # +1 to prevent iloc pandas from deleting the last observation
            folds = [
                [[fold[0][0], fold[0][-1] + 1], 
                [fold[1][0], fold[1][-1] + 1] if window_size is not None else [],
                [fold[2][0], fold[2][-1] + 1],
                [fold[3][0], fold[3][-1] + 1],
                fold[4]] 
                for fold in folds
            ]

        if as_pandas:
            if self.window_size is None:
                for fold in folds:
                    fold[1] = [None, None]

            if not self.return_all_indexes:
                folds = pd.DataFrame(
                    data = [list(itertools.chain(*fold[:-1])) + [fold[-1]] for fold in folds],
                    columns = [
                        'train_start',
                        'train_end',
                        'last_window_start',
                        'last_window_end',
                        'test_start',
                        'test_end',
                        'test_start_with_gap',
                        'test_end_with_gap',
                        'fit_forecaster'
                    ],
                )
            else:
                folds = pd.DataFrame(
                    data = folds,
                    columns = [
                        'train_index',
                        'last_window_index',
                        'test_index',
                        'test_index_with_gap',
                        'fit_forecaster'
                    ],
                )
            folds.insert(0, 'fold', range(len(folds)))
            

        return folds
    
    
    def _extract_index(
        self,
        X: Union[pd.Series, pd.DataFrame, pd.Index, dict]
    ) -> pd.Index:
        """
        Extracts and returns the index from the input data X.

        Parameters
        ----------
        X : pandas Series, DataFrame, Index, or dictionary
            Time series data or index to split.

        Returns
        -------
        pd.Index
            Index extracted from the input data.
        """

        if isinstance(X, (pd.Series, pd.DataFrame)):
            index = X.index
        elif isinstance(X, dict):
            freqs = [s.index.freq for s in X.values() if s.index.freq is not None]
            if not freqs:
                raise ValueError("At least one series must have a frequency.")
            if not all(f == freqs[0] for f in freqs):
                raise ValueError(
                    "All series with frequency must have the same frequency."
                )
            min_index = min([v.index[0] for v in X.values()])
            max_index = max([v.index[-1] for v in X.values()])
            index = pd.date_range(start=min_index, end=max_index, freq=freqs[0])
        else:
            index = X
            
        return index
    

    def set_params(self, params: dict) -> None:
        """
        Set the parameters of the TimeSeriesFold object. Before overwriting the
        current parameters, the input parameters are validated to ensure correctness.

        Parameters
        ----------
        params : dict
            Dictionary with the parameters to set.
        """

        if not isinstance(params, dict):
            raise ValueError(
                f"`params` must be a dictionary. Got {type(params)}."
            )
        
        current_params = deepcopy(vars(self))
        unknown_params = set(params.keys()) - set(current_params.keys())
        if unknown_params:
            warnings.warn(
                f"Unknown parameters: {unknown_params}. They have been ignored.",
                IgnoredArgumentWarning,
            )
        updated_params = {**current_params, **params}
        self._validate_params(**updated_params)
        for key, value in updated_params.items():
            setattr(self, key, value)
    

    def _print_info(
            self,
            index: pd.Index,
            folds: list,
            externally_fitted: bool,
            last_fold_excluded: bool,
            index_to_skip: list,
    ) -> None:
        """
        Print information about folds.
        """

        print("Information of folds")
        print("--------------------")
        if externally_fitted:
            print(
                f"An already trained forecaster is to be used. Window size: "
                f"{self.window_size}"
            )
        else:
            if self.differentiation is None:
                print(
                    f"Number of observations used for initial training: "
                    f"{self.initial_train_size}"
                )
            else:
                print(
                    f"Number of observations used for initial training: "
                    f"{self.initial_train_size - self.differentiation}"
                )
                print(f"    First {self.differentiation} observation/s in training sets "
                        f"are used for differentiation"
                )
        print(
            f"Number of observations used for backtesting: "
            f"{len(index) - self.initial_train_size}"
        )
        print(f"    Number of folds: {len(folds)}")
        print(
            f"    Number skipped folds: "
            f"{len(index_to_skip)} {index_to_skip if index_to_skip else ''}"
        )
        print(f"    Number of steps per fold: {self.steps}")
        print(
            f"    Number of steps to exclude from the end of each train set "
            f"before test (gap): {self.gap}"
        )
        if last_fold_excluded:
            print("    Last fold has been excluded because it was incomplete.")
        if len(folds[-1][3]) < self.steps:
            print(f"    Last fold only includes {len(folds[-1][3])} observations.")
        print("")

        if self.differentiation is None:
            self.differentiation = 0
        
        for i, fold in enumerate(folds):
            is_fold_skipped   = i in index_to_skip
            has_training      = fold[-1] if i != 0 else True
            training_start    = (
                index[fold[0][0] + self.differentiation] if fold[0] is not None else None
            )
            training_end      = index[fold[0][-1]] if fold[0] is not None else None
            training_length   = (
                len(fold[0]) - self.differentiation if fold[0] is not None else 0
            )
            validation_start  = index[fold[3][0]]
            validation_end    = index[fold[3][-1]]
            validation_length = len(fold[3])

            print(f"Fold: {i}")
            if is_fold_skipped:
                print("    Fold skipped")
            elif not externally_fitted and has_training:
                print(
                    f"    Training:   {training_start} -- {training_end}  "
                    f"(n={training_length})"
                )
                print(
                    f"    Validation: {validation_start} -- {validation_end}  "
                    f"(n={validation_length})"
                )
            else:
                print("    Training:   No training in this fold")
                print(
                    f"    Validation: {validation_start} -- {validation_end}  "
                    f"(n={validation_length})"
                )

        print("")


class OneStepAheadFold():
    """
    Class to split time series data into train and test folds for one-step-ahead
    forecasting.

    Parameters
    ----------
    initial_train_size : int, default=None
        Number of observations used for initial training.
    window_size : int, default=None
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int, default=None
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    return_all_indexes : bool, default=False
        Whether to return all indexes or only the start and end indexes of each fold.
    verbose : bool, default=True
        Whether to print information about generated folds.

    Attributes
    ----------
    initial_train_size : int, default=None
        Number of observations used for initial training.
    window_size : int
        Number of observations needed to generate the autoregressive predictors.
    differentiation : int, default=None 
        Number of observations to use for differentiation. This is used to extend the
        `last_window` as many observations as the differentiation order.
    return_all_indexes : bool, default=False
        Whether to return all indexes or only the start and end indexes of each fold.
    steps : any
        This attribute is not used in this class. It is included for API consistency.
    fixed_train_size : any
        This attribute is not used in this class. It is included for API consistency.
    gap : any
        This attribute is not used in this class. It is included for API consistency.
    skip_folds : any
        This attribute is not used in this class. It is included for API consistency.
    allow_incomplete_fold : any
        This attribute is not used in this class. It is included for API consistency.
    refit : any
        This attribute is not used in this class. It is included for API consistency.
    
    """

    def __init__(
            self,
            initial_train_size: Optional[int] = None,
            window_size: Optional[int] = None,
            differentiation: Optional[int] = None,
            return_all_indexes: bool = False,
            verbose: bool = True,
    ) -> None:

        self._validate_params(
            initial_train_size,
            window_size,
            differentiation,
            return_all_indexes,
        )

        self.initial_train_size = initial_train_size
        self.window_size = window_size
        self.differentiation = differentiation
        self.return_all_indexes = return_all_indexes
        self.verbose = verbose
        # Attributes not used, included for API consistency
        self.steps = 1 
        self.fixed_train_size = True
        self.gap = 0
        self.skip_folds = None
        self.allow_incomplete_fold = False
        self.refit = False


    def _validate_params(
        self,
        initial_train_size: Optional[int] = None,
        window_size: Optional[int] = None,
        differentiation: Optional[int] = None,
        return_all_indexes: bool = False,
        verbose: bool = True
    ) -> None:
                
        """
        Validate all input parameters to ensure correctness.
        """

        if (
            not isinstance(window_size, (int, np.integer, type(None)))
            or window_size is not None
            and window_size < 1
        ):
            raise ValueError(
                f"`window_size` must be an integer greater than 0. Got {window_size}."
            )
        if not isinstance(initial_train_size, (int, np.integer)) and initial_train_size is not None:
            raise ValueError(
                f"`initial_train_size` must be an integer or None. Got {initial_train_size}."
            )
    
        if not isinstance(return_all_indexes, bool):
            raise ValueError(
                f"`return_all_indexes` must be a boolean. Got {return_all_indexes}."
            )
        if differentiation is not None:
            if not isinstance(differentiation, (int, np.integer)) or differentiation < 0:
                raise ValueError(
                    f"differentiation must be None or an integer greater than or equal to 0. "
                    f"Got {differentiation}."
                )
        if not isinstance(verbose, bool):
            raise ValueError(
                f"`verbose` must be a boolean. Got {verbose}."
            )


    def __repr__(
        self
    ) -> str:
        """
        Information displayed when printed.
        """
            
        return (
            f"OneStepAheadFold(\n"
            f"    initial_train_size={self.initial_train_size},\n"
            f"    window_size={self.window_size},\n"
            f"    differentiation={self.differentiation},\n"
            f"    return_all_indexes={self.return_all_indexes},\n"
            f"    verbose={self.verbose}\n"
            f")"
        )
        

    def split(
            self,
            X: Union[pd.Series, pd.DataFrame, pd.Index, dict],
            as_pandas: bool = False,
            externally_fitted: any = None
    ) -> Union[list, pd.DataFrame]:
        """
        Split the time series data into train and test folds.

        Parameters
        ----------
        X : pandas Series, DataFrame, Index, or dictionary
            Time series data or index to split.
        as_pandas : bool, default=False
            If True, the folds are returned as a DataFrame. This is useful to visualize
            the folds in a more interpretable way.
        externally_fitted : any
            This argument is not used in this class. It is included for API consistency.
        
        Returns
        -------
        list, pd.DataFrame
            A list of lists containing the indices (position) for for each fold. Each list
            contains 2 lists the following information:

            - [train_start, train_end]: list with the start and end positions of the
            training set.
            - [test_start, test_end]: list with the start and end positions of the test
            set. These are the observations used to evaluate the forecaster.
        
            It is important to note that the returned values are the positions of the
            observations and not the actual values of the index, so they can be used to
            slice the data directly using iloc.

            If `as_pandas` is `True`, the folds are returned as a DataFrame with the
            following columns: 'fold', 'train_start', 'train_end', 'test_start', 'test_end'.

            Following the python convention, the start index is inclusive and the end
            index is exclusive. This means that the last index is not included in the
            slice.
        """

        index = self._extract_index(X)
        fold = [
            [0, self.initial_train_size],
            [self.initial_train_size, len(X)],
            True
        ]

        if self.verbose:
            self._print_info(
                index = index,
                fold = fold,
            )

        if self.return_all_indexes:
            fold = [
                [range(fold[0][0], fold[0][1])],
                [range(fold[1][0], fold[1][1])],
                fold[2]
            ]

        if as_pandas:
            if not self.return_all_indexes:
                fold = pd.DataFrame(
                    data = [list(itertools.chain(*fold[:-1])) + [fold[-1]]],
                    columns = [
                        'train_start',
                        'train_end',
                        'test_start',
                        'test_end',
                        'fit_forecaster'
                    ],
                )
            else:
                fold = pd.DataFrame(
                    data = [fold],
                    columns = [
                        'train_index',
                        'test_index',
                        'fit_forecaster'
                    ],
                )
            fold.insert(0, 'fold', range(len(fold)))

        return fold
    

    def _extract_index(
        self,
        X: Union[pd.Series, pd.DataFrame, pd.Index, dict]
    ) -> pd.Index:
        """
        Extracts and returns the index from the input data X.

        Parameters
        ----------
        X : pandas Series, DataFrame, Index, or dictionary
            Time series data or index to split.

        Returns
        -------
        pd.Index
            Index extracted from the input data.
        """

        if isinstance(X, (pd.Series, pd.DataFrame)):
            index = X.index
        elif isinstance(X, dict):
            freqs = [s.index.freq for s in X.values() if s.index.freq is not None]
            if not freqs:
                raise ValueError("At least one series must have a frequency.")
            if not all(f == freqs[0] for f in freqs):
                raise ValueError(
                    "All series with frequency must have the same frequency."
                )
            min_index = min([v.index[0] for v in X.values()])
            max_index = max([v.index[-1] for v in X.values()])
            index = pd.date_range(start=min_index, end=max_index, freq=freqs[0])
        else:
            index = X
            
        return index
    

    def set_params(self, params: dict) -> None:
        """
        Set the parameters of the OneStepAheadFold object.

        Parameters
        ----------
        params : dict
            Dictionary with the parameters to set.
        """

        if not isinstance(params, dict):
            raise ValueError(
                f"`params` must be a dictionary. Got {type(params)}."
            )
        
        current_params = deepcopy(vars(self))
        unknown_params = set(params.keys()) - set(current_params.keys())
        if unknown_params:
            warnings.warn(
                f"Unknown parameters: {unknown_params}. They have been ignored.",
                IgnoredArgumentWarning,
            )
        updated_params = {**current_params, **params}
        self._validate_params(**updated_params)
        for key, value in updated_params.items():
            setattr(self, key, value)


    def _print_info(
            self,
            index: pd.Index,
            fold: list,
    ) -> None:
        """
        Print information about folds.
        """

        print("Information of folds")
        print("--------------------")
        print(
            f"Number of observations used for initial training: "
            f"{self.initial_train_size}"
        )
        print(
            f"Number of observations in test: "
            f"{len(index) - self.initial_train_size}"
        )
 
        if self.differentiation is None:
            self.differentiation = 0
        
        training_start = index[fold[0][0] + self.differentiation]
        training_end = index[fold[0][-1]]
        training_length = self.initial_train_size
        test_start  = index[fold[1][0]]
        test_end    = index[fold[1][-1]-1]
        test_length = len(index) - self.initial_train_size

        print(
            f"Training : {training_start} -- {training_end} (n={training_length})"
        )
        print(
            f"Test     : {test_start} -- {test_end} (n={test_length})"
        )
        print("")



def _backtesting_forecaster(
    forecaster: object,
    y: pd.Series,
    metric: Union[str, Callable, list],
    cv: TimeSeriesFold,
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    interval: Optional[list] = None,
    n_boot: int = 250,
    random_state: int = 123,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = False,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = False,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of forecaster model following the folds generated by the TimeSeriesFold
    class and using the metric(s) provided.

    If `forecaster` is already trained and `initial_train_size` is set to `None` in the
    TimeSeriesFold class, no initial train will be done and all data will be used
    to evaluate the model. However, the first `len(forecaster.last_window)` observations
    are needed to create the initial predictors, so no predictions are calculated for
    them.
    
    A copy of the original forecaster is created so that it is not modified during the
    process.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
        **New in version 0.14.0**
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`. If `None`, no
        intervals are estimated.
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.
    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.
    use_in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction 
        error to create prediction intervals. If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    use_binned_residuals : bool, default `False`
        If `True`, residuals used in each bootstrapping iteration are selected
        conditioning on the predicted values. If `False`, residuals are selected
        randomly without conditioning on the predicted values.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.

        - column pred: predictions.
        - column lower_bound: lower bound of the interval.
        - column upper_bound: upper bound of the interval.
    
    """

    forecaster = deepcopy(forecaster)
    cv = deepcopy(cv)

    cv.set_params({
        'window_size': forecaster.window_size,
        'differentiation': forecaster.differentiation,
        'return_all_indexes': False,
        'verbose': verbose
    })

    if n_jobs == 'auto':
        n_jobs = select_n_jobs_backtesting(
                     forecaster = forecaster,
                     refit      = cv.refit
                 )
    elif not isinstance(cv.refit, bool) and cv.refit != 1 and n_jobs != 1:
        warnings.warn(
            ("If `refit` is an integer other than 1 (intermittent refit). `n_jobs` "
             "is set to 1 to avoid unexpected results during parallelization."),
             IgnoredArgumentWarning
        )
        n_jobs = 1
    else:
        n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    if not isinstance(metric, list):
        metrics = [
            _get_metric(metric=metric)
            if isinstance(metric, str)
            else add_y_train_argument(metric)
        ]
    else:
        metrics = [
            _get_metric(metric=m)
            if isinstance(m, str)
            else add_y_train_argument(m) 
            for m in metric
        ]

    store_in_sample_residuals = False if interval is None else True

    if cv.initial_train_size is not None:
        # First model training, this is done to allow parallelization when `refit`
        # is `False`. The initial Forecaster fit is outside the auxiliary function.
        exog_train = exog.iloc[:cv.initial_train_size, ] if exog is not None else None
        forecaster.fit(
            y                         = y.iloc[:cv.initial_train_size, ],
            exog                      = exog_train,
            store_in_sample_residuals = store_in_sample_residuals
        )
        externally_fitted = False
    else:
        # Although not used for training, first observations are needed to create
        # the initial predictors
        cv.set_params({'initial_train_size': forecaster.window_size})
        externally_fitted = True

    folds = cv.split(
                X                  = y,
                externally_fitted  = externally_fitted,
            )
    # This is done to allow parallelization when `refit` is `False`. The initial 
    # Forecaster fit is outside the auxiliary function.
    folds[0][4] = False

    if cv.refit:
        n_of_fits = int(len(folds) / cv.refit)
        if type(forecaster).__name__ != 'ForecasterAutoregDirect' and n_of_fits > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits} times. This can take substantial"
                 f" amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )
        elif type(forecaster).__name__ == 'ForecasterAutoregDirect' and n_of_fits * forecaster.steps > 50:
            warnings.warn(
                (f"The forecaster will be fit {n_of_fits * forecaster.steps} times "
                 f"({n_of_fits} folds * {forecaster.steps} regressors). This can take "
                 f"substantial amounts of time. If not feasible, try with `refit = False`.\n"),
                LongTrainingWarning
            )

    if show_progress:
        folds = tqdm(folds)

    def _fit_predict_forecaster(y, exog, forecaster, interval, fold, gap):
        """
        Fit the forecaster and predict `steps` ahead. This is an auxiliary 
        function used to parallelize the backtesting_forecaster function.
        """

        train_iloc_start       = fold[0][0]
        train_iloc_end         = fold[0][1]
        last_window_iloc_start = fold[1][0]
        last_window_iloc_end   = fold[1][1]
        test_iloc_start        = fold[2][0]
        test_iloc_end          = fold[2][1]

        if fold[4] is False:
            # When the model is not fitted, last_window must be updated to include
            # the data needed to make predictions.
            last_window_y = y.iloc[last_window_iloc_start:last_window_iloc_end]
        else:
            # The model is fitted before making predictions. If `fixed_train_size`
            # the train size doesn't increase but moves by `steps` in each iteration.
            # If `False` the train size increases by `steps` in each iteration.
            y_train = y.iloc[train_iloc_start:train_iloc_end, ]
            exog_train = (
                exog.iloc[train_iloc_start:train_iloc_end,] if exog is not None else None
            )
            last_window_y = None
            forecaster.fit(
                y                         = y_train, 
                exog                      = exog_train, 
                store_in_sample_residuals = store_in_sample_residuals
            )

        next_window_exog = exog.iloc[test_iloc_start:test_iloc_end, ] if exog is not None else None

        steps = len(range(test_iloc_start, test_iloc_end))
        if type(forecaster).__name__ == 'ForecasterAutoregDirect' and gap > 0:
            # Select only the steps that need to be predicted if gap > 0
            test_no_gap_iloc_start = fold[3][0]
            test_no_gap_iloc_end   = fold[3][1]
            steps = list(
                np.arange(len(range(test_no_gap_iloc_start, test_no_gap_iloc_end)))
                + gap
                + 1
            )

        if interval is None:
            pred = forecaster.predict(
                       steps       = steps,
                       last_window = last_window_y,
                       exog        = next_window_exog
                   )
        else:
            pred = forecaster.predict_interval(
                       steps                   = steps,
                       last_window             = last_window_y,
                       exog                    = next_window_exog,
                       interval                = interval,
                       n_boot                  = n_boot,
                       random_state            = random_state,
                       use_in_sample_residuals = use_in_sample_residuals,
                       use_binned_residuals    = use_binned_residuals,
                   )

        if type(forecaster).__name__ != 'ForecasterAutoregDirect' and gap > 0:
            pred = pred.iloc[gap:, ]

        return pred

    backtest_predictions = (
        Parallel(n_jobs=n_jobs)
        (delayed(_fit_predict_forecaster)
        (y=y, exog=exog, forecaster=forecaster, interval=interval, fold=fold, gap=cv.gap)
         for fold in folds)
    )

    backtest_predictions = pd.concat(backtest_predictions)
    if isinstance(backtest_predictions, pd.Series):
        backtest_predictions = pd.DataFrame(backtest_predictions)

    train_indexes = []
    for i, fold in enumerate(folds):
        fit_fold = fold[-1]
        if i == 0 or fit_fold:
            train_iloc_start = fold[0][0] + cv.window_size  # Exclude observations used to create predictors
            train_iloc_end = fold[0][1]
            train_indexes.append(np.arange(train_iloc_start, train_iloc_end))
    
    train_indexes = np.unique(np.concatenate(train_indexes))
    y_train = y.iloc[train_indexes]

    metric_values = [
        m(
            y_true = y.loc[backtest_predictions.index],
            y_pred = backtest_predictions['pred'],
            y_train = y_train
        ) 
        for m in metrics
    ]

    metric_values = pd.DataFrame(
        data    = [metric_values],
        columns = [m.__name__ for m in metrics]
    )
    
    return metric_values, backtest_predictions


def backtesting_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    interval: Optional[list] = None,
    n_boot: int = 250,
    random_state: int = 123,
    use_in_sample_residuals: bool = True,
    use_binned_residuals: bool = False,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = False,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting of forecaster model following the folds generated by the TimeSeriesFold
    class and using the metric(s) provided.

    If `forecaster` is already trained and `initial_train_size` is set to `None` in the
    TimeSeriesFold class, no initial train will be done and all data will be used
    to evaluate the model. However, the first `len(forecaster.last_window)` observations
    are needed to create the initial predictors, so no predictions are calculated for
    them.
    
    A copy of the original forecaster is created so that it is not modified during the
    process.

    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
        **New in version 0.14.0**
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    interval : list, default `None`
        Confidence of the prediction interval estimated. Sequence of percentiles
        to compute, which must be between 0 and 100 inclusive. For example, 
        interval of 95% should be as `interval = [2.5, 97.5]`. If `None`, no
        intervals are estimated.
    n_boot : int, default `500`
        Number of bootstrapping iterations used to estimate prediction
        intervals.
    random_state : int, default `123`
        Sets a seed to the random generator, so that boot intervals are always 
        deterministic.
    use_in_sample_residuals : bool, default `True`
        If `True`, residuals from the training data are used as proxy of prediction 
        error to create prediction intervals. If `False`, out_sample_residuals 
        are used if they are already stored inside the forecaster.
    use_binned_residuals : bool, default `False`
        If `True`, residuals used in each bootstrapping iteration are selected
        conditioning on the predicted values. If `False`, residuals are selected
        randomly without conditioning on the predicted values.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `False`
        Print number of folds and index of training and validation sets used 
        for backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.

    Returns
    -------
    metric_values : pandas DataFrame
        Value(s) of the metric(s).
    backtest_predictions : pandas DataFrame
        Value of predictions and their estimated interval if `interval` is not `None`.

        - column pred: predictions.
        - column lower_bound: lower bound of the interval.
        - column upper_bound: upper bound of the interval.
    
    """

    forecaters_allowed = [
        'ForecasterAutoreg', 
        'ForecasterAutoregCustom', 
        'ForecasterAutoregDirect',
        'ForecasterEquivalentDate'
    ]
    
    if type(forecaster).__name__ not in forecaters_allowed:
        raise TypeError(
            (f"`forecaster` must be of type {forecaters_allowed}, for all other types of "
             f" forecasters use the functions available in the other `model_selection` "
             f"modules.")
        )
    
    check_backtesting_input(
        forecaster              = forecaster,
        cv                      = cv,
        y                       = y,
        metric                  = metric,
        interval                = interval,
        n_boot                  = n_boot,
        random_state            = random_state,
        use_in_sample_residuals = use_in_sample_residuals,
        use_binned_residuals    = use_binned_residuals,
        n_jobs                  = n_jobs,
        verbose                 = verbose,
        show_progress           = show_progress
    )
    
    if type(forecaster).__name__ == 'ForecasterAutoregDirect' and \
       forecaster.steps < cv.steps + cv.gap:
        raise ValueError(
            (f"When using a ForecasterAutoregDirect, the combination of steps "
             f"+ gap ({cv.steps + cv.gap}) cannot be greater than the `steps` parameter "
             f"declared when the forecaster is initialized ({forecaster.steps}).")
        )
    
    metric_values, backtest_predictions = _backtesting_forecaster(
        forecaster              = forecaster,
        y                       = y,
        cv                      = cv,
        metric                  = metric,
        exog                    = exog,
        interval                = interval,
        n_boot                  = n_boot,
        random_state            = random_state,
        use_in_sample_residuals = use_in_sample_residuals,
        use_binned_residuals    = use_binned_residuals,
        n_jobs                  = n_jobs,
        verbose                 = verbose,
        show_progress           = show_progress
    )

    return metric_values, backtest_predictions


def grid_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    param_grid: dict,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Exhaustive search over specified parameter values for a Forecaster object.
    Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.14.0**
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try. Ignored 
        if the forecaster is an instance of `ForecasterAutoregCustom` or 
        `ForecasterAutoregMultiSeriesCustom`.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    
    """
    if type(cv).__name__ not in ['TimeSeriesFold', 'OneStepAheadFold']:
        raise ValueError(
            f"`cv` must be an instance of TimeSeriesFold or OneStepAheadFold. "
            f"Got {type(cv)}."
        )

    param_grid = list(ParameterGrid(param_grid))

    results = _evaluate_grid_hyperparameters(
        forecaster            = forecaster,
        y                     = y,
        cv                    = cv,
        param_grid            = param_grid,
        metric                = metric,
        exog                  = exog,
        lags_grid             = lags_grid,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress,
        output_file           = output_file
    )

    return results


def random_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    param_distributions: dict,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    n_iter: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Random search over specified parameter values or distributions for a Forecaster 
    object. Validation is done using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.14.0**
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and 
        distributions or lists of parameters to try.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try. Ignored 
        if the forecaster is an instance of `ForecasterAutoregCustom` or 
        `ForecasterAutoregMultiSeriesCustom`.
    n_iter : int, default `10`
        Number of parameter settings that are sampled per lags configuration. 
        n_iter trades off runtime vs quality of the solution.
    random_state : int, default `123`
        Sets a seed to the random sampling for reproducible output.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    
    """
    if type(cv).__name__ not in ['TimeSeriesFold', 'OneStepAheadFold']:
        raise ValueError(
            f"`cv` must be an instance of TimeSeriesFold or OneStepAheadFold. "
            f"Got {type(cv)}."
        )

    param_grid = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

    results = _evaluate_grid_hyperparameters(
        forecaster            = forecaster,
        y                     = y,
        cv                    = cv,
        param_grid            = param_grid,
        metric                = metric,
        exog                  = exog,
        lags_grid             = lags_grid,
        return_best           = return_best,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress,
        output_file           = output_file
    )

    return results


def _calculate_metrics_one_step_ahead(
    forecaster: object,
    y: pd.Series,
    metrics: list,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, dict],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, dict]
) -> list:
    """
    Calculate metrics when predictions are one-step-ahead. When forecaster is
    of type ForecasterAutoregDirect only the regressor for step 1 is used.

    Parameters
    ----------
    forecaster : object
        Forecaster model.
    y : pandas Series
        Time series data used to train and test the model.
    metrics : list
        List of metrics.
    X_train : pandas DataFrame
        Predictor values used to train the model.
    y_train : pandas Series
        Target values related to each row of `X_train`.
    X_test : pandas DataFrame
        Predictor values used to test the model.
    y_test : pandas Series
        Target values related to each row of `X_test`.

    Returns
    -------
    metric_values : list
        List with metric values.
    
    """

    if type(forecaster).__name__ == 'ForecasterAutoregDirect':

        step = 1  # Only model for step 1 is optimized.
        X_train, y_train = forecaster.filter_train_X_y_for_step(
                               step    = step,
                               X_train = X_train,
                               y_train = y_train
                           )
        X_test, y_test = forecaster.filter_train_X_y_for_step(
                             step    = step,  
                             X_train = X_test,
                             y_train = y_test
                         )
        forecaster.regressors_[step].fit(X_train, y_train)
        pred = forecaster.regressors_[step].predict(X_test)

    else:
        forecaster.regressor.fit(X_train, y_train)
        pred = forecaster.regressor.predict(X_test)

    pred = pred.ravel()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    if forecaster.differentiation is not None:
        differentiator = copy(forecaster.differentiator)
        differentiator.initial_values = (
            [y.iloc[forecaster.window_size - forecaster.differentiation]]
        )
        pred = differentiator.inverse_transform_next_window(pred)
        y_test = differentiator.inverse_transform_next_window(y_test)
        y_train = differentiator.inverse_transform(y_train)

    if forecaster.transformer_y is not None:
        pred = forecaster.transformer_y.inverse_transform(pred.reshape(-1, 1))
        y_test = forecaster.transformer_y.inverse_transform(y_test.reshape(-1, 1))
        y_train = forecaster.transformer_y.inverse_transform(y_train.reshape(-1, 1))

    metric_values = []
    for m in metrics:
        metric_values.append(
            m(y_true=y_test.ravel(), y_pred=pred.ravel(), y_train=y_train.ravel())
        )

    return metric_values


def _evaluate_grid_hyperparameters(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    param_grid: dict,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    lags_grid: Optional[Union[list, dict]] = None,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate parameter values for a Forecaster object using time series backtesting.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.14.0**
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i]. 
    lags_grid : list, dict, default `None`
        Lists of lags to try, containing int, lists, numpy ndarray, or range 
        objects. If `dict`, the keys are used as labels in the `results` 
        DataFrame, and the values are used as the lists of lags to try. Ignored 
        if the forecaster is an instance of `ForecasterAutoregCustom` or 
        `ForecasterAutoregMultiSeriesCustom`.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column lags_label: descriptive label or alias for the lags.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.

    """

    if type(cv).__name__ not in ['TimeSeriesFold', 'OneStepAheadFold']:
        raise ValueError(
            f"`cv` must be an instance of TimeSeriesFold or OneStepAheadFold. "
            f"Got {type(cv)}."
        )
    
    if type(cv).__name__ == 'OneStepAheadFold':
        warnings.warn(
            ("One-step-ahead predictions are used for faster model comparison, but they "
             "may not fully represent multi-step prediction performance. It is recommended "
             "to backtest the final model for a more accurate multi-step performance "
             "estimate.")
        )

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            (f"`exog` must have same number of samples as `y`. "
             f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
        )

    lags_grid, lags_label = initialize_lags_grid(forecaster, lags_grid)
   
    if not isinstance(metric, list):
        metric = [metric] 
    metric = [
            _get_metric(metric=m)
            if isinstance(m, str)
            else add_y_train_argument(m) 
            for m in metric
        ]
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] 
                   for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )

    if verbose:
        print(f"Number of models compared: {len(param_grid) * len(lags_grid)}.")

    if show_progress:
        lags_grid_tqdm = tqdm(lags_grid.items(), desc='lags grid', position=0)  # ncols=90
        param_grid = tqdm(param_grid, desc='params grid', position=1, leave=False)
    else:
        lags_grid_tqdm = lags_grid.items()
    
    if output_file is not None and os.path.isfile(output_file):
        os.remove(output_file)

    lags_list = []
    lags_label_list = []
    params_list = []
    for lags_k, lags_v in lags_grid_tqdm:
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(lags_v)
            lags_v = forecaster.lags.copy()
            if lags_label == 'values':
                lags_k = lags_v

        if type(cv).__name__ == 'OneStepAheadFold':

            (
                X_train,
                y_train,
                X_test,
                y_test
            ) = forecaster._train_test_split_one_step_ahead(
                y=y, initial_train_size=cv.initial_train_size, exog=exog
            )

        for params in param_grid:

            forecaster.set_params(params)

            if type(cv).__name__ == 'TimeSeriesFold':

                metric_values = backtesting_forecaster(
                                    forecaster            = forecaster,
                                    y                     = y,
                                    cv                    = cv,
                                    metric                = metric,
                                    exog                  = exog,
                                    interval              = None,
                                    n_jobs                = n_jobs,
                                    verbose               = verbose,
                                    show_progress         = False
                                )[0]
                metric_values = metric_values.iloc[0, :].to_list()

            else:

                metric_values = _calculate_metrics_one_step_ahead(
                                    forecaster = forecaster,
                                    y          = y,
                                    metrics    = metric,
                                    X_train    = X_train,
                                    y_train    = y_train,
                                    X_test     = X_test,
                                    y_test     = y_test
                                )

            warnings.filterwarnings(
                'ignore',
                category = RuntimeWarning, 
                message  = "The forecaster will be fit.*"
            )
            
            lags_list.append(lags_v)
            lags_label_list.append(lags_k)
            params_list.append(params)
            for m, m_value in zip(metric, metric_values):
                m_name = m if isinstance(m, str) else m.__name__
                metric_dict[m_name].append(m_value)
        
            if output_file is not None:
                header = ['lags', 'lags_label', 'params', 
                          *metric_dict.keys(), *params.keys()]
                row = [lags_v, lags_k, params, 
                       *metric_values, *params.values()]
                if not os.path.isfile(output_file):
                    with open(output_file, 'w', newline='') as f:
                        f.write('\t'.join(header) + '\n')
                        f.write('\t'.join([str(r) for r in row]) + '\n')
                else:
                    with open(output_file, 'a', newline='') as f:
                        f.write('\t'.join([str(r) for r in row]) + '\n')
    
    results = pd.DataFrame({
                  'lags': lags_list,
                  'lags_label': lags_label_list,
                  'params': params_list,
                  **metric_dict
              })
    
    results = (
        results
        .sort_values(by=list(metric_dict.keys())[0], ascending=True)
        .reset_index(drop=True)
    )
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        
        best_lags = results.loc[0, 'lags']
        best_params = results.loc[0, 'params']
        best_metric = results.loc[0, list(metric_dict.keys())[0]]
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  {'Backtesting' if type(cv).__name__ == 'TimeSeriesFold' else 'One-step-ahead'} "
            f"metric: {best_metric}"
        )
            
    return results


def bayesian_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    search_space: Callable,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None,
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using time series backtesting and 
    optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series.
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.14.0**
    search_space : Callable (optuna)
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.
    random_state : int, default `123`
        Sets a seed to the sampling for reproducible output. When a new sampler 
        is passed in `kwargs_create_study`, the seed must be set within the 
        sampler. For example `{'sampler': TPESampler(seed=145)}`.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**
    kwargs_create_study : dict, default `{}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study().
        If default, the direction is set to 'minimize' and a TPESampler(seed=123) 
        sampler is used during optimization.
    kwargs_study_optimize : dict, default `{}`
        Other keyword arguments (key, value mappings) to pass to study.optimize().

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    best_trial : optuna object
        The best optimization result returned as a FrozenTrial optuna object.
    
    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            (f"`exog` must have same number of samples as `y`. "
             f"length `exog`: ({len(exog)}), length `y`: ({len(y)})")
        )
    
    if type(cv).__name__ not in ['TimeSeriesFold', 'OneStepAheadFold']:
        raise ValueError(
            f"`cv` must be an instance of TimeSeriesFold or OneStepAheadFold. "
            f"Got {type(cv)}."
        )
            
    results, best_trial = _bayesian_search_optuna(
                                forecaster            = forecaster,
                                y                     = y,
                                cv                    = cv,
                                exog                  = exog,
                                search_space          = search_space,
                                metric                = metric,
                                n_trials              = n_trials,
                                random_state          = random_state,
                                return_best           = return_best,
                                n_jobs                = n_jobs,
                                verbose               = verbose,
                                show_progress         = show_progress,
                                output_file           = output_file,
                                kwargs_create_study   = kwargs_create_study,
                                kwargs_study_optimize = kwargs_study_optimize
                          )

    return results, best_trial


def _bayesian_search_optuna(
    forecaster: object,
    y: pd.Series,
    cv: Union[TimeSeriesFold, OneStepAheadFold],
    search_space: Callable,
    metric: Union[str, Callable, list],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    n_trials: int = 10,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: Union[int, str] = 'auto',
    verbose: bool = True,
    show_progress: bool = True,
    output_file: Optional[str] = None,
    kwargs_create_study: dict = {},
    kwargs_study_optimize: dict = {}
) -> Tuple[pd.DataFrame, object]:
    """
    Bayesian optimization for a Forecaster object using time series backtesting 
    and optuna library.
    
    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom, ForecasterAutoregDirect
        Forecaster model.
    y : pandas Series
        Training time series. 
    cv : TimeSeriesFold, OneStepAheadFold
        TimeSeriesFold or OneStepAheadFold object with the information needed to split
        the data into folds.
        **New in version 0.
    search_space : Callable
        Function with argument `trial` which returns a dictionary with parameters names 
        (`str`) as keys and Trial object from optuna (trial.suggest_float, 
        trial.suggest_int, trial.suggest_categorical) as values.
    metric : str, Callable, list
        Metric used to quantify the goodness of fit of the model.
        
        - If `string`: {'mean_squared_error', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_log_error',
        'mean_absolute_scaled_error', 'root_mean_squared_scaled_error'}
        - If `Callable`: Function with arguments `y_true`, `y_pred` and `y_train`
        (Optional) that returns a float.
        - If `list`: List containing multiple strings and/or Callables.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    n_trials : int, default `10`
        Number of parameter settings that are sampled in each lag configuration.
    random_state : int, default `123`
        Sets a seed to the sampling for reproducible output. When a new sampler 
        is passed in `kwargs_create_study`, the seed must be set within the 
        sampler. For example `{'sampler': TPESampler(seed=145)}`.
    return_best : bool, default `True`
        Refit the `forecaster` using the best found parameters on the whole data.
    n_jobs : int, 'auto', default `'auto'`
        The number of jobs to run in parallel. If `-1`, then the number of jobs is 
        set to the number of cores. If 'auto', `n_jobs` is set using the function
        skforecast.utils.select_n_jobs_backtesting.
    verbose : bool, default `True`
        Print number of folds used for cv or backtesting.
    show_progress : bool, default `True`
        Whether to show a progress bar.
    output_file : str, default `None`
        Specifies the filename or full path where the results should be saved. 
        The results will be saved in a tab-separated values (TSV) format. If 
        `None`, the results will not be saved to a file.
        **New in version 0.12.0**
    kwargs_create_study : dict, default `{}`
        Keyword arguments (key, value mappings) to pass to optuna.create_study().
        If default, the direction is set to 'minimize' and a TPESampler(seed=123) 
        sampler is used during optimization.
    kwargs_study_optimize : dict, default `{}`
        Other keyword arguments (key, value mappings) to pass to study.optimize().

    Returns
    -------
    results : pandas DataFrame
        Results for each combination of parameters.

        - column lags: lags configuration for each iteration.
        - column params: parameters configuration for each iteration.
        - column metric: metric value estimated for each iteration.
        - additional n columns with param = value.
    best_trial : optuna object
        The best optimization result returned as an optuna FrozenTrial object.

    """

    if type(cv).__name__ not in ['TimeSeriesFold', 'OneStepAheadFold']:
        raise ValueError(
            f"`cv` must be an instance of TimeSeriesFold or OneStepAheadFold. "
            f"Got {type(cv)}."
        )
    
    if type(cv).__name__ == 'OneStepAheadFold':
        warnings.warn(
            ("One-step-ahead predictions are used for faster model comparison, but they "
             "may not fully represent multi-step prediction performance. It is recommended "
             "to backtest the final model for a more accurate multi-step performance "
             "estimate.")
        )
    
    if not isinstance(metric, list):
        metric = [metric]
    metric = [
            _get_metric(metric=m)
            if isinstance(m, str)
            else add_y_train_argument(m) 
            for m in metric
        ]
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] 
                   for m in metric}
    
    if len(metric_dict) != len(metric):
        raise ValueError(
            "When `metric` is a `list`, each metric name must be unique."
        )
        
    # Objective function using backtesting_forecaster
    if type(cv).__name__ == 'TimeSeriesFold':

        def _objective(
            trial,
            search_space          = search_space,
            forecaster            = forecaster,
            y                     = y,
            cv                    = cv,
            exog                  = exog,
            metric                = metric,
            n_jobs                = n_jobs,
            verbose               = verbose,
        ) -> float:
            
            sample = search_space(trial)
            sample_params = {k: v for k, v in sample.items() if k != 'lags'}
            forecaster.set_params(sample_params)
            if type(forecaster).__name__ != 'ForecasterAutoregCustom':
                if "lags" in sample:
                    forecaster.set_lags(sample['lags'])
            
            metrics, _ = backtesting_forecaster(
                            forecaster            = forecaster,
                            y                     = y,
                            cv                    = cv,
                            exog                  = exog,
                            metric                = metric,
                            n_jobs                = n_jobs,
                            verbose               = verbose,
                            show_progress         = False
                        )
            metrics = metrics.iloc[0, :].to_list()
            
            # Store metrics in the variable `metric_values` defined outside _objective.
            nonlocal metric_values
            metric_values.append(metrics)

            return metrics[0]
        
    else:

        def _objective(
            trial,
            search_space          = search_space,
            forecaster            = forecaster,
            y                     = y,
            cv                    = cv,
            exog                  = exog,
            metric                = metric
        ) -> float:
            
            sample = search_space(trial)
            sample_params = {k: v for k, v in sample.items() if k != 'lags'}
            forecaster.set_params(sample_params)
            if type(forecaster).__name__ != 'ForecasterAutoregCustom':
                if "lags" in sample:
                    forecaster.set_lags(sample['lags'])

            (
                X_train,
                y_train,
                X_test,
                y_test
            ) = forecaster._train_test_split_one_step_ahead(
                y=y, initial_train_size=cv.initial_train_size, exog=exog
            )

            metrics = _calculate_metrics_one_step_ahead(
                            forecaster = forecaster,
                            y          = y,
                            metrics    = metric,
                            X_train    = X_train,
                            y_train    = y_train,
                            X_test     = X_test,
                            y_test     = y_test
                    )

            # Store all metrics in the variable `metric_values` defined outside _objective.
            nonlocal metric_values
            metric_values.append(metrics)

            return metrics[0]

    if show_progress:
        kwargs_study_optimize['show_progress_bar'] = True

    if output_file is not None:
        # Redirect optuna logging to file
        optuna.logging.disable_default_handler()
        logger = logging.getLogger('optuna')
        logger.setLevel(logging.INFO)
        for handler in logger.handlers.copy():
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
        handler = logging.FileHandler(output_file, mode="w")
        logger.addHandler(handler)
    else:
        logging.getLogger("optuna").setLevel(logging.WARNING)
        optuna.logging.disable_default_handler()

    # `metric_values` will be modified inside _objective function. 
    # It is a trick to extract multiple values from _objective since
    # only the optimized value can be returned.
    metric_values = []

    warnings.filterwarnings(
        "ignore",
        category = UserWarning,
        message  = "Choices for a categorical distribution should be*"
    )

    study = optuna.create_study(**kwargs_create_study)

    if 'sampler' not in kwargs_create_study.keys():
        study.sampler = TPESampler(seed=random_state)

    study.optimize(_objective, n_trials=n_trials, **kwargs_study_optimize)
    best_trial = study.best_trial

    if output_file is not None:
        handler.close()

    if search_space(best_trial).keys() != best_trial.params.keys():
        raise ValueError(
            (f"Some of the key values do not match the search_space key names.\n"
             f"  Search Space keys  : {list(search_space(best_trial).keys())}\n"
             f"  Trial objects keys : {list(best_trial.params.keys())}.")
        )
    warnings.filterwarnings('default')
    
    lags_list = []
    params_list = []
    for i, trial in enumerate(study.get_trials()):
        regressor_params = {k: v for k, v in trial.params.items() if k != 'lags'}
        lags = trial.params.get(
                   'lags',
                   forecaster.lags if hasattr(forecaster, 'lags') else None
               )
        params_list.append(regressor_params)
        lags_list.append(lags)
        for m, m_values in zip(metric, metric_values[i]):
            m_name = m if isinstance(m, str) else m.__name__
            metric_dict[m_name].append(m_values)
    
    if type(forecaster).__name__ != 'ForecasterAutoregCustom':
        lags_list = [
            initialize_lags(forecaster_name=type(forecaster).__name__, lags = lag)
            for lag in lags_list
        ]
    else:
        lags_list = [
            f"custom function: {forecaster.fun_predictors.__name__}"
            for _
            in lags_list
        ]

    results = pd.DataFrame({
                  'lags': lags_list,
                  'params': params_list,
                  **metric_dict
              })
    
    results = (
        results
        .sort_values(by=list(metric_dict.keys())[0], ascending=True)
        .reset_index(drop=True)
    )
    results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
    
    if return_best:
        best_lags = results.loc[0, 'lags']
        best_params = results.loc[0, 'params']
        best_metric = results.loc[0, list(metric_dict.keys())[0]]
        
        if type(forecaster).__name__ != 'ForecasterAutoregCustom':
            forecaster.set_lags(best_lags)
        forecaster.set_params(best_params)

        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        
        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  {'Backtesting' if type(cv).__name__ == 'TimeSeriesFold' else 'One-step-ahead'} "
            f"metric: {best_metric}"
        )
            
    return results, best_trial


def select_features(
    forecaster: object,
    selector: object,
    y: Union[pd.Series, pd.DataFrame],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    select_only: Optional[str] = None,
    force_inclusion: Optional[Union[list, str]] = None,
    subsample: Union[int, float] = 0.5,
    random_state: int = 123,
    verbose: bool = True
) -> Union[list, list]:
    """
    Feature selection using any of the sklearn.feature_selection module selectors 
    (such as `RFECV`, `SelectFromModel`, etc.). Two groups of features are
    evaluated: autoregressive features and exogenous features. By default, the 
    selection process is performed on both sets of features at the same time, 
    so that the most relevant autoregressive and exogenous features are selected. 
    However, using the `select_only` argument, the selection process can focus 
    only on the autoregressive or exogenous features without taking into account 
    the other features. Therefore, all other features will remain in the model. 
    It is also possible to force the inclusion of certain features in the final 
    list of selected features using the `force_inclusion` parameter.

    Parameters
    ----------
    forecaster : ForecasterAutoreg, ForecasterAutoregCustom
        Forecaster model.
    selector : object
        A feature selector from sklearn.feature_selection.
    y : pandas Series, pandas DataFrame
        Target time series to which the feature selection will be applied.
    exog : pandas Series, pandas DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
    select_only : str, default `None`
        Decide what type of features to include in the selection process. 
        
        - If `'autoreg'`, only autoregressive features (lags or custom 
        predictors) are evaluated by the selector. All exogenous features are 
        included in the output (`selected_exog`).
        - If `'exog'`, only exogenous features are evaluated without the presence
        of autoregressive features. All autoregressive features are included 
        in the output (`selected_autoreg`).
        - If `None`, all features are evaluated by the selector.
    force_inclusion : list, str, default `None`
        Features to force include in the final list of selected features.
        
        - If `list`, list of feature names to force include.
        - If `str`, regular expression to identify features to force include. 
        For example, if `force_inclusion="^sun_"`, all features that begin 
        with "sun_" will be included in the final list of selected features.
    subsample : int, float, default `0.5`
        Proportion of records to use for feature selection.
    random_state : int, default `123`
        Sets a seed for the random subsample so that the subsampling process 
        is always deterministic.
    verbose : bool, default `True`
        Print information about feature selection process.

    Returns
    -------
    selected_autoreg : list
        List of selected autoregressive features.
    selected_exog : list
        List of selected exogenous features.

    """

    valid_forecasters = [
        'ForecasterAutoreg',
        'ForecasterAutoregCustom'
    ]

    if type(forecaster).__name__ not in valid_forecasters:
        raise TypeError(
            f"`forecaster` must be one of the following classes: {valid_forecasters}."
        )
    
    if select_only not in ['autoreg', 'exog', None]:
        raise ValueError(
            "`select_only` must be one of the following values: 'autoreg', 'exog', None."
        )

    if subsample <= 0 or subsample > 1:
        raise ValueError(
            "`subsample` must be a number greater than 0 and less than or equal to 1."
        )
    
    forecaster = deepcopy(forecaster)
    forecaster.fitted = False
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)

    if hasattr(forecaster, 'lags'):
        autoreg_cols = [f"lag_{lag}" for lag in forecaster.lags]
    else:
        if forecaster.name_predictors is not None:
            autoreg_cols = forecaster.name_predictors
        else:
            autoreg_cols = [
                col
                for col in X_train.columns
                if re.match(r'^custom_predictor_\d+', col)
            ]
    exog_cols = [col for col in X_train.columns if col not in autoreg_cols]

    forced_autoreg = []
    forced_exog = []
    if force_inclusion is not None:
        if isinstance(force_inclusion, list):
            forced_autoreg = [col for col in force_inclusion if col in autoreg_cols]
            forced_exog = [col for col in force_inclusion if col in exog_cols]
        elif isinstance(force_inclusion, str):
            forced_autoreg = [col for col in autoreg_cols if re.match(force_inclusion, col)]
            forced_exog = [col for col in exog_cols if re.match(force_inclusion, col)]

    if select_only == 'autoreg':
        X_train = X_train.drop(columns=exog_cols)
    elif select_only == 'exog':
        X_train = X_train.drop(columns=autoreg_cols)

    if isinstance(subsample, float):
        subsample = int(len(X_train) * subsample)

    rng = np.random.default_rng(seed=random_state)
    sample = rng.choice(X_train.index, size=subsample, replace=False)
    X_train_sample = X_train.loc[sample, :]
    y_train_sample = y_train.loc[sample]
    selector.fit(X_train_sample, y_train_sample)
    selected_features = selector.get_feature_names_out()

    if select_only == 'exog':
        selected_autoreg = autoreg_cols
    else:
        selected_autoreg = [
            feature
            for feature in selected_features
            if feature in autoreg_cols
        ]

    if select_only == 'autoreg':
        selected_exog = exog_cols
    else:
        selected_exog = [
            feature
            for feature in selected_features
            if feature in exog_cols
        ]

    if force_inclusion is not None: 
        if select_only != 'autoreg':
            forced_exog_not_selected = set(forced_exog) - set(selected_features)
            selected_exog.extend(forced_exog_not_selected)
            selected_exog.sort(key=exog_cols.index)
        if select_only != 'exog':
            forced_autoreg_not_selected = set(forced_autoreg) - set(selected_features)
            selected_autoreg.extend(forced_autoreg_not_selected)
            selected_autoreg.sort(key=autoreg_cols.index)

    if len(selected_autoreg) == 0:
        warnings.warn(
            ("No autoregressive features have been selected. Since a Forecaster "
             "cannot be created without them, be sure to include at least one "
             "using the `force_inclusion` parameter.")
        )
    else:
        if hasattr(forecaster, 'lags'):
            selected_autoreg = [int(feature.replace('lag_', '')) 
                                for feature in selected_autoreg]

    if verbose:
        print(f"Recursive feature elimination ({selector.__class__.__name__})")
        print("--------------------------------" + "-" * len(selector.__class__.__name__))
        print(f"Total number of records available: {X_train.shape[0]}")
        print(f"Total number of records used for feature selection: {X_train_sample.shape[0]}")
        print(f"Number of features available: {X_train.shape[1]}") 
        print(f"    Autoreg (n={len(autoreg_cols)})")
        print(f"    Exog    (n={len(exog_cols)})")
        print(f"Number of features selected: {len(selected_features)}")
        print(f"    Autoreg (n={len(selected_autoreg)}) : {selected_autoreg}")
        print(f"    Exog    (n={len(selected_exog)}) : {selected_exog}")

    return selected_autoreg, selected_exog
