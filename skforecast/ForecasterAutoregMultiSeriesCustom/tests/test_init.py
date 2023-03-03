# Unit test __init__ ForecasterAutoregMultiSeriesCustom
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeriesCustom import ForecasterAutoregMultiSeriesCustom
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


