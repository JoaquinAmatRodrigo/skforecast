# Fixtures preprocessing
# ==============================================================================
import numpy as np
import pandas as pd


# Datetime Functions
year_cols_onehot = [f"year_{i}" for i in [2021, 2022, 2023]]
month_cols_onehot = [f"month_{i}" for i in range(1, 13)]
week_cols_onehot = [f"week_{i}" for i in range(1, 54)]
day_of_week_cols_onehot = [f"day_of_week_{i}" for i in range(0, 7)]
day_of_month_cols_onehot = [f"day_of_month_{i}" for i in range(1, 32)]
day_of_year_cols_onehot = [f"day_of_year_{i}" for i in range(1, 366)]
weekend_cols_onehot = ["weekend_0", "weekend_1"]
hour_cols_onehot = [f"hour_{i}" for i in range(0, 24)]
minute_cols_onehot = [f"minute_{i}" for i in range(0, 60)]
second_cols_onehot = [f"second_{i}" for i in range(0, 60)]

features_all_onehot = (
    year_cols_onehot
    + month_cols_onehot
    + week_cols_onehot
    + day_of_week_cols_onehot
    + day_of_month_cols_onehot
    + day_of_year_cols_onehot
    + weekend_cols_onehot
    + hour_cols_onehot
    + minute_cols_onehot
    + second_cols_onehot
)


# Series and exog to long
n_A = 10
n_B = 5
n_C = 15
series_id = np.repeat(["A", "B", "C"], [n_A, n_B, n_C])
index_A = pd.date_range("2020-01-01", periods=n_A, freq="D")
index_B = pd.date_range("2020-01-01", periods=n_B, freq="D")
index_C = pd.date_range("2020-01-01", periods=n_C, freq="D")
index = index_A.append(index_B).append(index_C)
values_A = np.arange(n_A)
values_B = np.arange(n_B)
values_C = np.arange(n_C)
values = np.concatenate([values_A, values_B, values_C])

series_long = pd.DataFrame(
    {
        "series_id": series_id,
        "datetime": index,
        "values": values,
    }
)

n_exog_A = 10
n_exog_B = 5
n_exog_C = 15
index_exog_A = pd.date_range("2020-01-01", periods=n_exog_A, freq="D")
index_exog_B = pd.date_range("2020-01-01", periods=n_exog_B, freq="D")
index_exog_C = pd.date_range("2020-01-01", periods=n_exog_C, freq="D")
exog_A = pd.DataFrame(
    {
        "series_id": "A",
        "datetime": index_exog_A,
        "exog_1": np.arange(n_exog_A),
    }
)
exog_B = pd.DataFrame(
    {
        "series_id": "B",
        "datetime": index_exog_B,
        "exog_1": np.arange(n_exog_B),
        "exog_2": "b",
    }
)
exog_C = pd.DataFrame(
    {
        "series_id": "C",
        "datetime": index_exog_C,
        "exog_1": np.arange(n_exog_C),
        "exog_3": 1,
    }
)
exog_long = pd.concat([exog_A, exog_B, exog_C], axis=0)
