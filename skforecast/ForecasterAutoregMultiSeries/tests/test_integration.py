# Integration test ForecasterAutoregMultiSeries
# ==============================================================================
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

# Fixtures
series_dict = joblib.load('./fixture_sample_multi_series.joblib')
exog_dict = joblib.load('./fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}


def test_integration_predict():
    """
    Integration test ForecasterAutoregMultiSeries predict method.
    """
    forecaster = ForecasterAutoregMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    predictions = forecaster.predict(
        steps=5, exog=exog_dict_test, suppress_warnings=True
    )
    expected = pd.DataFrame(
        data=np.array(
            [
                [1438.14154717, 2090.79352613, 2166.9832933, 7285.52781428],
                [1438.14154717, 2089.11038884, 2074.55994929, 7488.18398744],
                [1438.14154717, 2089.11038884, 2035.99448247, 7488.18398744],
                [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
                [1403.93625654, 2089.11038884, 2035.99448247, 7488.18398744],
            ]
        ),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=["id_1000", "id_1001", "id_1003", "id_1004"],
    )

    pd.testing.assert_frame_equal(predictions, expected)


def test_integration_predict_interval():
    """
    Integration test ForecasterAutoregMultiSeries predict_interval method.
    """
    forecaster = ForecasterAutoregMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    forecaster.fit(
        series=series_dict_train, exog=exog_dict_train, suppress_warnings=True
    )
    predictions = forecaster.predict_interval(
        steps=5, exog=exog_dict_test, suppress_warnings=True, n_boot=10, interval=[5, 95]
    )
    expected = pd.DataFrame(
        data=np.array([
            [1438.14154717,  984.89302371, 1866.71388628, 2090.79352613,
            902.35566771, 2812.9492079 , 2166.9832933 , 1478.23569823,
            2519.27378966, 7285.52781428, 4241.35466074, 8500.84534052],
            [1438.14154717, 1368.43204887, 1893.86659752, 2089.11038884,
            900.67253042, 2812.29873034, 2074.55994929, 1414.59564291,
            3006.64848384, 7488.18398744, 5727.33006697, 8687.00097656],
            [1438.14154717, 1051.65779678, 1863.99279153, 2089.11038884,
            933.86328276, 2912.07131797, 2035.99448247, 1455.42927055,
            3112.84247337, 7488.18398744, 4229.48737525, 8490.6467214 ],
            [1403.93625654, 1130.69596556, 1740.72910642, 2089.11038884,
            900.67253042, 2912.07131797, 2035.99448247, 1522.39553008,
            1943.48913692, 7488.18398744, 5983.32345864, 9175.53503076],
            [1403.93625654, 1075.26038028, 1851.5525839 , 2089.11038884,
            933.86328276, 2909.31686272, 2035.99448247, 1732.93979512,
            2508.99760524, 7488.18398744, 5073.26873656, 8705.7520813 ]]),
        index=pd.date_range(start="2016-08-01", periods=5, freq="D"),
        columns=[
            'id_1000',
            'id_1000_lower_bound',
            'id_1000_upper_bound',
            'id_1001',
            'id_1001_lower_bound',
            'id_1001_upper_bound',
            'id_1003',
            'id_1003_lower_bound',
            'id_1003_upper_bound',
            'id_1004',
            'id_1004_lower_bound',
            'id_1004_upper_bound'
    ]
    )

    pd.testing.assert_frame_equal(predictions, expected)