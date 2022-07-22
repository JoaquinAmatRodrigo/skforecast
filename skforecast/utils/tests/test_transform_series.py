# Unit test transform_series
# ==============================================================================
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skforecast.utils import transform_series


def test_transform_series_when_transformer_is_None():
    """
    Test the output of transform_series when transformer is None.
    """
    input = pd.Series([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49])
    expected = input
    transformer = None
    results =  transform_series(
                    series = input,
                    transformer = transformer,
                    fit = True,
                    inverse_transform = False
                )
    pd.testing.assert_series_equal(results, expected)


def test_transform_series_when_transformer_is_StandardScaler():
    """
    Test the output of transform_series when transformer is StandardScaler.
    """
    input = pd.Series([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                      name = 'y')
    expected = pd.Series([0.67596768, -0.47871021, -0.19805933,  1.67027365, -0.0537246,
                         -0.70323091, -1.39283021,  0.75615365,  1.17312067, -1.44896038],
                         name = 'y')
    transformer = StandardScaler()
    results =  transform_series(
                    series = input,
                    transformer = transformer,
                    fit = True,
                    inverse_transform = False
                )
    pd.testing.assert_series_equal(results, expected)

def test_transform_series_when_transformer_is_StandardScaler_and_inverse_transform_is_True():
    """
    Test the output of transform_series when transformer is StandardScaler.
    """
    input_1 = pd.Series([1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                      name = 'y')
    transformer = StandardScaler()
    transformer.fit(input_1.to_numpy().reshape(-1, 1))
    input_2 = transformer.transform(input_1.to_numpy().reshape(-1, 1))
    input_2 = pd.Series(input_2.flatten(), name= 'y')
    expected = input_1
    results =  transform_series(
                    series = input_2,
                    transformer = transformer,
                    fit = False,
                    inverse_transform = True
                )
    pd.testing.assert_series_equal(results, expected)