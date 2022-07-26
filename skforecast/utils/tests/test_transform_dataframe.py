# Unit test transform_dataframe
# ==============================================================================
import pytest
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.utils import transform_dataframe


def test_transform_dataframe_when_transformer_is_None():
    """
    Test the output of transform_dataframe when transformer is None.
    """
    df_input = pd.DataFrame({
                'A': [1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                'B': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 2.7, 59.6]
               })  
    expected = df_input
    transformer = None
    results =  transform_dataframe(
                    df = df_input,
                    transformer = transformer,
                    fit = True,
                    inverse_transform = False
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_when_transformer_is_StandardScaler():
    """
    Test the output of transform_dataframe when transformer is StandardScaler.
    """
    df_input = pd.DataFrame({
                'A': [1.16, -0.28, 0.07, 2.4, 0.25, -0.56, -1.42, 1.26, 1.78, -1.49],
                'B': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 2.7, 59.6]
               })
    expected = pd.DataFrame({
                'A': [0.67596768, -0.47871021, -0.19805933,  1.67027365, -0.0537246 ,
                     -0.70323091, -1.39283021,  0.75615365,  1.17312067, -1.44896038],
                'B': [-1.47939551, -0.79158852,  0.6694926 ,  0.54739669,  0.27878567,
                      -0.09971166,  1.76428598,  0.14448017, -1.67474897,  0.64100356]
               })
    transformer = StandardScaler()
    results =  transform_dataframe(
                    df = df_input,
                    transformer = transformer,
                    fit = True,
                    inverse_transform = False
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_when_transformer_is_OneHotEncoder():
    """
    Test the output of transform_dataframe when transformer is OneHotEncoder.
    """
    df_input = pd.DataFrame({
                'col_1': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
               })
    expected = pd.DataFrame({
                    'col_1_a': [1., 1., 1., 1., 0., 0., 0., 0.],
                    'col_1_b': [0., 0., 0., 0., 1., 1., 1., 1.],
                    'col_2_a': [1., 1., 1., 1., 0., 0., 0., 0.],
                    'col_2_b': [0., 0., 0., 0., 1., 1., 1., 1.]
               })
    transformer = OneHotEncoder()
    results =  transform_dataframe(
                    df = df_input,
                    transformer = transformer,
                    fit = True,
                    inverse_transform = False
               )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.skip(reason="work in progress")
def test_transform_dataframe_when_transformer_is_OneHotEncoder_and_inverse_transform():
    """
    Test the output of transform_dataframe when transformer is OneHotEncoder and
    inverse_transform is True.
    """
    df_input = pd.DataFrame({
                'col_1': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
            })
    transformer = OneHotEncoder()
    transformer.fit(df_input)
    df_input_2 = transformer.transform(df_input)

    results =  transform_dataframe(
                    df = df_input_2,
                    transformer = transformer,
                    fit = False,
                    inverse_transform = True
               )
    
    pd.testing.assert_frame_equal(results, df_input)


def test_transform_dataframe_when_transformer_is_ColumnTransformer():
    """
    Test the output of transform_dataframe when transformer is ColumnTransformer.
    """
    df_input = pd.DataFrame({
                'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
                'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
               })
    expected = pd.DataFrame({
                'col_1': [-1.76425513, -1.00989936, 0.59254869, 0.45863938,
                           0.1640389 , -0.25107995, 1.79326881, 0.01673866],
                'col_2_a': [1., 1., 1., 1., 0., 0., 0., 0.],
                'col_2_b': [0., 0., 0., 0., 1., 1., 1., 1.]
               })
    transformer = ColumnTransformer(
                    [('scale', StandardScaler(), ['col_1']),
                    ('onehot', OneHotEncoder(), ['col_2'])],
                    remainder = 'passthrough',
                    verbose_feature_names_out = False
                  )
    results =  transform_dataframe(
                    df = df_input,
                    transformer = transformer,
                    fit = True,
                    inverse_transform = False
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_transform_dataframe_exception_when_transformer_is_ColumnTransformer_and_inverse_transform_is_true():
    """
    Test that transform_dataframe raise exception when transformer is ColumnTransformer
    and argument inverse_transform is True.
    """
    df_input = pd.DataFrame({
                'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
                'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
               })
    transformer = ColumnTransformer(
                    [('scale', StandardScaler(), ['col_1']),
                    ('onehot', OneHotEncoder(), ['col_2'])],
                    remainder = 'passthrough',
                    verbose_feature_names_out = False
                  )

    err_msg = re.escape('`inverse_transform` is not available when using ColumnTransformers.')
    with pytest.raises(Exception, match = err_msg):
        transform_dataframe(
            df = df_input,
            transformer = transformer,
            fit = True,
            inverse_transform = True
        )
