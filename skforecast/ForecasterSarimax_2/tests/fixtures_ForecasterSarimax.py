# Fixtures ForecasterSarimax
# ==============================================================================
import numpy as np
import pandas as pd

# Fixtures
# np.random.seed(123)
# y = np.random.rand(50)
# exog = np.random.rand(50)
y = pd.Series(
        data = np.array(
                   [0.379808  , 0.361801  , 0.410534  , 0.48338867, 0.47546342,
                    0.53476104, 0.56860613, 0.59522329, 0.77125778, 0.7515028 ,
                    0.38755434, 0.42728322, 0.41389018, 0.42885882, 0.47012642,
                    0.50920969, 0.558443  , 0.60151406, 0.6329471 , 0.69960539,
                    0.96308051, 0.81932534, 0.4376698 , 0.50612127, 0.47049117,
                    0.51069626, 0.54051379, 0.55811892, 0.67285206, 0.68589738,
                    0.68969198, 0.74130358, 0.81330763, 0.80311257, 0.47525824,
                    0.5525723 , 0.52710782, 0.56124982, 0.58897764, 0.62313362,
                    0.74083723, 0.72537176, 0.81580302, 0.81400947, 0.92665305,
                    0.93727594, 0.52876165, 0.55933994, 0.57787166, 0.61492741]
               ),
        name = 'y'
    )

exog = pd.Series(
           data = np.array(
                      [1.1660294 , 1.1178592 , 1.0679422 , 1.09737593, 1.12219902,
                       1.15318963, 1.19455065, 1.23148851, 1.28906233, 1.34427021,
                       1.31482887, 1.28656429, 1.25029766, 1.18181787, 1.1255426 ,
                       1.14987367, 1.17610562, 1.2136304 , 1.25444805, 1.30034385,
                       1.39111801, 1.44329448, 1.41052563, 1.38516046, 1.33933762,
                       1.24886077, 1.19309846, 1.21718828, 1.25053444, 1.29361568,
                       1.32941483, 1.36957278, 1.42061053, 1.44666263, 1.4045348 ,
                       1.37711086, 1.33427171, 1.28386015, 1.24103316, 1.27060824,
                       1.30826123, 1.34791401, 1.39882465, 1.44383102, 1.50453491,
                       1.54382265, 1.50450063, 1.45320801, 1.40598045, 1.34363532]
                  ),
           name = 'exog'
       )

exog_predict = pd.Series(
                   data = np.array(
                              [1.27501789, 1.31081724, 1.34284965, 1.37614005, 1.41412559,
                               1.45299631, 1.5056625 , 1.53112882, 1.47502858, 1.4111122 ]
                          ),
                   name = 'exog',
                   index = pd.RangeIndex(start=50, stop=60)
               )

df_exog = pd.DataFrame({
              'exog_1': exog.to_numpy(),
              'exog_2': ['a']*25+['b']*25}
          )
df_exog_predict = df_exog.copy()
df_exog_predict.index = pd.RangeIndex(start=50, stop=100)

y_datetime = pd.Series(data=y.to_numpy())
y_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
y_datetime.name = 'y'

lw_datetime = pd.Series(data=y.to_numpy())
lw_datetime.index = pd.date_range(start='2050', periods=50, freq='A')
lw_datetime.name = 'y'

exog_datetime = pd.Series(data=exog.to_numpy())
exog_datetime.index = pd.date_range(start='2000', periods=50, freq='A')
exog_datetime.name = 'exog'

exog_predict_datetime = pd.Series(data=exog_predict.to_numpy())
exog_predict_datetime.index = pd.date_range(start='2050', periods=10, freq='A')
exog_predict_datetime.name = 'exog'

lw_exog_datetime = pd.Series(data=exog.to_numpy())
lw_exog_datetime.index = pd.date_range(start='2050', periods=50, freq='A')
lw_exog_datetime.name = 'exog'

exog_predict_lw_datetime = pd.Series(data=exog_predict.to_numpy())
exog_predict_lw_datetime.index = pd.date_range(start='2100', periods=10, freq='A')
exog_predict_lw_datetime.name = 'exog'

df_exog_datetime = df_exog.copy()
df_exog_datetime.index = pd.date_range(start='2000', periods=50, freq='A')

df_lw_exog_datetime = df_exog.copy()
df_lw_exog_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

df_exog_predict_datetime = df_exog.copy()
df_exog_predict_datetime.index = pd.date_range(start='2100', periods=50, freq='A')