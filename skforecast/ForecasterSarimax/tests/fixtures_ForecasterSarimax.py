# Fixtures ForecasterSarimax
# ==============================================================================
import numpy as np
import pandas as pd

# Fixtures
# np.random.seed(123)
# y = np.random.rand(50)
# exog = np.random.rand(50)
y = pd.Series(
        data = np.array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897,
                         0.42310646, 0.9807642 , 0.68482974, 0.4809319 , 0.39211752,
                         0.34317802, 0.72904971, 0.43857224, 0.0596779 , 0.39804426,
                         0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759,
                         0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338,
                         0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612,
                         0.09210494, 0.43370117, 0.43086276, 0.4936851 , 0.42583029,
                         0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668,
                         0.62395295, 0.1156184 , 0.31728548, 0.41482621, 0.86630916,
                         0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453]
            ),
        name = 'y'
    )

exog = pd.Series(
           data = np.array([0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                            0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234,
                            0.66931378, 0.58593655, 0.6249035 , 0.67468905, 0.84234244,
                            0.08319499, 0.76368284, 0.24366637, 0.19422296, 0.57245696,
                            0.09571252, 0.88532683, 0.62724897, 0.72341636, 0.01612921,
                            0.59443188, 0.55678519, 0.15895964, 0.15307052, 0.69552953,
                            0.31876643, 0.6919703 , 0.55438325, 0.38895057, 0.92513249,
                            0.84167   , 0.35739757, 0.04359146, 0.30476807, 0.39818568,
                            0.70495883, 0.99535848, 0.35591487, 0.76254781, 0.59317692,
                            0.6917018 , 0.15112745, 0.39887629, 0.2408559 , 0.34345601]
               ),
           name = 'exog'
       )

exog_predict = pd.Series(
                  data = np.array([0.12062867, 0.8263408 , 0.60306013, 0.54506801, 0.34276383,
                                   0.30412079, 0.41702221, 0.68130077, 0.87545684, 0.51042234]
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

lw_exog_datetime = pd.Series(data=exog.to_numpy())
lw_exog_datetime.index = pd.date_range(start='2050', periods=50, freq='A')
lw_exog_datetime.name = 'exog'

exog_predict_datetime = pd.Series(data=exog_predict.to_numpy())
exog_predict_datetime.index = pd.date_range(start='2100', periods=10, freq='A')
exog_predict_datetime.name = 'exog'

df_exog_datetime = df_exog.copy()
df_exog_datetime.index = pd.date_range(start='2000', periods=50, freq='A')

df_lw_exog_datetime = df_exog.copy()
df_lw_exog_datetime.index = pd.date_range(start='2050', periods=50, freq='A')

df_exog_predict_datetime = df_exog.copy()
df_exog_predict_datetime.index = pd.date_range(start='2100', periods=50, freq='A')