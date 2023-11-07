
################################################################################
#                            skforecast.datsets                                #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import pandas as pd


def fetch_dataset(
    name : str,
    version: str = 'latest',
    raw: bool = False,
    kwargs_read_csv: dict = {},
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch a dataset from the skforecast repository.

    Parameters
    ----------
    name: str
        Name of the dataset to fetch.
    version: str, int, default `'latest'`
        Version of the dataset to fetch. If 'latest', the lastest version will be 
        fetched (the one in the master branch). For a list of available versions, 
        see the repository branches.
    raw: bool, default `False`
        If True, the raw dataset is fetched. If False, the preprocessed dataset 
        is fetched. The preprocessing consists of setting the column with the 
        date/time as index and converting the index to datetime. A frequency is 
        also set to the index.
    kwargs_read_csv: dict, default `{}`
        Kwargs to pass to pandas `read_csv` function.
    verbose: bool, default `True`
        If True, print information about the dataset.
    
    Returns
    -------
    df: pandas DataFrame
        Dataset.
    
    """

    version = 'master' if version == 'latest' else f'{version}'

    datasets = {
        'h2o': {
            'url': 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o.csv',
            'sep': ',',
            'index_col': 'fecha',
            'date_format': '%Y-%m-%d',
            'freq': 'MS',
            'description': (
                'Monthly expenditure ($AUD) on corticosteroid drugs that the Australian '
                'health system had between 1991 and 2008.\nObtained from the book: '
                'Forecasting: Principles and Practice by Rob J Hyndman and George Athanasopoulos.'
            )
        },
        'items_sales': {
            'url': (
                'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/'
                'data/simulated_items_sales.csv'
            ),
            'sep': ',',
            'index_col': 'date',
            'date_format': '%Y-%m-%d',
            'freq': 'D',
            'description': 'Simulated time series for the sales of 3 different items.'
        },
        'air_pollution': {
            'url' : (
                'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/'
                'data/guangyuan_air_pollution.csv'
            ),
            'sep': ',',
            'index_col': 'date',
            'date_format': '%Y-%m-%d',
            'freq': 'D',
            'description': ''
        },
        'fuel_consumption': {
            'url' : (
                'https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/'
                'master/data/consumos-combustibles-mensual.csv'
            ),
            'sep': ',',
            'index_col': 'Fecha',
            'date_format': '%Y-%m-%d',
            'freq': 'MS',
            'description': (
                'Monthly fuel consumption in Spain from 1969-01-01 to 2022-08-01.'
                '\nObtained from Corporación de Reservas Estratégicas de Productos '
                'Petrolíferos and Corporación de Derecho Público tutelada por el '
                'Ministerio para la Transición Ecológica y el Reto Demográfico. '
                'https://www.cores.es/es/estadisticas'
            )
        },
        'web_visits': {
            'url' : (
                'https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/'
                'master/data/visitas_por_dia_web_cienciadedatos.csv'
            ),
            'sep': ',',
            'index_col': 'date',
            'date_format': '%Y-%m-%d',
            'freq': '1D',
            'description': (
                'Daily visits to the cienciadedatos.net website registered with the google '
                'analytics service.'
            )
        },
        'bike_sharing': {
            'url' : (
                'https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-'
                'learning-python/master/data/bike_sharing_dataset_clean.csv'
            ),
            'sep': ',',
            'index_col': 'date_time',
            'date_format': '%Y-%m-%d %H:%M:%S',
            'freq': 'H',
            'description': (
                """
                Hourly usage of the bike share system in the city of Washington,
                D.C. during the years 2011 and 2012. In addition to the number of
                users per hour, information about weather conditions and holidays
                is available. The original data was obtained from:
                Fanaee-T,Hadi. (2013). Bike Sharing Dataset. UCI Machine Learning
                Repository. https://doi.org/10.24432/C5W894.
                """
            )
        },
        'store_item_demand': {
            'url' : (
                'https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-'
                'learning-python/master/data/bike_sharing_dataset_clean.csv'
            ),
            'sep': ',',
            'index_col': 'date_time',
            'date_format': '%Y-%m-%d %H:%M:%S',
            'freq': 'H',
            'description': (
                """
                This dataset contains 913,000 sales transactions from 2013-01-01
                to 2017-12-31 for 50 products (SKU) in 10 stores. The original data
                was obtained from:
                inversion. (2018). Store Item Demand Forecasting Challenge. Kaggle.
                https://kaggle.com/competitions/demand-forecasting-kernels-only
                """
            )
        },
    }
    
    if name not in datasets.keys():
        raise ValueError(
            f"Dataset {name} not found. Available datasets are: {list(datasets.keys())}"
        )
    
    url = datasets[name]['url']
    if version != 'latest':
        url = url.replace('master', f'{version}')

    try:
        sep = datasets[name]['sep']
        df = pd.read_csv(url, sep=sep, **kwargs_read_csv)
    except:
        raise ValueError(
            f"Error reading dataset {name} from {url}. Try to version = 'latest'"
        )

    if not raw:
        index_col = datasets[name]['index_col']
        freq = datasets[name]['freq']
        date_format = datasets[name]['date_format']
        df = df.set_index(index_col)
        df.index = pd.to_datetime(df.index, format=date_format)
        df.index.freq = freq
        df = df.sort_index()
    
    if verbose:
        print(name)
        print('-'*len(name))
        print(datasets[name]['description'])
        print(f"Shape of the dataset: {df.shape}")

    return df


def load_demo_dataset() -> pd.Series:
    """
    Load demo data set with monthly expenditure ($AUD) on corticosteroid drugs that
    the Australian health system had between 1991 and 2008. Obtained from the book:
    Forecasting: Principles and Practice by Rob J Hyndman and George Athanasopoulos.
    Index is set to datetime with monthly frequency and sorted.

    Parameters
    ----------
    None

    Returns
    -------
    df: pandas Series
        Dataset.
    
    """

    url = (
        'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/'
        'data/h2o.csv'
    )

    df = pd.read_csv(url, sep=',', header=0, names=['y', 'datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    df = df.set_index('datetime')
    df = df.asfreq('MS')
    df = df['y']
    df = df.sort_index()

    return df