
################################################################################
#                            skforecast.datsets                                #
#                                                                              #
# This work by Joaquin Amat Rodrigo and Javier Escobar Ortiz is licensed       #
# under the BSD 3-Clause License.                                              #
################################################################################
# coding=utf-8

import pandas as pd


def fetch_dataset(
        name : str,
        version: str = 'latest',
        raw: bool = False,
        read_csv_kwargs: dict = {},
        verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch a dataset from skforecast repository.

    Parameters
    ----------
    name: str
        Name of the dataset to fetch.
    version: str or int, default 'latest'
        Version of the dataset to fetch. If 'latest', the last version will be fetched
        (the one in the master branch). For a list of available versions, see the
        repository branchs.
    raw: bool, default False
        If True, the raw dataset will be fetched. If False, the preprocessed dataset will
        be fetched. The preprocessing consists in seting the column with the date/time as index,
        and convert the index to datetime. Also, a frequency is setted to the index.
    read_csv_kwargs: dict, default {}
        Kwargs to pass to pandas read_csv function.
    verbose: bool, default True
        If True, print information about the dataset.
    
    Returns
    -------
    df: pandas dataframe
        Dataset.
    """

    version = 'master' if version == 'active' else f'{version}'

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
            }
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
        df = pd.read_csv(url, sep=sep, **read_csv_kwargs)
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
        print(datasets[name]['description'])
        print(f"Shape of the dataset: {df.shape}")

    return df


def load_demo_dataset() -> pd.Series:
    """
    Load demo data set with monthly expenditure ($AUD) on corticosteroid drugs that
    the Australian health system had between 1991 and 2008. Obtained from the book:
    Forecasting: Principles and Practice by Rob J Hyndman and George Athanasopoulos.
    Index is set to datetime with monthly frequency and sorted.

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