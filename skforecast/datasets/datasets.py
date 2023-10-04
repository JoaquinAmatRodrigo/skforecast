
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
        version: str = 'active',
        raw: bool = False,
        read_csv_kwargs: dict = None,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch a dataset from skforecast repository.

    Parameters
    ----------
    name: str
        Name of the dataset to fetch.
    version: str or int, default 'active'
        Version of the dataset to fetch. If 'active', the last version will be fetched
        (the one in the master branch). For a list of available versions, see the
        repository branchs.
    raw: bool, default False
        If True, the raw dataset will be fetched. If False, the preprocessed dataset will
        be fetched. The preprocessing consists in seting the column with the date/time as index,
        and convert the index to datetime. Also, a frequency is setted to the index.
    read_csv_kwargs: dict, default None
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
            'filename': 'file_1.csv',
            'url': 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o.csv',
            'index_col': 'date',
            'freq': 'D',
            'description': 'This dataset contains information about X.'
            },
        'dataset_2': {
            'filename': 'file_2.csv',
            'index_col': 'date',
            'freq': 'D',
            'description': 'This dataset contains information about Y.'
            },
        'dataset_3': {
            'filename': 'file_3.csv',
            'index_col': 'date',
            'freq': 'D',
            'description': 'This dataset contains information about Z.'
            }
        }
    
    if name not in datasets.keys():
        raise ValueError(
            f"Dataset {name} not found. Available datasets are: {list(datasets.keys())}"
        )
    
    if version != 'active':
        url = url.replace('master', f'{version}')

    try:
        df = pd.read_csv(datasets['name']['url'], **read_csv_kwargs)
    except:
        raise ValueError(
            f"Error reading dataset {name} from {url}. Try to version = 'active'"
        )

    if not raw:
        df = df.set_index(datasets['name']['index_col'])
        df.index = pd.to_datetime(df.index)
        df.index.freq = datasets['name']['freq']
    
    if verbose:
        print(datasets['name']['info'])

    return df