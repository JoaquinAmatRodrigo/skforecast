
################################################################################
#                            skforecast.datsets                                #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

import pandas as pd
import textwrap


def fetch_dataset(
    name : str,
    version: str = 'latest',
    raw: bool = False,
    kwargs_read_csv: dict = {},
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch a dataset from the skforecast-datasets repository. Available datasets
    are: 'h2o', 'items_sales', 'air_pollution', 'fuel_consumption', 'web_visits',
    'bike_sharing', 'store_item_demand'.

    Parameters
    ----------
    name: str
        Name of the dataset to fetch.
    version: str, int, default `'latest'`
        Version of the dataset to fetch. If 'latest', the lastest version will be 
        fetched (the one in the main branch). For a list of available versions, 
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

    version = 'main' if version == 'latest' else f'{version}'

    datasets = {
        'h2o': {
            'url': (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/h2o.csv'
            ),
            'sep': ',',
            'index_col': 'fecha',
            'date_format': '%Y-%m-%d',
            'freq': 'MS',
            'description': (
                'Monthly expenditure ($AUD) on corticosteroid drugs that the '
                'Australian health system had between 1991 and 2008. '
            ),
            'source': (
                'Hyndman R (2023). fpp3: Data for Forecasting: Principles and Practice'
                '(3rd Edition). http://pkg.robjhyndman.com/fpp3package/,'
                'https://github.com/robjhyndman/fpp3package, http://OTexts.com/fpp3.'
            )
        },
        'h2o_exog': {
            'url': (
                f"https://raw.githubusercontent.com/JoaquinAmatRodrigo/"
                f"skforecast-datasets/{version}/data/h2o_exog.csv"
            ),
            'sep': ',',
            'index_col': 'fecha',
            'date_format': '%Y-%m-%d',
            'freq': 'MS',
            'description': (
                'Monthly expenditure ($AUD) on corticosteroid drugs that the '
                'Australian health system had between 1991 and 2008. Two additional '
                'variables (exog_1, exog_2) are simulated.'
            ),
            'source': (
                "Hyndman R (2023). fpp3: Data for Forecasting: Principles and Practice "
                "(3rd Edition). http://pkg.robjhyndman.com/fpp3package/, "
                "https://github.com/robjhyndman/fpp3package, http://OTexts.com/fpp3."
            )
        },
        'fuel_consumption': {
            'url' : (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/consumos-combustibles-mensual.csv'
            ),
            'sep': ',',
            'index_col': 'Fecha',
            'date_format': '%Y-%m-%d',
            'freq': 'MS',
            'description': (
                'Monthly fuel consumption in Spain from 1969-01-01 to 2022-08-01.'
            ),
            'source': (
                'Obtained from Corporación de Reservas Estratégicas de Productos '
                'Petrolíferos and Corporación de Derecho Público tutelada por el '
                'Ministerio para la Transición Ecológica y el Reto Demográfico. '
                'https://www.cores.es/es/estadisticas'
            )
        },
        'items_sales': {
            'url': (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/simulated_items_sales.csv'
            ),
            'sep': ',',
            'index_col': 'date',
            'date_format': '%Y-%m-%d',
            'freq': 'D',
            'description': 'Simulated time series for the sales of 3 different items.',
            'source': 'Simulated data.'
        },
        'air_quality_valencia': {
            'url': (
                f"https://raw.githubusercontent.com/JoaquinAmatRodrigo/"
                f"skforecast-datasets/{version}/data/air_quality_valencia.csv"
            ),
            'sep': ',',
            'index_col': 'datetime',
            'date_format': '%Y-%m-%d %H:%M:%S',
            'freq': 'H',
            'description': (
                'Hourly measures of several air quimical pollutant (pm2.5, co, no, '
                'no2, pm10, nox, o3, so2) at Valencia city.'
            ),
            'source': (
                " Red de Vigilancia y Control de la Contaminación Atmosférica, "
                "46250054-València - Centre, "
                "https://mediambient.gva.es/es/web/calidad-ambiental/datos-historicos."
            )
        },
        'website_visits': {
            'url' : (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/visitas_por_dia_web_cienciadedatos.csv'
            ),
            'sep': ',',
            'index_col': 'date',
            'date_format': '%Y-%m-%d',
            'freq': '1D',
            'description': (
                'Daily visits to the cienciadedatos.net website registered with the '
                'google analytics service.'
            ),
            'source': (
                "Amat Rodrigo, J. (2021). cienciadedatos.net (1.0.0). Zenodo. "
                "https://doi.org/10.5281/zenodo.10006330"
            )
        },
        'bike_sharing': {
            'url' : (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/bike_sharing_dataset_clean.csv'
            ),
            'sep': ',',
            'index_col': 'date_time',
            'date_format': '%Y-%m-%d %H:%M:%S',
            'freq': 'H',
            'description': (
                'Hourly usage of the bike share system in the city of Washington D.C. '
                'during the years 2011 and 2012. In addition to the number of users per '
                'hour, information about weather conditions and holidays is available.'
            ),
            'source': (
                "Fanaee-T,Hadi. (2013). Bike Sharing Dataset. UCI Machine Learning "
                "Repository. https://doi.org/10.24432/C5W894."
            )
        },
        'australia_tourism': {
            'url' : (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/australia_tourism.csv'
            ),
            'sep': ',',
            'index_col': 'date_time',
            'date_format': '%Y-%m-%d',
            'freq': 'Q',
            'description': (
                "Quarterly overnight trips (in thousands) from 1998 Q1 to 2016 Q4 "
                "across Australia. The tourism regions are formed through the "
                "aggregation of Statistical Local Areas (SLAs) which are defined by "
                "the various State and Territory tourism authorities according to "
                "their research and marketing needs."
            ),
            'source': (
                "Wang, E, D Cook, and RJ Hyndman (2020). A new tidy data structure to "
                "support exploration and modeling of temporal data, Journal of "
                "Computational and Graphical Statistics, 29:3, 466-478, "
                "doi:10.1080/10618600.2019.1695624."
            )
        },
        'uk_daily_flights': {
            'url': (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/uk_daily_flights.csv'
            ),
            'sep': ',',
            'index_col': 'Date',
            'date_format': '%d/%m/%Y',
            'freq': 'D',
            'description': 'Daily number of flights in UK from 02/01/2019 to 23/01/2022.',
            'source': (
                'Experimental statistics published as part of the Economic activity and '
                'social change in the UK, real-time indicators release, Published 27 '
                'January 2022. Daily flight numbers are available in the dashboard '
                'provided by the European Organisation for the Safety of Air Navigation '
                '(EUROCONTROL). '
                'https://www.ons.gov.uk/economy/economicoutputandproductivity/output/'
                'bulletins/economicactivityandsocialchangeintheukrealtimeindicators/latest'
            )
        },
        'wikipedia_visits': {
            'url': (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/wikipedia_visits.csv'
            ),
            'sep': ',',
            'index_col': 'date',
            'date_format': '%Y-%m-%d',
            'freq': 'D',
            'description': (
                'Log daily page views for the Wikipedia page for Peyton Manning. '
                'Scraped data using the Wikipediatrend package in R.'
            ),
            'source': (
                'https://github.com/facebook/prophet/blob/main/examples/'
                'example_wp_log_peyton_manning.csv'
            )
        },
        'vic_electricity': {
            'url': (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/vic_electricity.csv'
            ),
            'sep': ',',
            'index_col': 'Time',
            'date_format': '%Y-%m-%dT%H:%M:%SZ',
            'freq': '30min',
            'description': 'Half-hourly electricity demand for Victoria, Australia',
            'source': (
                "O'Hara-Wild M, Hyndman R, Wang E, Godahewa R (2022).tsibbledata: Diverse "
                "Datasets for 'tsibble'. https://tsibbledata.tidyverts.org/, "
                "https://github.com/tidyverts/tsibbledata/. "
                "https://tsibbledata.tidyverts.org/reference/vic_elec.html"
            )
        },
        'store_sales': {
            'url': (
                f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/'
                f'skforecast-datasets/{version}/data/store_sales.csv'
            ),
            'sep': ',',
            'index_col': 'date',
            'date_format': '%Y-%m-%d',
            'freq': '1D',
            'description': (
                'This dataset contains 913,000 sales transactions from 2013-01-01 to '
                '2017-12-31 for 50 products (SKU) in 10 stores.'
            ),
            'source': (
                'The original data was obtained from: inversion. (2018). Store Item '
                'Demand Forecasting Challenge. Kaggle. '
                'https://kaggle.com/competitions/demand-forecasting-kernels-only'
            )
        }
    }
    
    if name not in datasets.keys():
        raise ValueError(
            f"Dataset {name} not found. Available datasets are: {list(datasets.keys())}"
        )
    
    url = datasets[name]['url']

    try:
        sep = datasets[name]['sep']
        df = pd.read_csv(url, sep=sep, **kwargs_read_csv)
    except:
        raise ValueError(
            f"Error reading dataset {name} from {url}. Try to version = 'latest'"
        )

    if not raw:
        try:
            index_col = datasets[name]['index_col']
            freq = datasets[name]['freq']
            date_format = datasets[name]['date_format']
            df = df.set_index(index_col)
            df.index = pd.to_datetime(df.index, format=date_format)
            df = df.asfreq(freq)
            df = df.sort_index()
        except:
            pass
    
    if verbose:
        print(name)
        print('-'*len(name))
        description = textwrap.fill(datasets[name]['description'], width=80)
        source = textwrap.fill(datasets[name]['source'], width=80)
        print(description)
        print(source)
        print(f"Shape of the dataset: {df.shape}")

    return df


def load_demo_dataset(version: str = 'latest') -> pd.Series:
    """
    Load demo data set with monthly expenditure ($AUD) on corticosteroid drugs that
    the Australian health system had between 1991 and 2008. Obtained from the book:
    Forecasting: Principles and Practice by Rob J Hyndman and George Athanasopoulos.
    Index is set to datetime with monthly frequency and sorted.

    Parameters
    ----------
    version: str, default `'latest'`
        Version of the dataset to fetch. If 'latest', the lastest version will be
        fetched (the one in the main branch). For a list of available versions,
        see the repository branches.

    Returns
    -------
    df: pandas Series
        Dataset.
    
    """

    version = 'main' if version == 'latest' else f'{version}'

    url = (
        f'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast-datasets/{version}/'
        'data/h2o.csv'
    )

    df = pd.read_csv(url, sep=',', header=0, names=['y', 'datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    df = df.set_index('datetime')
    df = df.asfreq('MS')
    df = df['y']
    df = df.sort_index()

    return df