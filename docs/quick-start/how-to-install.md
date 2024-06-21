# Installation Guide

This guide will help you install `skforecast`, a powerful library for time series forecasting in Python. The default installation of `skforecast` includes only the essential dependencies required for basic functionality. Additional optional dependencies can be installed for extended features.

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue) [![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/)


## **Basic installation**

To install the basic version of `skforecast` with its core dependencies, run:

```bash
pip install skforecast
```

Specific version:

```bash
pip install skforecast==0.13.0
```

Latest (unstable):

```bash
pip install git+https://github.com/JoaquinAmatRodrigo/skforecast#master
```

The following dependencies are installed with the default installation:

+ numpy>=1.20, <1.27
+ pandas>=1.2, <2.3
+ tqdm>=4.57, <4.67
+ scikit-learn>=1.2, <1.5
+ optuna>=2.10, <3.7
+ joblib>=1.1, <1.5


## **Optional dependencies**

To install the full version with all optional dependencies:

```bash
pip install skforecast[full]
```

For specific use cases, you can install these dependencies as needed:

### Sarimax

```bash
pip install skforecast[sarimax]
```

+ pmdarima>=2.0, <2.1
+ statsmodels>=0.12, <0.15


### Plotting

```bash
pip install skforecast[plotting]
```

+ matplotlib>=3.3, <3.9
+ seaborn>=0.11, <0.14
+ statsmodels>=0.12, <0.15


### Deeplearning

```bash
pip install skforecast[deeplearning]
```

+ matplotlib>=3.3, <3.9
+ keras>=2.6, <4.0