import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skforecast",
    version="0.0.1",
    author="Joaquin Amat Rodrigo",
    author_email="j.amatrodrigo@gmail.com",
    description="Forecasting time series with scikitlearn regressors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JoaquinAmatRodrigo/skforecast",
    packages=setuptools.find_packages(),
    classifiers=[]
)
