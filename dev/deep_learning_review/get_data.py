"""Download the datasets used by the experiments into ./data."""
import os
from skforecast.datasets import fetch_dataset

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
for name in ["store_sales", "bike_sharing_extended_features"]:
    fetch_dataset(name, verbose=False).to_pickle(os.path.join(DATA, f"{name}.pkl"))
