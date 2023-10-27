#Real-time Stock Price Data Visualization from Yahoo Stocks using Python

##imported libraries
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
today = date.today()

#configuring date
d1 = today.strftime('%Y-%m-%d')

##importing streamlit library
import streamlit as st

import yfinance as yf
stock_list = ["VEEV", "GOOG"]
print('stock_list:', stock_list)
data = yf.download(stock_list, start="2020-01-01", end=d1)
print('data fields downloaded:', set(data.columns.get_level_values(0)))
st.table(data.tail())
