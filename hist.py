def address(ticker):
    key = open('auth.txt','r').read()
    return f'https://financialmodelingprep.com/stable/historical-price-eod/light?symbol={ticker}&apikey={key}'

# Stock Historical Data Collector

import numpy as np
import pandas as pd
import time
import requests
import json

portfolio = ['AAPL','MSFT','NVDA','SPY','JPM']

close = []
the_date = []
for stock in portfolio:
    url = address(stock)
    resp = requests.get(url).json()
    df = pd.DataFrame(resp)[::-1]
    the_prices = df['price'].values.tolist()
    close.append(the_prices)
    time.sleep(0.5)
    print('Stock has loaded: ', stock)

close = np.array(close).T

ds = pd.DataFrame(close, columns=portfolio)

print(ds)

ds.to_csv('data.csv')