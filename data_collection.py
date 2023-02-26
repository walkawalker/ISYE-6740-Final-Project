# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:06:32 2022

@author: whill
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from ta import add_all_ta_features
import matplotlib
import requests
import time
import pickle
import csv
api_key = 'Replaced because its a private key'
frames = {}
sp500_df = pd.read_csv('data_project//Stocks in the SP 500 Index.csv')
tickers = sp500_df['Symbol'].tolist()
#testtick = tickers[4:10]
#print(testtick[0])
for id_ticker in tickers:
    print(id_ticker)
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={id_ticker}&apikey=api_key&outputsize=full'
    r = requests.get(url)
    data2 = r.json()
    dailydata = data2.get('Time Series (Daily)')
    d_dates = []
    d_open = []
    d_high = []
    d_low = []
    d_close = []
    d_vol = []
    d_split = []
    for x, y in dailydata.items():
        d_dates.append(x)
        d_open.append(y.get('1. open'))
        d_high.append(y.get('2. high'))
        d_low.append(y.get('3. low'))
        d_close.append(y.get('4. close'))
        d_vol.append(y.get('6. volume'))
        d_split.append(y.get('8. split coefficient'))
    #fulldata = [m_dates, m_open, m_high, m_low, m_close, m_vol]
    dailystockdf = pd.DataFrame([d_dates, d_open, d_high, d_low, d_close, d_vol, d_split]).T 
    dailystockdf.rename(columns={0: 'Dates', 1: 'Open',2: 'High',3: 'Low',4: 'Close',5: 'Volume', 6: 'Split Coef'}, inplace = True)
    cols = [ 'Open','High', 'Low', 'Close', 'Volume','Split Coef']
    dailystockdf = dailystockdf.reindex(index=dailystockdf.index[::-1])
    dailystockdf.reset_index(inplace = True, drop = True)
    dailystockdf[cols] = dailystockdf[cols].apply(pd.to_numeric)
    for v in range(len(dailystockdf)):
        if dailystockdf['Split Coef'].iloc[v] != 1:
            dailystockdf['Open'].iloc[0:v] = dailystockdf['Open'].iloc[0:v]/dailystockdf['Split Coef'].iloc[v] 
            dailystockdf['High'].iloc[0:v] = dailystockdf['High'].iloc[0:v]/dailystockdf['Split Coef'].iloc[v] 
            dailystockdf['Low'].iloc[0:v] = dailystockdf['Low'].iloc[0:v]/dailystockdf['Split Coef'].iloc[v] 
            dailystockdf['Close'].iloc[0:v] = dailystockdf['Close'].iloc[0:v]/dailystockdf['Split Coef'].iloc[v] 
            dailystockdf['Volume'].iloc[0:v] = dailystockdf['Volume'].iloc[0:v]*dailystockdf['Split Coef'].iloc[v] 
    '''
    #performance
    dailystockdf.ta.log_return(cumulative=True, append=True)
    #momentum
    dailystockdf.ta.rsi(append=True, fillna=0)
    #statistics
    dailystockdf.ta.entropy(append=True, fillna=0)
    dailystockdf.ta.kurtosis(append=True, fillna=0)
    dailystockdf.ta.median(append=True, fillna=0)
    dailystockdf.ta.variance(append=True, fillna=0)
    dailystockdf.ta.skew(append=True, fillna=0)
    #trend
    dailystockdf.ta.decay(append=True, fillna=0)
    '''
    dailystockdf = add_all_ta_features(dailystockdf, open='Open', high='High', low='Low',close='Close',volume='Volume',fillna=True )
    frames.update({f'{id_ticker}':dailystockdf})
    time.sleep(10)
#dailystockdf.to_csv('amznexample.csv')
with open("data_project/stocksandindicators.pickle", "wb") as outfile:
    pickle.dump(frames, outfile)