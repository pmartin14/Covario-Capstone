# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:24:46 2022

@author: payma
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import pytz

july1 = pd.read_csv('July_1_mids.csv', index_col=0)

july5 = pd.read_csv('July_5_mids.csv', index_col=0)

july10 = pd.read_csv('July_10_mids.csv', index_col=0)

july15 = pd.read_csv('July_15_mids.csv', index_col=0)

july20 = pd.read_csv('July_20_mids.csv', index_col=0)

july25 = pd.read_csv('July_25_mids.csv', index_col=0)

july30 = pd.read_csv('July_30_mids.csv', index_col=0)


def squared_error(df, seconds_window, name):
    
    vol_mids = df.loc[:,name]
    vol_mids = vol_mids[:-seconds_window]
    future_mids = df.loc[:,'Mid Prices']
    
    vol_mids = np.asarray(vol_mids)
    future_mids = np.asarray(future_mids)
    future_mids = future_mids[seconds_window:]

    
    sqdif = (vol_mids - future_mids)**2
    
    squared_error = sum(sqdif)
    
    return squared_error

def change(df, seconds_window, name):
    
    vol_mids = df.loc[:,name]
    
    vol_diffs = vol_mids.diff(periods = seconds_window).dropna()
    vol_mids = np.asarray(vol_mids)
    
    return sum(vol_diffs**2)

def r_squared(df, seconds_window, name):
    
    vol_mids = df.loc[:,name]
    vol_mids = vol_mids[:-seconds_window]
    future_mids = df.loc[:,'Mid Prices']
    
    vol_mids = np.asarray(vol_mids)
    future_mids = np.asarray(future_mids)
    future_mids = future_mids[seconds_window:]
    
    X = sm.add_constant(vol_mids)
    res = sm.OLS(future_mids, X).fit()
    
    rsq = res.rsquared
    
    return rsq

def simple_trading(mid_prices, fair_prices):
    """
    Parameters
    ----------
    mid_prices : array-like (every second)
    fair_prices : array-like (every second)

    Returns
    -------
    P&L
    """
    #Set this threshold to 1 standard deviation of your (fair_price - mid_price) (FROM TRAINING SET)
    THRESHOLD = 0.85
    position = 0
    num_trades = 0
    value = 1
    for i in range(len(mid_prices)):
        signal = fair_prices[i] - mid_prices[i]
        if abs(signal) > THRESHOLD:
            if signal < 0:
                if position==0:
                    initial = mid_prices[i]
                    position = -1
                    num_trades += 1
                if position > 0:
                    final = mid_prices[i]
                    pct_change = position*((final - initial)/initial)
                    value = value*(1+pct_change)
                    initial = final
                    position = -1
                    num_trades += 2
            elif signal > 0:
                if position==0:
                    initial = mid_prices[i]
                    position = 1
                    num_trades += 1
                if position < 0:
                    final = mid_prices[i]
                    pct_change = position*((final - initial)/initial)
                    value = value*(1+pct_change)
                    initial = final
                    position = 1
                    num_trades += 2
    return value, num_trades


sec1_sqer = []
sec5_sqer = []
sec15_sqer = []
sec30_sqer = []
sec60_sqer = []

sec_list = [1, 5, 15, 30, 60]
df_list = [july20, july25, july30]


for i in range(len(df_list)):
    sec1_sqer.append(squared_error(df_list[i], 1, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec5_sqer.append(squared_error(df_list[i], 5, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec15_sqer.append(squared_error(df_list[i], 15, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec30_sqer.append(squared_error(df_list[i], 30, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec60_sqer.append(squared_error(df_list[i], 60, 'vol_mid_10'))
    
print(np.mean(sec1_sqer))
print(np.mean(sec5_sqer))
print(np.mean(sec15_sqer))
print(np.mean(sec30_sqer))
print(np.mean(sec60_sqer))

sec1_sqer = []
sec5_sqer = []
sec15_sqer = []
sec30_sqer = []
sec60_sqer = []

sec_list = [1, 5, 15, 30, 60]
df_list = [july20, july25, july30]


for i in range(len(df_list)):
    sec1_sqer.append(change(df_list[i], 1, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec5_sqer.append(change(df_list[i], 5, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec15_sqer.append(change(df_list[i], 15, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec30_sqer.append(change(df_list[i], 30, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec60_sqer.append(change(df_list[i], 60, 'vol_mid_10'))
    
print(np.mean(sec1_sqer))
print(np.mean(sec5_sqer))
print(np.mean(sec15_sqer))
print(np.mean(sec30_sqer))
print(np.mean(sec60_sqer))

sec1_sqer = []
sec5_sqer = []
sec15_sqer = []
sec30_sqer = []
sec60_sqer = []

sec_list = [1, 5, 15, 30, 60]
df_list = [july20, july25, july30]


for i in range(len(df_list)):
    sec1_sqer.append(r_squared(df_list[i], 1, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec5_sqer.append(r_squared(df_list[i], 5, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec15_sqer.append(r_squared(df_list[i], 15, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec30_sqer.append(r_squared(df_list[i], 30, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec60_sqer.append(r_squared(df_list[i], 60, 'vol_mid_10'))
    
print(np.mean(sec1_sqer))
print(np.mean(sec5_sqer))
print(np.mean(sec15_sqer))
print(np.mean(sec30_sqer))
print(np.mean(sec60_sqer))

sec1_sqer = []
sec5_sqer = []
sec15_sqer = []
sec30_sqer = []
sec60_sqer = []

sec_list = [1, 5, 15, 30, 60]
df_list = [july20, july25, july30]


for i in range(len(df_list)):
    sec1_sqer.append(simple_trading(df_list[i].loc[:,'Mid Prices'], df_list[i].loc[:,'vol_mid_10']))
    
for i in range(len(df_list)):
    sec5_sqer.append(r_squared(df_list[i], 5, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec15_sqer.append(r_squared(df_list[i], 15, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec30_sqer.append(r_squared(df_list[i], 30, 'vol_mid_10'))
    
for i in range(len(df_list)):
    sec60_sqer.append(r_squared(df_list[i], 60, 'vol_mid_10'))
    
pnls = []
trades = []
for i in range(len(sec1_sqer)):
    pnls.append(sec1_sqer[i][0])
    trades.append(sec1_sqer[i][1])

print(np.mean(pnls))
print(np.mean(trades))
print(np.mean(sec15_sqer))
print(np.mean(sec30_sqer))
print(np.mean(sec60_sqer))


sec1_sqer = []
sec5_sqer = []
sec15_sqer = []
sec30_sqer = []
sec60_sqer = []

sec_list = [1, 5, 15, 30, 60]
df_list = [july20, july25, july30]


for i in range(len(df_list)):
    sec1_sqer.append(squared_error(df_list[i], 1, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec5_sqer.append(squared_error(df_list[i], 5, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec15_sqer.append(squared_error(df_list[i], 15, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec30_sqer.append(squared_error(df_list[i], 30, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec60_sqer.append(squared_error(df_list[i], 60, 'Mid Prices'))
    
print(np.mean(sec1_sqer)/86400)
print(np.mean(sec5_sqer)/86400)
print(np.mean(sec15_sqer)/86400)
print(np.mean(sec30_sqer)/86400)
print(np.mean(sec60_sqer)/86400)

sec1_sqer = []
sec5_sqer = []
sec15_sqer = []
sec30_sqer = []
sec60_sqer = []

sec_list = [1, 5, 15, 30, 60]
df_list = [july20, july25, july30]


for i in range(len(df_list)):
    sec1_sqer.append(change(df_list[i], 1, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec5_sqer.append(change(df_list[i], 5, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec15_sqer.append(change(df_list[i], 15, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec30_sqer.append(change(df_list[i], 30, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec60_sqer.append(change(df_list[i], 60, 'Mid Prices'))
    
print(np.mean(sec1_sqer)/86400)
print(np.mean(sec5_sqer)/86400)
print(np.mean(sec15_sqer)/86400)
print(np.mean(sec30_sqer)/86400)
print(np.mean(sec60_sqer)/86400)

sec1_sqer = []
sec5_sqer = []
sec15_sqer = []
sec30_sqer = []
sec60_sqer = []

sec_list = [1, 5, 15, 30, 60]
df_list = [july20, july25, july30]


for i in range(len(df_list)):
    sec1_sqer.append(r_squared(df_list[i], 1, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec5_sqer.append(r_squared(df_list[i], 5, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec15_sqer.append(r_squared(df_list[i], 15, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec30_sqer.append(r_squared(df_list[i], 30, 'Mid Prices'))
    
for i in range(len(df_list)):
    sec60_sqer.append(r_squared(df_list[i], 60, 'Mid Prices'))
    
print(np.mean(sec1_sqer))
print(np.mean(sec5_sqer))
print(np.mean(sec15_sqer))
print(np.mean(sec30_sqer))
print(np.mean(sec60_sqer))



