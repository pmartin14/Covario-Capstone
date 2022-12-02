# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:11:02 2022

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

vol_mids = july1.loc[:,'vol_mid_10']
vol_mids = vol_mids[:-1]
future_mids = july1.loc[:,'Mid Prices']

vol_mids = np.asarray(vol_mids)
future_mids = np.asarray(future_mids)
future_mids = future_mids[1:]


sqdif = ((vol_mids - future_mids)**2)

squared_error = sum(sqdif)





july1_1_10_sqer = squared_error(july1, 1, 'vol_mid_10')
july1_1_50_sqer = squared_error(july1, 1, 'vol_mid_50')
july1_1_100_sqer = squared_error(july1, 1, 'vol_mid_100')
july1_1_200_sqer = squared_error(july1, 1, 'vol_mid_200')

july1_5_10_sqer = squared_error(july1, 5, 'vol_mid_10')
july1_5_50_sqer = squared_error(july1, 5, 'vol_mid_50')
july1_5_100_sqer = squared_error(july1, 5, 'vol_mid_100')
july1_5_200_sqer = squared_error(july1, 5, 'vol_mid_200')

july1_15_10_sqer = squared_error(july1, 15, 'vol_mid_10')
july1_15_50_sqer = squared_error(july1, 15, 'vol_mid_50')
july1_15_100_sqer = squared_error(july1, 15, 'vol_mid_100')
july1_15_200_sqer = squared_error(july1, 15, 'vol_mid_200')

july1_30_10_sqer = squared_error(july1, 30, 'vol_mid_10')
july1_30_50_sqer = squared_error(july1, 30, 'vol_mid_50')
july1_30_100_sqer = squared_error(july1, 30, 'vol_mid_100')
july1_30_200_sqer = squared_error(july1, 30, 'vol_mid_200')

july1_60_10_sqer = squared_error(july1, 60, 'vol_mid_10')
july1_60_50_sqer = squared_error(july1, 60, 'vol_mid_50')
july1_60_100_sqer = squared_error(july1, 60, 'vol_mid_100')
july1_60_200_sqer = squared_error(july1, 60, 'vol_mid_200')


july5_1_10_sqer = squared_error(july5, 1, 'vol_mid_10')
july5_1_50_sqer = squared_error(july5, 1, 'vol_mid_50')
july5_1_100_sqer = squared_error(july5, 1, 'vol_mid_100')
july5_1_200_sqer = squared_error(july5, 1, 'vol_mid_200')

july5_5_10_sqer = squared_error(july5, 5, 'vol_mid_10')
july5_5_50_sqer = squared_error(july5, 5, 'vol_mid_50')
july5_5_100_sqer = squared_error(july5, 5, 'vol_mid_100')
july5_5_200_sqer = squared_error(july5, 5, 'vol_mid_200')

july5_15_10_sqer = squared_error(july5, 15, 'vol_mid_10')
july5_15_50_sqer = squared_error(july5, 15, 'vol_mid_50')
july5_15_100_sqer = squared_error(july5, 15, 'vol_mid_100')
july5_15_200_sqer = squared_error(july5, 15, 'vol_mid_200')

july5_30_10_sqer = squared_error(july5, 30, 'vol_mid_10')
july5_30_50_sqer = squared_error(july5, 30, 'vol_mid_50')
july5_30_100_sqer = squared_error(july5, 30, 'vol_mid_100')
july5_30_200_sqer = squared_error(july5, 30, 'vol_mid_200')

july5_60_10_sqer = squared_error(july5, 60, 'vol_mid_10')
july5_60_50_sqer = squared_error(july5, 60, 'vol_mid_50')
july5_60_100_sqer = squared_error(july5, 60, 'vol_mid_100')
july5_60_200_sqer = squared_error(july5, 60, 'vol_mid_200')

july1np = np.zeros((5, 4))
july5np = np.zeros((5, 4))

july1df = pd.DataFrame(july1np, columns=['10k', '50k', '100k', '200k'])
july5df = pd.DataFrame(july5np, columns=['10k', '50k', '100k', '200k'])

july5_10k = [july5_1_10_sqer, july5_5_10_sqer, july5_15_10_sqer, july5_30_10_sqer, july5_60_10_sqer]
july5_50k = [july5_1_50_sqer, july5_5_50_sqer, july5_15_50_sqer, july5_30_50_sqer, july5_60_50_sqer]
july5_100k = [july5_1_100_sqer, july5_5_100_sqer, july5_15_100_sqer, july5_30_100_sqer, july5_60_100_sqer]
july5_200k = [july5_1_200_sqer, july5_5_200_sqer, july5_15_200_sqer, july5_30_200_sqer, july5_60_200_sqer]

july1_10k = [july1_1_10_sqer, july1_5_10_sqer, july1_15_10_sqer, july1_30_10_sqer, july1_60_10_sqer]
july1_50k = [july1_1_50_sqer, july1_5_50_sqer, july1_15_50_sqer, july1_30_50_sqer, july1_60_50_sqer]
july1_100k = [july1_1_100_sqer, july1_5_100_sqer, july1_15_100_sqer, july1_30_100_sqer, july1_60_100_sqer]
july1_200k = [july1_1_200_sqer, july1_5_200_sqer, july1_15_200_sqer, july1_30_200_sqer, july1_60_200_sqer]

seconds_list = ['1', '5', '15', '30', '60']

july1df.iloc[:,0] = july1_10k
july1df.iloc[:,1] = july1_50k
july1df.iloc[:,2] = july1_100k
july1df.iloc[:,3] = july1_200k

july5df.iloc[:,0] = july5_10k
july5df.iloc[:,1] = july5_50k
july5df.iloc[:,2] = july5_100k
july5df.iloc[:,3] = july5_200k

july1df.index = seconds_list
july5df.index = seconds_list


july1_1_10_change = change(july1, 1, 'vol_mid_10')
july5_1_10_change = change(july5, 1, 'vol_mid_10')

july_mid_change = change(july1, 1, 'Mid Prices')

july1_1_10_rsq = r_squared(july1, 1, 'vol_mid_10')
july5_1_10_rsd = r_squared(july5, 1, 'vol_mid_10')

def squared_error1(df, seconds_window):
    
    vol_mids = df.loc[:,'Mid Prices']
    vol_mids = vol_mids[:-seconds_window]
    future_mids = df.loc[:,'Mid Prices']
    
    vol_mids = np.asarray(vol_mids)
    future_mids = np.asarray(future_mids)
    future_mids = future_mids[seconds_window:]

    
    sqdif = (vol_mids - future_mids)**2
    
    squared_error = sum(sqdif)
    
    return squared_error

mids_sqer = squared_error1(july1, 1)
    

    
    
    