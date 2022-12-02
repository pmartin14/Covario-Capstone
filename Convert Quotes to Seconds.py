# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:58:27 2022

@author: payma
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

aug_1 = pd.read_csv('August_1.csv')

times = list(aug_1.iloc[:,0])

import datetime
for i in range(len(times)):
    times[i] = datetime.datetime.strptime(times[i], '%Y-%m-%d %H:%M:%S.%f')
    
aug_1.index = times
    
aug_1_new = pd.DataFrame(columns=aug_1.columns)
aug_1_new = aug_1_new.append(aug_1.iloc[0,:])


time_new = datetime.datetime(2020, 8, 1, 0, 0, 0)
new_index = [time_new]
time_new += datetime.timedelta(seconds=1)

for i in range(len(aug_1)):
    time_old = aug_1.index[i]
    if time_old > time_new:
        aug_1_new = aug_1_new.append(aug_1.iloc[i-1,:])
        new_index.append(time_new)
        time_new += datetime.timedelta(seconds=1)
        
aug_1_new.index = new_index
aug_1_new = aug_1_new.drop(aug_1_new.columns[[0]], axis=1)

aug_1_new.to_csv(r'C:\Users\payma\OneDrive\Documents\PAYTON\Cornell MFE\Covario\Data Analysis\CSVs\August_1_Seconds.csv')

vol_mids = aug_1_new.loc[:,'Volume Mid Prices']

mids = aug_1_new.loc[:,'Mid Prices']

mid_dif = mids - vol_mids

returns = mids.pct_change(periods=1)
returns = returns.dropna()

mid_dif_1 = mid_dif[:-1]
mid_dif_5 = mid_dif[:-5]
mid_dif_10 = mid_dif[:-10]
mid_dif_30 = mid_dif[:-30]
mid_dif_60 = mid_dif[:-60]

plt.scatter(mid_dif_10, returns)
plt.title('8/1 60 Seconds')
plt.xlabel('Mid price - vol mid price')
plt.ylabel('Rolling 60 Second returns')
plt.show()

mid_dif_1.hist(bins=100)
mid_dif_1.skew()
mid_dif_10.hist()