# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:14:22 2022

@author: payma
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('August_20.csv', index_col=0)

vol_mids = df.loc[:,'Volume Mid Prices']

mids = df.loc[:,'Mid Prices']

mid_dif = mids - vol_mids

#change periods to get rolling window
returns = mids.pct_change(periods=500)
returns = returns.dropna()

mid_dif_5 = mid_dif[:-5]
mid_dif_10 = mid_dif[:-10]
mid_dif_50 = mid_dif[:-50]
mid_dif_100 = mid_dif[:-100]
mid_dif_200 = mid_dif[:-200]
mid_dif_500 = mid_dif[:-500]
mid_dif_1 = mid_dif[:-1]

plt.scatter(mid_dif_500, returns)
plt.title('7/20 500 Quotes')
plt.xlabel('Mid price - vol mid price')
plt.ylabel('Rolling 500 quote returns')
plt.show()

plt.plot(mids)