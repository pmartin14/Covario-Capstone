# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:54:01 2022

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

from sklearn.metrics import f1_score

fulldata = pd.read_csv('FULL_DATASET.csv')
fulldata.columns = ['Indx', 'Timestamp', 'Mid Prices', 'vol_mid_10', 'vol_mid_40', 'vol_mid_50', 'vol_mid_60', 'vol_mid_100']
fulldata = fulldata.sort_values(by=['Timestamp'])
fulldata = fulldata.set_index(fulldata.loc[:,'Timestamp'])
fulldata = fulldata.iloc[:,2:]
fulldata = fulldata.loc['2020-08-01 00:00:00':,:]

vol_mid_data = pd.DataFrame(fulldata.iloc[:,0])
vol_mid_data['vol_mid_60'] = fulldata.loc[:,'vol_mid_60']

trades = pd.read_csv('trade_imbalance.csv', index_col=0)

quotes = pd.read_csv('quotes_imbalance.csv', index_col=0)


def F1_score(df, threshold, window):
    x = df.loc[:,threshold] - df.loc[:,'Mid Prices']
    x = x[:-window]
    y = df.loc[:,'Mid Prices'].diff(periods=window).dropna()
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    x = x/np.abs(x)
    y = y/np.abs(y)
    
    x = pd.DataFrame(x)
    x['1'] = y
    x = x.dropna()

    x_new = np.asarray(x.iloc[:,0])
    y_new = np.asarray(x.iloc[:,1])
        
    f1score = f1_score(y_new, x_new)
    
    return f1score

def F1_upward_direction_score(df, threshold, window):
    x = df.loc[:,threshold] - df.loc[:,'Mid Prices']
    x = x[:-window]
    y = df.loc[:,'Mid Prices'].diff(periods=window).dropna()
    
    x = np.asarray(x)
    y = np.asarray(y)

    x = x/np.abs(x)
    y = y/np.abs(y)

    x = pd.DataFrame(x)
    x['1'] = y
    x = x.dropna()
    x = x.reset_index()
    drop_list = [i for i,v in enumerate(np.asarray(x.iloc[:,2])) if v < 0]
    x = x.drop(labels=drop_list, axis=0)

    x_new = np.asarray(x.iloc[:,1])
    y_new = np.asarray(x.iloc[:,2])

    f1score = f1_score(y_new, x_new)
    
    return f1score
    
def F1_downward_direction_score(df, threshold, window):
    x = df.loc[:,threshold] - df.loc[:,'Mid Prices']
    x = x[:-window]
    y = df.loc[:,'Mid Prices'].diff(periods=window).dropna()
    
    x = np.asarray(x)
    y = np.asarray(y)

    x = x/np.abs(x)
    y = y/np.abs(y)

    x = pd.DataFrame(x)
    x['1'] = y
    x = x.dropna()
    x = x.reset_index()
    drop_list = [i for i,v in enumerate(np.asarray(x.iloc[:,2])) if v > 0]
    x = x.drop(labels=drop_list, axis=0)

    x_new = -np.asarray(x.iloc[:,1])
    y_new = -np.asarray(x.iloc[:,2])

    f1score = f1_score(y_new, x_new)
    
    return f1score
    
    
def plot_f1_accuracy(df, thresholds, windows, plot=False):
    scores = np.zeros((len(windows), len(thresholds)))
    scores = pd.DataFrame(scores, index=windows, columns=thresholds)
    for i in thresholds:
        for j in windows:
            f1score= F1_score(df, i, j)
            scores.loc[j,i] = f1score
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('F1 Score ')
    plt.xlabel('Seconds Window')
    plt.ylabel('F1 Score')
    plt.show()   
    
def plot_F1_updown_accuracy(df, thresholds, windows, plot=False):
    up_scores = np.zeros((len(windows), len(thresholds)))
    down_scores = np.zeros((len(windows), len(thresholds)))
    up_scores = pd.DataFrame(up_scores, index=windows, columns=thresholds)
    down_scores = pd.DataFrame(down_scores, index=windows, columns=thresholds)
    for i in thresholds:
        for j in windows:
            up_score= F1_upward_direction_score(df, i, j)
            down_score = F1_downward_direction_score(df, i, j)
            up_scores.loc[j,i] = up_score
            down_scores.loc[j,i] = down_score

        
    plt.plot(up_scores, label=up_scores.columns)
    plt.legend()
    plt.title('Upward Direction F1 Score ' + df.index[0].split(' ')[0])
    plt.xlabel('Seconds Window')
    plt.ylabel('F1 Score')
    plt.show()
    
    plt.plot(down_scores, label=down_scores.columns)
    plt.legend()
    plt.title('Downward Direction F1 Score ' + df.index[0].split(' ')[0])
    plt.xlabel('Seconds Window')
    plt.ylabel('F1 Score')
    plt.show()    
    
    
def plot_f1_accuracy_fulldata(df, thresholds, trades_df, fullquotes_df, quotes_defs, trades_defs, windows, plot=False):
    scores = np.zeros((len(windows), len(thresholds)))
    scores = pd.DataFrame(scores, index=windows, columns=thresholds)
    for i in thresholds:
        for j in windows:
            f1score= F1_score(df, i, j)
            scores.loc[j,i] = f1score
            
    quotes_scores = np.zeros((len(windows), len(quotes_defs)))
    quotes_scores = pd.DataFrame(quotes_scores, index=windows, columns=quotes_defs)
    for i in quotes_defs:
        for j in windows:
            quotes_f1_score = F1_score(fullquotes_df, i, j)
            quotes_scores.loc[j,i] = quotes_f1_score
            
    trade_scores = np.zeros((len(windows), len(trades_defs)))
    trade_scores = pd.DataFrame(trade_scores, index=windows, columns=trades_defs)
    for i in trades_defs:
        for j in windows:
            trade_f1_score = F1_score(trades_df, i, j)
            trade_scores.loc[j,i] = trade_f1_score
        
    plt.plot(scores, label=scores.columns[0])
    plt.plot(trade_scores, label=trade_scores.columns[0])
    plt.plot(quotes_scores, label=quotes_scores.columns[0:])

    plt.legend()
    plt.title('F1 Score')
    plt.xlabel('Seconds Window')
    plt.ylabel('F1 Score')
    plt.show()   
    
def plot_F1_updown_accuracy_fulldata(df, thresholds, windows, plot=False):
    up_scores = np.zeros((len(windows), len(thresholds)))
    down_scores = np.zeros((len(windows), len(thresholds)))
    up_scores = pd.DataFrame(up_scores, index=windows, columns=thresholds)
    down_scores = pd.DataFrame(down_scores, index=windows, columns=thresholds)
    for i in thresholds:
        for j in windows:
            up_score= F1_upward_direction_score(df, i, j)
            down_score = F1_downward_direction_score(df, i, j)
            up_scores.loc[j,i] = up_score
            down_scores.loc[j,i] = down_score

        
    plt.plot(up_scores, label=up_scores.columns)
    plt.legend()
    plt.title('Upward Direction F1 Score')
    plt.xlabel('Seconds Window')
    plt.ylabel('F1 Score')
    plt.show()
    
    plt.plot(down_scores, label=down_scores.columns)
    plt.legend()
    plt.title('Downward Direction F1 Score')
    plt.xlabel('Seconds Window')
    plt.ylabel('F1 Score')
    plt.show()  


windows = list(range(1,60))
thresholds = fulldata.columns[1:]
plot_f1_accuracy_fulldata(fulldata, thresholds, windows)
plot_F1_updown_accuracy_fulldata(fulldata, thresholds, windows)

windows = list(range(1,60))
thresholds = vol_mid_data.columns[1:]
quotes_defs = quotes.columns[1:]
trades_defs = trades.columns[1:]
plot_f1_accuracy_fulldata(vol_mid_data, thresholds, trades, quotes, quotes_defs, trades_defs, windows)

