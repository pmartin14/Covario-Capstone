# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:25:40 2022

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
from sklearn.metrics import confusion_matrix
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

combined = pd.read_csv('quotes_imbalance.csv', index_col=0)

def price_change_plot(df, windows):
    prices = df.loc[:,'Mid Prices']
    changes = []
    for i in windows:
        dif = prices.diff(periods=i).dropna().abs()
        changes.append(dif.mean())
        
    plt.plot(changes, label=['Mean Price Change'])
    plt.legend()
    plt.title('Mean Price Change ')
    plt.xlabel('Seconds Window')
    plt.ylabel('Change')
    plt.show()

    
windows = list(range(1,61))
price_change_plot(fulldata, windows)

def get_thresholds(df, windows):
    prices = df.loc[:,'Mid Prices']
    thresholds = []
    for i in windows:
        dif = prices.diff(periods=i).dropna()
        thresholds.append(dif.std())
        
    return thresholds
    
windows = list(range(1,61))
thresholds = get_thresholds(fulldata, windows) 
plt.plot(thresholds)
plt.title('Standard Deviation of Price Movements')
plt.xlabel('Seconds Window')
plt.ylabel('Standard Deviation')
plt.show()

trade_thresholds = get_thresholds(trades, windows)
quotes_thresholds = get_thresholds(quotes, windows) 
combined_thresholds = get_thresholds(combined, windows) 


def classification_matrix(df, threshold, window, definition):
    x = df.loc[:,definition] - df.loc[:,'Mid Prices']
    x = x[:-window]
    y = df.loc[:,'Mid Prices'].diff(periods=window).dropna()
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    x = np.where(np.abs(x) < np.abs(threshold), 0, x)
    y = np.where(np.abs(y) < np.abs(threshold), 0, y)

    x = np.where(x < 0, -1, x)
    y = np.where(y < 0, -1, y)

    x = np.where(x > 0, 1, x)
    y = np.where(y > 0, 1, y)
    
    c_matrix = confusion_matrix(y,x)
    
    c_matrix = pd.DataFrame(c_matrix, index=['-1', '0', '1'], columns=['-1', '0', '1'])
    
    return c_matrix, len(x)

c_matrix, length = classification_matrix(vol_mid_data, thresholds[9], 10, 'vol_mid_60')
c_matrix, length = classification_matrix(fulldata, thresholds[59], 60, 'vol_mid_10')


def f_1_micro(df, thresholds, windows, definition):
    a_score = []
    for i in range(len(windows)):
        x = df.loc[:,definition] - df.loc[:,'Mid Prices']
        x = x[:-windows[i]]
        y = df.loc[:,'Mid Prices'].diff(periods=windows[i]).dropna()
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        x = np.where(np.abs(x) < np.abs(thresholds[i]), 0, x)
        y = np.where(np.abs(y) < np.abs(thresholds[i]), 0, y)
    
        x = np.where(x < 0, -1, x)
        y = np.where(y < 0, -1, y)
    
        x = np.where(x > 0, 1, x)
        y = np.where(y > 0, 1, y)
        
        score = f1_score(y, x, average='micro')
        a_score.append(score)
        
    return a_score

f1_micro_scores = f_1_micro(vol_mid_data, thresholds, windows, 'vol_mid_60')

f1_micro_scores.plot()

def f_1_macro(df, thresholds, windows, definition):
    a_score = []
    for i in range(len(windows)):
        x = df.loc[:,definition] - df.loc[:,'Mid Prices']
        x = x[:-windows[i]]
        y = df.loc[:,'Mid Prices'].diff(periods=windows[i]).dropna()
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        x = np.where(np.abs(x) < np.abs(thresholds[i]), 0, x)
        y = np.where(np.abs(y) < np.abs(thresholds[i]), 0, y)
    
        x = np.where(x < 0, -1, x)
        y = np.where(y < 0, -1, y)
    
        x = np.where(x > 0, 1, x)
        y = np.where(y > 0, 1, y)
        
        score = f1_score(y, x, average='macro')
        a_score.append(score)
        
    return a_score

f1_macro_scores = f_1_micro(vol_mid_data, thresholds, windows, 'vol_mid_60')

f1_macro_scores.plot()

def f_1_weighted(df, thresholds, windows, definition):
    a_score = []
    for i in range(len(windows)):
        x = df.loc[:,definition] - df.loc[:,'Mid Prices']
        x = x[:-windows[i]]
        y = df.loc[:,'Mid Prices'].diff(periods=windows[i]).dropna()
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        x = np.where(np.abs(x) < np.abs(thresholds[i]), 0, x)
        y = np.where(np.abs(y) < np.abs(thresholds[i]), 0, y)
    
        x = np.where(x < 0, -1, x)
        y = np.where(y < 0, -1, y)
    
        x = np.where(x > 0, 1, x)
        y = np.where(y > 0, 1, y)
        
        score = f1_score(y, x, average='weighted')
        a_score.append(score)
        
    return a_score

windows = list(range(1,61))
thresholds = get_thresholds(fulldata, windows)
definitions = list(fulldata.columns[1:])
trades_definitions = trades.columns[1:][0]
quotes_definitions = list(quotes.columns[1:])
combined_definitions = combined.columns[1:][0]

def plot_f1_micro(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('F1 Micro (Accuracy)')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()
    
scores = pd.DataFrame(index=windows, columns=definitions)
for i in definitions:
    scores[i] = f_1_micro(fulldata, thresholds, windows, i)

plot_f1_micro(scores)

def plot_f1_macro(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('F1 Macro (Average of F1 Across Classes)')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()
    
macro_scores = pd.DataFrame(index=windows, columns=definitions)
for i in definitions:
    macro_scores[i] = f_1_macro(fulldata, thresholds, windows, i)

plot_f1_macro(macro_scores)

def plot_f1_weighted(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('F1 Weighted')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()

#Look at scores across definitions

micro_scores = pd.DataFrame(index=windows, columns=definitions)
micro_scores['vol_mid_60'] = f_1_micro(vol_mid_data, thresholds, windows, 'vol_mid_60')
micro_scores['Trade Imbalance'] = f_1_micro(trades, thresholds, windows, trades_definitions)
for i in quotes_definitions:
    micro_scores[i] = f_1_micro(quotes, thresholds, windows, i)
micro_scores = micro_scores.dropna(axis=1)

plot_f1_micro(micro_scores)

macro_scores = pd.DataFrame(index=windows)
macro_scores['vol_mid_60'] = f_1_macro(vol_mid_data, thresholds, windows, 'vol_mid_60')
macro_scores['Trade Imbalance'] = f_1_macro(trades, thresholds, windows, trades_definitions)
for i in quotes_definitions:
    macro_scores[i] = f_1_macro(quotes, thresholds, windows, i)

plot_f1_macro(macro_scores)

weighted_scores = pd.DataFrame(index=windows)
weighted_scores['vol_mid_60'] = f_1_weighted(vol_mid_data, thresholds, windows, 'vol_mid_60')
weighted_scores['Trade Imbalance'] = f_1_weighted(trades, thresholds, windows, trades_definitions)
for i in quotes_definitions:
    weighted_scores[i] = f_1_weighted(quotes, thresholds, windows, i)

plot_f1_weighted(weighted_scores)


