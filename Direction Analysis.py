# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:58:47 2022

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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

fulldata = pd.read_csv('FULL_DATASET.csv')
fulldata.columns = ['Indx', 'Timestamp', 'Mid Prices', 'vol_mid_10', 'vol_mid_40', 'vol_mid_50', 'vol_mid_60', 'vol_mid_100']
fulldata = fulldata.sort_values(by=['Timestamp'])
fulldata = fulldata.set_index(fulldata.loc[:,'Timestamp'])
fulldata = fulldata.iloc[:,2:]
fulldata = fulldata.loc['2020-08-01 00:00:00':,:]

vol_mid_data = pd.DataFrame(fulldata.iloc[:,0])
vol_mid_data['VAMP'] = fulldata.loc[:,'vol_mid_60']

trades = pd.read_csv('trade_imbalance.csv', index_col=0)

quotes = pd.read_csv('Quote Imbalance.csv', index_col=0)

combined = pd.read_csv('combined_model.csv', index_col=1)
combined = combined.iloc[:,1:]
combined.columns = ['Mid Prices', 'Combined Model']

definitions = list(fulldata.columns[1:])
vol_definition = vol_mid_data.columns[1:][0]
trades_definition = trades.columns[1:][0]
quotes_definition = quotes.columns[1:][0]
combined_definition = combined.columns[1:][0]


windows = list(range(1,61))


def binary_accuracy(df, windows, definition):
    a_score = []
    for i in range(len(windows)):
        x = df.loc[:,definition] - df.loc[:,'Mid Prices']
        x = x[:-windows[i]]
        y = df.loc[:,'Mid Prices'].diff(periods=windows[i]).dropna()
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        x = x/np.abs(x)
        y = y/np.abs(y)
        
        x = pd.DataFrame(x)
        x['1'] = y
        x = x.dropna()

        x_new = np.asarray(x.iloc[:,0])
        y_new = np.asarray(x.iloc[:,1])
            
        score = accuracy_score(y_new, x_new)
        a_score.append(score)
        
    return a_score
    

def plot_accuracy(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('Binary Accuracy')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()
    
scores = pd.DataFrame(index=windows, columns=definitions)
for i in definitions:
    scores[i] = binary_accuracy(fulldata, windows, i)
    
plot_accuracy(scores)

scores = pd.DataFrame(index=windows)
scores[vol_definition] = binary_accuracy(vol_mid_data, windows, vol_definition)
scores[trades_definition] = binary_accuracy(trades, windows, trades_definition)
scores[quotes_definition] = binary_accuracy(quotes, windows, quotes_definition)
scores[combined_definition] = binary_accuracy(combined, windows, combined_definition)

plot_accuracy(scores)

def binary_recall(df, windows, definition):
    a_score = []
    for i in range(len(windows)):
        x = df.loc[:,definition] - df.loc[:,'Mid Prices']
        x = x[:-windows[i]]
        y = df.loc[:,'Mid Prices'].diff(periods=windows[i]).dropna()
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        x = x/np.abs(x)
        y = y/np.abs(y)
        
        x = pd.DataFrame(x)
        x['1'] = y
        x = x.dropna()

        x_new = np.asarray(x.iloc[:,0])
        y_new = np.asarray(x.iloc[:,1])
            
        score = recall_score(y_new, x_new)
        a_score.append(score)
        
    return a_score
    

def plot_recall(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('Binary Recall Scores')
    plt.xlabel('Seconds Window')
    plt.ylabel('Score')
    plt.show()
    
scores = pd.DataFrame(index=windows, columns=definitions)
for i in definitions:
    scores[i] = binary_recall(fulldata, windows, i)
    
plot_recall(scores)

scores = pd.DataFrame(index=windows)
scores[vol_definition] = binary_recall(vol_mid_data, windows, vol_definition)
scores[trades_definition] = binary_recall(trades, windows, trades_definition)
scores[quotes_definition] = binary_recall(quotes, windows, quotes_definition)
scores[combined_definition] = binary_recall(combined, windows, combined_definition)


plot_recall(scores)

def binary_precision(df, windows, definition):
    a_score = []
    for i in range(len(windows)):
        x = df.loc[:,definition] - df.loc[:,'Mid Prices']
        x = x[:-windows[i]]
        y = df.loc[:,'Mid Prices'].diff(periods=windows[i]).dropna()
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        x = x/np.abs(x)
        y = y/np.abs(y)
        
        x = pd.DataFrame(x)
        x['1'] = y
        x = x.dropna()

        x_new = np.asarray(x.iloc[:,0])
        y_new = np.asarray(x.iloc[:,1])
            
        score = precision_score(y_new, x_new)
        a_score.append(score)
        
    return a_score
    

def plot_precision(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('Binary Precision Scores')
    plt.xlabel('Seconds Window')
    plt.ylabel('Score')
    plt.show()
    
scores = pd.DataFrame(index=windows, columns=definitions)
for i in definitions:
    scores[i] = binary_precision(fulldata, windows, i)
    
plot_precision(scores)

scores = pd.DataFrame(index=windows)
scores[vol_definition] = binary_precision(vol_mid_data, windows, vol_definition)
scores[trades_definition] = binary_precision(trades, windows, trades_definition)
scores[quotes_definition] = binary_precision(quotes, windows, quotes_definition)
scores[combined_definition] = binary_precision(combined, windows, combined_definition)

plot_precision(scores)

def binary_f1(df, windows, definition):
    a_score = []
    for i in range(len(windows)):
        x = df.loc[:,definition] - df.loc[:,'Mid Prices']
        x = x[:-windows[i]]
        y = df.loc[:,'Mid Prices'].diff(periods=windows[i]).dropna()
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        x = x/np.abs(x)
        y = y/np.abs(y)
        
        x = pd.DataFrame(x)
        x['1'] = y
        x = x.dropna()

        x_new = np.asarray(x.iloc[:,0])
        y_new = np.asarray(x.iloc[:,1])
            
        score = f1_score(y_new, x_new)
        a_score.append(score)
        
    return a_score
    

def plot_f1(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('Binary F1 Scores')
    plt.xlabel('Seconds Window')
    plt.ylabel('Score')
    plt.show()
    
scores = pd.DataFrame(index=windows, columns=definitions)
for i in definitions:
    scores[i] = binary_f1(fulldata, windows, i)
    
plot_f1(scores)

scores = pd.DataFrame(index=windows)
scores[vol_definition] = binary_f1(vol_mid_data, windows, vol_definition)
scores[trades_definition] = binary_f1(trades, windows, trades_definition)
scores[quotes_definition] = binary_f1(quotes, windows, quotes_definition)
scores[combined_definition] = binary_f1(combined, windows, combined_definition)

plot_f1(scores)

#scatter plots to show vol mid is best

trade_imbalance = pd.read_csv('TI_midprice.csv', index_col=0)
trade_imbalance = trade_imbalance.iloc[:,[0,2]]
trade_imbalance = trade_imbalance[trade_imbalance.trade_imbalance != 0]
trade_imbalance = trade_imbalance.loc['2020-08-01 00:00:00':,:]

quote_imbalance = pd.read_csv('aug_sep_full_data.csv')
quote_imbalance = quote_imbalance.iloc[:,6:]


np.mean(np.asarray(quotes['Mid Prices'].diff(periods=60).dropna()))
np.std(np.asarray(quotes['Mid Prices'].diff(periods=60).dropna()))

plt.hist(quotes['Mid Prices'].diff(periods=1).dropna(), bins=5000)
plt.title('Distribution of Price Changes at 1 Second')
plt.xlim(-2,2)
plt.show()

plt.hist((vol_mid_data['Mid Prices'] - vol_mid_data['VAMP']), bins=1000)
plt.title('Distribution of Difference Between Mid Price and Vamp')
plt.xlim(-15,15)
plt.show()

plt.hist((quotes['Mid Prices'] - quotes['Quote Imbalance']), bins=1000)
plt.title('Distribution of Difference Between Mid Price and Quote Imbalance')
plt.xlim(-2,2)
plt.show()

plt.hist((trades['Mid Prices'] - trades['Trade Imbalance']), bins=1000)
plt.title('Distribution of Difference Between Mid Price and Trade Imbalance')
plt.xlim(-20,20)
plt.show()


vol_mid_deciles = pd.qcut((vol_mid_data['VAMP'] - vol_mid_data['Mid Prices'])[:-60], 10,labels = False)
vol_mid_deciles = pd.DataFrame(vol_mid_deciles, columns=['Decile'])
vol_mid_deciles['VAMP - Mid'] = (vol_mid_data['VAMP'] - vol_mid_data['Mid Prices'])[:-60]
vol_mid_deciles['Price Difference'] = list(vol_mid_data['Mid Prices'].diff(periods=60).dropna())
vol_mid_deciles['Price Changes Deciles'] = pd.qcut(vol_mid_deciles['Price Difference'], 10,labels = False)


vol_mid_means = vol_mid_deciles.groupby('Decile').mean()

trade_deciles = pd.qcut(trade_imbalance['trade_imbalance'][:-60], 10,labels = False).dropna()
trade_deciles = pd.DataFrame(trade_deciles)
trade_deciles.columns = ['Decile']
trade_deciles['Trade Imbalance'] = trade_imbalance['trade_imbalance'][:-60]
trade_deciles['Price Difference'] = list(trade_imbalance['Mid Prices'].diff(periods=60).dropna())
trade_deciles['Price Changes Deciles'] = pd.qcut(trade_deciles['Price Difference'], 10,labels = False)


trade_means = trade_deciles.groupby('Decile').mean()

quote_deciles = pd.qcut(quote_imbalance['order_imbalance5'][:-60], 10,labels = False).dropna()
quote_deciles = pd.DataFrame(quote_deciles)
quote_deciles.columns = ['Decile']
quote_deciles['Order Imbalance'] = quote_imbalance['order_imbalance5'][:-60]
quote_deciles['Price Difference'] = list(quote_imbalance['mid_price'].diff(periods=60).dropna())
quote_deciles['Price Changes Deciles'] = pd.qcut(quote_deciles['Price Difference'], 10,labels = False)


quote_means = quote_deciles.groupby('Decile').mean()

vol = plt.scatter(list(vol_mid_means.index), vol_mid_means['Price Difference'], marker='o')
trd = plt.scatter(list(trade_means.index), trade_means['Price Difference'], marker='x')
quo = plt.scatter(list(quote_means.index), quote_means['Price Difference'], marker='^')
plt.legend((vol, trd, quo),('VAMP - Mid Price', 'Trade Imbalance', 'Quote Imbalance'))
plt.title('60 Sec Change in Mid price vs Signals')
plt.xlabel('Signal')
plt.ylabel('60 Sec Change in Mid Price')
plt.show()

quote_deciles1 = pd.qcut(quote_imbalance['order_imbalance1'][:-60], 10,labels = False).dropna()
quote_deciles1 = pd.DataFrame(quote_deciles1)
quote_deciles1.columns = ['Decile']
quote_deciles1['Order Imbalance'] = quote_imbalance['order_imbalance1'][:-60]
quote_deciles1['Price Difference'] = list(quote_imbalance['mid_price'].diff(periods=60).dropna())
quote_deciles1['Price Changes Deciles'] = pd.qcut(quote_deciles1['Price Difference'], 10,labels = False)


quote_means1 = quote_deciles1.groupby('Decile').mean()

quote_deciles2 = pd.qcut(quote_imbalance['order_imbalance2'][:-60], 10,labels = False).dropna()
quote_deciles2 = pd.DataFrame(quote_deciles2)
quote_deciles2.columns = ['Decile']
quote_deciles2['Order Imbalance'] = quote_imbalance['order_imbalance2'][:-60]
quote_deciles2['Price Difference'] = list(quote_imbalance['mid_price'].diff(periods=60).dropna())
quote_deciles2['Price Changes Deciles'] = pd.qcut(quote_deciles2['Price Difference'], 10,labels = False)


quote_means2 = quote_deciles2.groupby('Decile').mean()

quote_deciles3 = pd.qcut(quote_imbalance['order_imbalance3'][:-60], 10,labels = False).dropna()
quote_deciles3 = pd.DataFrame(quote_deciles3)
quote_deciles3.columns = ['Decile']
quote_deciles3['Order Imbalance'] = quote_imbalance['order_imbalance3'][:-60]
quote_deciles3['Price Difference'] = list(quote_imbalance['mid_price'].diff(periods=60).dropna())
quote_deciles3['Price Changes Deciles'] = pd.qcut(quote_deciles2['Price Difference'], 10,labels = False)


quote_means3 = quote_deciles3.groupby('Decile').mean()

quote_deciles4 = pd.qcut(quote_imbalance['order_imbalance4'][:-60], 10,labels = False).dropna()
quote_deciles4 = pd.DataFrame(quote_deciles4)
quote_deciles4.columns = ['Decile']
quote_deciles4['Order Imbalance'] = quote_imbalance['order_imbalance4'][:-60]
quote_deciles4['Price Difference'] = list(quote_imbalance['mid_price'].diff(periods=60).dropna())
quote_deciles4['Price Changes Deciles'] = pd.qcut(quote_deciles2['Price Difference'], 10,labels = False)


quote_means4 = quote_deciles4.groupby('Decile').mean()

quote_deciles5 = pd.qcut(quote_imbalance['order_imbalance5'][:-60], 10,labels = False).dropna()
quote_deciles5 = pd.DataFrame(quote_deciles5)
quote_deciles5.columns = ['Decile']
quote_deciles5['Order Imbalance'] = quote_imbalance['order_imbalance5'][:-60]
quote_deciles5['Price Difference'] = list(quote_imbalance['mid_price'].diff(periods=60).dropna())
quote_deciles5['Price Changes Deciles'] = pd.qcut(quote_deciles2['Price Difference'], 10,labels = False)


quote_means5 = quote_deciles5.groupby('Decile').mean()


quo1 = plt.scatter(quote_means1['Order Imbalance'], quote_means1['Price Difference'])
quo2 = plt.scatter(quote_means2['Order Imbalance'], quote_means2['Price Difference'])
quo3 = plt.scatter(quote_means3['Order Imbalance'], quote_means3['Price Difference'])
quo4 = plt.scatter(quote_means4['Order Imbalance'], quote_means4['Price Difference'])
quo5 = plt.scatter(quote_means5['Order Imbalance'], quote_means5['Price Difference'])
plt.legend((quo1, quo2, quo3, quo4, quo5),('Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'))
plt.title('60 Sec Change in Mid price vs Different Levels of Quote Imbalance')
plt.xlabel('Signal')
plt.ylabel('60 Sec Change in Mid Price')
plt.show()

