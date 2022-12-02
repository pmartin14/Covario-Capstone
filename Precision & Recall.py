# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:32:48 2022

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


fulldata = pd.read_csv('FULL_DATASET.csv')
fulldata.columns = ['Indx', 'Timestamp', 'Mid Prices', 'vol_mid_10', 'vol_mid_40', 'vol_mid_50', 'vol_mid_60', 'vol_mid_100']
fulldata = fulldata.sort_values(by=['Timestamp'])
fulldata = fulldata.set_index(fulldata.loc[:,'Timestamp'])
fulldata = fulldata.iloc[:,2:]
fulldata = fulldata.loc['2020-08-01 00:00:00':,:]

vol_mid_data = pd.DataFrame(fulldata.iloc[:,0])
vol_mid_data['vol_mid_60'] = fulldata.loc[:,'vol_mid_60']

vol_mid_data.to_csv(r'C:\Users\payma\OneDrive\Documents\PAYTON\Cornell MFE\Covario\Data Analysis\CSVs\Volume Mid Data.csv')

trades = pd.read_csv('trade_imbalance.csv', index_col=0)

quotes = pd.read_csv('Quote Imbalance.csv', index_col=0)

definitions = list(fulldata.columns[1:])
vol_60_def = vol_mid_data.columns[1:][0]
trades_definition = trades.columns[1:][0]
quotes_definition = quotes.columns[1:][0]

def get_thresholds(df, windows):
    prices = df.loc[:,'Mid Prices']
    thresholds = []
    for i in windows:
        dif = prices.diff(periods=i).dropna()
        thresholds.append(dif.std())
        
    return thresholds
    
windows = list(range(1,61))
thresholds = get_thresholds(vol_mid_data, windows)
trades_thresholds = get_thresholds(trades, windows) 

def precision_per_class(df, thresholds, windows, definition, classes):
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
        
        score = precision_score(y, x, labels= classes, average=None)
        a_score.append(score)
        
    return a_score

classes = [-1, 0, 1]
vol_p_score = precision_per_class(vol_mid_data, thresholds, windows, vol_60_def, classes)
trades_p_score = precision_per_class(trades, thresholds, windows, trades_definition, classes)
quotes_p_score = precision_per_class(quotes, thresholds, windows, quotes_definition, classes)
classneg1_precisions = pd.DataFrame(index=windows)
class0_precisions = pd.DataFrame(index=windows)
class1_precisions = pd.DataFrame(index=windows)

vol_neg1 = [item[0] for item in vol_p_score]
vol_0 = [item[1] for item in vol_p_score]
vol_1 = [item[2] for item in vol_p_score]

trades_neg1 = [item[0] for item in trades_p_score]
trades_0 = [item[1] for item in trades_p_score]
trades_1 = [item[2] for item in trades_p_score]

quotes_neg1 = [item[0] for item in quotes_p_score]
quotes_0 = [item[1] for item in quotes_p_score]
quotes_1 = [item[2] for item in quotes_p_score]

classneg1_precisions['vol_mid_60'] = vol_neg1
classneg1_precisions['Trade Imbalance'] = trades_neg1

class0_precisions['vol_mid_60'] = vol_0
class0_precisions['Trade Imbalance'] = trades_0
class0_precisions['Quote Imbalance'] = quotes_0

class1_precisions['vol_mid_60'] = vol_1
class1_precisions['Trade Imbalance'] = trades_1


def plot_precision(scores, clas):
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('Precision Class: ' + clas)
    plt.xlabel('Seconds Window')
    plt.ylabel('Score')
    plt.show()
    
plot_precision(classneg1_precisions, '-1')
plot_precision(class0_precisions, '0')
plot_precision(class1_precisions, '1')


#recall per class is the exact same as accuracy per class
def recall_per_class(df, thresholds, windows, definition, classes):
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
        
        score = recall_score(y, x, labels= classes, average=None)
        a_score.append(score)
        
    return a_score

classes = [-1, 0, 1]
vol_r_score = recall_per_class(vol_mid_data, thresholds, windows, vol_60_def, classes)
trades_r_score = recall_per_class(trades, thresholds, windows, trades_definition, classes)
quotes_r_score = recall_per_class(quotes, thresholds, windows, quotes_definition, classes)
classneg1_recall = pd.DataFrame(index=windows)
class0_recall = pd.DataFrame(index=windows)
class1_recall = pd.DataFrame(index=windows)

vol_neg1 = [item[0] for item in vol_r_score]
vol_0 = [item[1] for item in vol_r_score]
vol_1 = [item[2] for item in vol_r_score]

trades_neg1 = [item[0] for item in trades_r_score]
trades_0 = [item[1] for item in trades_r_score]
trades_1 = [item[2] for item in trades_r_score]

quotes_neg1 = [item[0] for item in quotes_r_score]
quotes_0 = [item[1] for item in quotes_r_score]
quotes_1 = [item[2] for item in quotes_r_score]

classneg1_recall['vol_mid_60'] = vol_neg1
classneg1_recall['Trade Imbalance'] = trades_neg1

class0_recall['vol_mid_60'] = vol_0
class0_recall['Trade Imbalance'] = trades_0
#class0_recall['Quote Imbalance'] = quotes_0

class1_recall['vol_mid_60'] = vol_1
class1_recall['Trade Imbalance'] = trades_1


def plot_recall(scores, clas):
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('Recall (Accuracy) Class: ' + clas)
    plt.xlabel('Seconds Window')
    plt.ylabel('Score')
    plt.show()
    
plot_recall(classneg1_recall, '-1')
plot_recall(class0_recall, '0')
plot_recall(class1_recall, '1')

def f1_per_class(df, thresholds, windows, definition, classes):
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
        
        score = f1_score(y, x, labels= classes, average=None)
        a_score.append(score)
        
    return a_score

classes = [-1, 0, 1]
vol_f1_score = f1_per_class(vol_mid_data, thresholds, windows, vol_60_def, classes)
trades_f1_score = f1_per_class(trades, thresholds, windows, trades_definition, classes)
quotes_f1_score = f1_per_class(quotes, thresholds, windows, quotes_definition, classes)
classneg1_f1 = pd.DataFrame(index=windows)
class0_f1 = pd.DataFrame(index=windows)
class1_f1 = pd.DataFrame(index=windows)

vol_neg1 = [item[0] for item in vol_f1_score]
vol_0 = [item[1] for item in vol_f1_score]
vol_1 = [item[2] for item in vol_f1_score]

trades_neg1 = [item[0] for item in trades_f1_score]
trades_0 = [item[1] for item in trades_f1_score]
trades_1 = [item[2] for item in trades_f1_score]

quotes_neg1 = [item[0] for item in quotes_f1_score]
quotes_0 = [item[1] for item in quotes_f1_score]
quotes_1 = [item[2] for item in quotes_f1_score]

classneg1_f1['vol_mid_60'] = vol_neg1
classneg1_f1['Trade Imbalance'] = trades_neg1

class0_f1['vol_mid_60'] = vol_0
class0_f1['Trade Imbalance'] = trades_0
#class0_f1['Quote Imbalance'] = quotes_0

class1_f1['vol_mid_60'] = vol_1
class1_f1['Trade Imbalance'] = trades_1


def plot_f1(scores, clas):
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('F1 Class: ' + clas)
    plt.xlabel('Seconds Window')
    plt.ylabel('Score')
    plt.show()
    
plot_f1(classneg1_f1, '-1')
plot_f1(class0_f1, '0')
plot_f1(class1_f1, '1')
