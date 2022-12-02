# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:33:07 2022

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
from sklearn.metrics import accuracy_score

sec_1 = pd.read_csv('order_im 1s.csv', index_col=0)
sec_1.columns=['Mid Prices', '2_1_sec', '3_1_sec', '4_1_sec', '5_1_sec']
sec_5 = pd.read_csv('order_im 5s.csv', index_col=0)
sec_5.columns=['Mid Prices', '2_5_sec', '3_5_sec', '4_5_sec', '5_5_sec']
sec_15 = pd.read_csv('order_im 15s.csv', index_col=0)
sec_15.columns=['Mid Prices', '2_15_sec', '3_15_sec', '4_15_sec', '5_15_sec']
sec_30 = pd.read_csv('order_im 30s.csv', index_col=0)
sec_30.columns=['Mid Prices', '2_30_sec', '3_30_sec', '4_30_sec', '5_30_sec']
sec_60 = pd.read_csv('order_im 60s.csv', index_col=0)
sec_60.columns=['Mid Prices', '2_60_sec', '3_60_sec', '4_60_sec', '5_60_sec']

def get_thresholds(df, windows):
    prices = df.loc[:,'Mid Prices']
    thresholds = []
    for i in windows:
        dif = prices.diff(periods=i).dropna()
        thresholds.append(dif.std())
        
    return thresholds
    
windows = list(range(1,61))
imbalance_thresholds = get_thresholds(sec_1, windows)

def binary_accuracy(df, thresholds, windows, definition):
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

def plot_accuracy(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('Binary Accuracy')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()

def plot_f1_micro(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('F1 Micro (Accuracy)')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()
    
def plot_f1_macro(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('F1 Macro (Average of F1 Across Classes)')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()
    
def plot_f1_weighted(scores):
        
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('F1 Weighted')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()
    
    
accuracy_scores_1sec = pd.DataFrame(index=windows)
for i in list(sec_1.columns[1:]):
    accuracy_scores_1sec[i] = binary_accuracy(sec_1, imbalance_thresholds, windows, i)
    
accuracy_scores_5sec = pd.DataFrame(index=windows)
for i in list(sec_5.columns[1:]):
    accuracy_scores_5sec[i] = binary_accuracy(sec_5, imbalance_thresholds, windows, i)
    
accuracy_scores_15sec = pd.DataFrame(index=windows)
for i in list(sec_15.columns[1:]):
    accuracy_scores_15sec[i] = binary_accuracy(sec_15, imbalance_thresholds, windows, i)
    
accuracy_scores_30sec = pd.DataFrame(index=windows)
for i in list(sec_30.columns[1:]):
    accuracy_scores_30sec[i] = binary_accuracy(sec_30, imbalance_thresholds, windows, i)
    
accuracy_scores_60sec = pd.DataFrame(index=windows)
for i in list(sec_60.columns[1:]):
    accuracy_scores_60sec[i] = binary_accuracy(sec_60, imbalance_thresholds, windows, i)
    

plot_accuracy(accuracy_scores_1sec)
plot_accuracy(accuracy_scores_5sec)
plot_accuracy(accuracy_scores_15sec)
plot_accuracy(accuracy_scores_30sec)
plot_accuracy(accuracy_scores_60sec)

final_quotes = pd.DataFrame(index=sec_60.index)
final_quotes['Mid Prices'] = sec_60['Mid Prices']
final_quotes['Quote Imbalance'] = sec_60['2_60_sec']
final_quotes.to_csv(r"C:\Users\payma\OneDrive\Documents\PAYTON\Cornell MFE\Covario\Data Analysis\Fair Price CSVs\Quote Imbalance.csv")
    

micro_scores_1sec = pd.DataFrame(index=windows)
for i in list(sec_1.columns[1:]):
    micro_scores_1sec[i] = f_1_micro(sec_1, imbalance_thresholds, windows, i)
    
micro_scores_5sec = pd.DataFrame(index=windows)
for i in list(sec_5.columns[1:]):
    micro_scores_5sec[i] = f_1_micro(sec_5, imbalance_thresholds, windows, i)
    
micro_scores_15sec = pd.DataFrame(index=windows)
for i in list(sec_15.columns[1:]):
    micro_scores_15sec[i] = f_1_micro(sec_15, imbalance_thresholds, windows, i)
    
micro_scores_30sec = pd.DataFrame(index=windows)
for i in list(sec_30.columns[1:]):
    micro_scores_30sec[i] = f_1_micro(sec_30, imbalance_thresholds, windows, i)
    
micro_scores_60sec = pd.DataFrame(index=windows)
for i in list(sec_60.columns[1:]):
    micro_scores_60sec[i] = f_1_micro(sec_60, imbalance_thresholds, windows, i)
    

plot_f1_micro(micro_scores_1sec)
plot_f1_micro(micro_scores_5sec)
plot_f1_micro(micro_scores_15sec)
plot_f1_micro(micro_scores_30sec)
plot_f1_micro(micro_scores_60sec)

def direction_score(df, threshold, window):
    x = df.loc[:,threshold] - df.loc[:,'Mid Prices']
    x = x[:-window]
    y = df.loc[:,'Mid Prices'].diff(periods=window).dropna()
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    x = x/np.abs(x)
    y = y/np.abs(y)
    
    score = x+y
    score = np.abs(score/2)
    score = pd.DataFrame(score).dropna()
    score = np.asarray(score)
    direction_score = np.sum(score)/len(score)
    
    return direction_score, len(score)

def plot_accuracy(df, thresholds, windows, plot=False):
    scores = np.zeros((len(windows), len(thresholds)))
    scores = pd.DataFrame(scores, index=windows, columns=thresholds)
    for i in thresholds:
        for j in windows:
            d_score, change = direction_score(df, i, j)
            scores.loc[j,i] = d_score
        
            
    plt.plot(scores, label=scores.columns)
    plt.legend()
    plt.title('Direction Accuracy ')
    plt.xlabel('Seconds Window')
    plt.ylabel('Accuracy')
    plt.show()

thresholds = list(sec_1.columns[1:])
plot_accuracy(sec_1, thresholds, windows)

thresholds = list(sec_5.columns[1:])
plot_accuracy(sec_5, thresholds, windows)

thresholds = list(sec_15.columns[1:])
plot_accuracy(sec_15, thresholds, windows)

thresholds = list(sec_30.columns[1:])
plot_accuracy(sec_30, thresholds, windows)

thresholds = list(sec_60.columns[1:])
plot_accuracy(sec_60, thresholds, windows)