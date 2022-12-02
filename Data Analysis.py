# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 12:08:09 2022

@author: payma
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
    

prev_eod1 = pd.read_csv(r'2020-07-31_level2eodsnapshot_bitstamp_btcusd.csv.gz', compression='gzip',
                   error_bad_lines=False)

df1 = pd.read_csv(r'2020-08-01_level2clean_bitstamp_btcusd.csv.gz', compression='gzip',
                   error_bad_lines=False)


from sortedcontainers import SortedDict

class Book:
    def __init__(self):
        self.bids = SortedDict([[prev_eod1['price'][x], prev_eod1['volume'][x]]
                                for x in prev_eod1.loc[(prev_eod1['side'] == 'bid')].index])
        self.asks = SortedDict([[prev_eod1['price'][x], prev_eod1['volume'][x]] 
                                for x in prev_eod1.loc[(prev_eod1['side'] == 'ask')].index])
        
        self.time = ""
        
        self.best_bid = 0
        
        self.best_ask = 0
        
        self.spreads = []
        
        self.VWAP_spreads = []
        
        self.mid_price = []
        
        self.vol_spreads = []
        
        self.vol_mid_price = []
        
        self.best_bid_list = []
        
        self.best_ask_list = []
        
        self.vol_best_bid_list = []
        
        self.vol_best_ask_list = []
        
        self.VWAP_best_bid_list = []
        
        self.VWAP_best_ask_list = []



        
    def bid_ask_quotes(self):
        self.best_bid = self.bids.peekitem(index=-1)[0]
        self.best_ask = self.asks.peekitem(index=0)[0]
        return self.best_bid, self.best_ask
    
    def calc_spread_with_vol(self):
        bid_count = 0 
        ask_count = 0
        bid_vol_total =0
        ask_vol_total = 0
        i_a = 0
        i_b = 1
        
        while ask_count < 100000:
            ask = float(self.asks.peekitem(index = i_a)[0])
            vol = self.asks.peekitem(index = i_a)[1]
            ask_count += ask*vol
            ask_vol_total += vol
            i_a += 1
        vol_ask = ask
        ask_VWAP = ask_count/ask_vol_total
            
        while bid_count < 100000:
            bid = float(self.bids.peekitem(index = -1*i_b)[0])
            vol = self.bids.peekitem(index = -1*i_b)[1]
            bid_count += bid*vol
            bid_vol_total += vol
            i_b += 1
        vol_bid = bid
        bid_VWAP = bid_count/bid_vol_total
            
        spread = ask - bid
        VWAP_spread = ask_VWAP - bid_VWAP
        mid_price = (ask_VWAP + bid_VWAP)/2
        return spread, VWAP_spread, mid_price, vol_ask, vol_bid, ask_VWAP, bid_VWAP
    
    def calc_spread_and_midprice(self):
        spread = self.best_ask - self.best_bid
        mid_price = (self.best_ask + self.best_bid)/2
        return spread, mid_price, self.best_ask, self.best_bid
    
    def update_order_book(self, endtime):
        """
        Parameters:
            endtime: is a datetime variable designating the point in time through which to update the limit order book
        Return: 
            updated_order_book: updated snapshot of the limit order book at time = endtime
        """
        i=0
        time = prev_eod1.iloc[-1][0]
        
        while time < endtime:
            entry = df1.iloc[i]
            time = entry[0]
            if time > endtime:
                break
            price = float(entry[1])
            bid_ask = entry[2]
            vol = entry[3]
            new = {price: vol}

            if bid_ask == "bid":
                if vol == 0:
                    zero_vol = self.bids.pop(price,None)
                else:
                    self.bids.update(new)
            elif bid_ask == "ask":
                if vol == 0:
                    zero_vol = self.asks.pop(price,None)
                else:
                    self.asks.update(new)
            i+=1
            print(i)
            #this next part changes the frequency(every 500 ticks in this case) of when spread and mid 
            #are calulated
            if i%1 == 0:
                self.bid_ask_quotes()
                spread, mid_price, ask, bid = self.calc_spread_and_midprice()
                vol_spread, VWAP_spread, vol_mid_price, vol_ask, vol_bid, VWAP_ask, VWAP_bid = self.calc_spread_with_vol()
                
                self.spreads.append([spread, time])
                self.VWAP_spreads.append([VWAP_spread, time])
                self.mid_price.append([mid_price,time])
                self.vol_spreads.append([vol_spread, time])
                self.vol_mid_price.append([vol_mid_price,time])
                self.best_ask_list.append([ask,time])
                self.best_bid_list.append([bid,time])
                self.vol_best_ask_list.append([vol_ask,time])
                self.vol_best_bid_list.append([vol_bid,time])
                self.VWAP_best_ask_list.append([vol_ask,time])
                self.VWAP_best_bid_list.append([vol_bid,time])
                
            if i >= len(df1):

                break
        print(time)
        return
    
mybook = Book()
bidsbook = mybook.bids

endtime = "2020-08-01T23:59:59.999000Z"

mybook.update_order_book(endtime)


sprds = []
for i in range(len(mybook.spreads)):
    sprds.append(mybook.spreads[i][0])
    
mids = []
for i in range(len(mybook.mid_price)):
    mids.append(mybook.mid_price[i][0])

volsprds = []
times = []
for i in range(len(mybook.vol_spreads)):
    volsprds.append(mybook.vol_spreads[i][0])
    times.append(mybook.vol_spreads[i][1])
    
vol_mids = []
for i in range(len(mybook.vol_mid_price)):
    vol_mids.append(mybook.vol_mid_price[i][0])
    
VWAP_sprds = []
for i in range(len(mybook.VWAP_spreads)):
    VWAP_sprds.append(mybook.VWAP_spreads[i][0])
    
asks = []
for i in range(len(mybook.best_ask_list)):
    asks.append(mybook.best_ask_list[i][0])

bids = []
for i in range(len(mybook.best_bid_list)):
    bids.append(mybook.best_bid_list[i][0])

vol_asks = []
for i in range(len(mybook.vol_best_ask_list)):
    vol_asks.append(mybook.vol_best_ask_list[i][0])

vol_bids = []
for i in range(len(mybook.vol_best_bid_list)):
    vol_bids.append(mybook.vol_best_bid_list[i][0])

VWAP_asks = []
for i in range(len(mybook.VWAP_best_ask_list)):
    VWAP_asks.append(mybook.VWAP_best_ask_list[i][0])

VWAP_bids = []
for i in range(len(mybook.VWAP_best_bid_list)):
    VWAP_bids.append(mybook.VWAP_best_bid_list[i][0])


import datetime
for i in range(len(times)):
    times[i] = datetime.datetime.strptime(times[i], '%Y-%m-%dT%H:%M:%S.%fZ')
    

columns = ['Spreads', 'Mid Prices', 'Volume Spreads', 'Volume Mid Prices', 'VWAP Spreads', 'Best Bids', 'Best Asks', 'Best Volume Bids', 'Best Volume Asks', 'Best VWAP Bids', 'Best VWAP Asks']

aug_1_df = pd.DataFrame(index=times, columns=columns)
aug_1_df.loc[:,'Spreads'] = sprds
aug_1_df.loc[:,'Mid Prices'] = mids
aug_1_df.loc[:,'Volume Spreads'] = volsprds
aug_1_df.loc[:,'Volume Mid Prices'] = vol_mids
aug_1_df.loc[:,'VWAP Spreads'] = VWAP_sprds
aug_1_df.loc[:,'Best Bids'] = bids
aug_1_df.loc[:,'Best Asks'] = asks
aug_1_df.loc[:,'Best Volume Asks'] = vol_asks
aug_1_df.loc[:,'Best Volume Bids'] = vol_bids
aug_1_df.loc[:,'Best VWAP Asks'] = VWAP_asks
aug_1_df.loc[:,'Best VWAP Bids'] = VWAP_bids

aug_1_df.to_csv(r'C:\Users\payma\OneDrive\Documents\PAYTON\Cornell MFE\Covario\Data Analysis\CSVs\August_1.csv')

# signal generation for scatter plots

mid_dif = mids - vol_mids

avg_asks = []
avg_bids = []
for i in range(len(asks)):
    avg_asks.append((asks[i] + vol_asks[i])/2)
    avg_bids.append((bids[i] + vol_bids[i])/2)
    

returns = (pd.DataFrame(avg_bids)-pd.DataFrame(avg_asks).shift(1))/pd.DataFrame(avg_asks)
returns = returns.dropna()

mid_dif_20 = mid_dif[:-20]
mid_dif_10 = mid_dif[:-10]
mid_dif_5 = mid_dif[:-5]
mid_dif_1 = mid_dif[:-1]


plt.scatter(mid_dif_1, returns)
plt.title('8/1 Mid Price difference vs ~3 sec rolling returns')
plt.xlabel('Mid price - vol mid price')
plt.ylabel('~3 sec returns')
plt.show()

#trading strategy

midsdf = pd.DataFrame(mids)

p_short=midsdf.rolling(1).mean()
p_long=midsdf.rolling(2).mean()

sig=((p_short>p_long)*2-1+p_long.isnull()) 

plt.plot(mids)
plt.plot(p_short)
plt.plot(p_long)
plt.show()

returns = (midsdf/midsdf.shift(1))-1

stratreturns = returns * sig

stratpnl = ((stratreturns + 1).cumprod()-1)

plt.plot(stratpnl)
plt.title('20-30 sec Momentum Strategy')
plt.show()

trades = 1
for i in range(len(sig)-1):
    t = sig.iloc[i,0]
    if t != float(sig.iloc[i+1,0]):
        trades += 1
        
stratstd = stratreturns.std()
avgret = stratreturns.mean()

