import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from scipy import stats
from datetime import datetime, timedelta

#%%
def bs_price(F,K,S,T,sigma):
    d1 = (np.log(F/K)+((sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    price = (F*stats.norm.cdf(d1)-K*stats.norm.cdf(d2))*S/F
    return(price)

def find_vol(target_value,F,K,S,T):
    lower, upper = 0.0001, 20
    tol = 0.001
    lowerP = bs_price(F,K,S,T,lower)
    upperP = bs_price(F,K,S,T,upper)
    while (upper-lower>tol):
        mid = (lower+upper)/2
        midP = bs_price(F,K,S,T,mid)
        if ((midP-target_value)*(lowerP-target_value)<0):
            upper = mid
            upperP = midP
        else:
            lower = mid
            lowerP = midP
    return(lower)

def compute_expiration(row):
    row['T'] = (row['Expiration']-pd.to_datetime('20180216',format="%Y%m%d"))/ timedelta (days=365)
    return(row)

def f(row):
    row['Imp_vol_bid'] = find_vol(row['Bid_Call'],row['F'],row['Strike'],row['UnderlyingPrice'],row['T'])
    row['Imp_vol_ask'] = find_vol(row['Ask_Call'],row['F'],row['Strike'],row['UnderlyingPrice'],row['T'])
    row['Imp_vol_mid'] = find_vol(row['Mid_Call'],row['F'],row['Strike'],row['UnderlyingPrice'],row['T'])
    return(row)

prices = pd.read_excel('vix 1602.xlsx')
prices = prices[['UnderlyingPrice','Type','Expiration','Strike','Ask','Bid']]

prices['Expiration'] = pd.to_datetime(prices['Expiration'], format="%m/%d/%Y")
prices['Mid'] = (prices['Bid']+prices['Ask'])/2

# seperate calls and puts
calls = prices[~(prices.Type=='Put')]
puts = prices[~(prices.Type=='Call')]
calls = calls.drop(['Type'],axis=1)
puts = puts.drop(['Type'],axis=1)
calls.columns = ['UnderlyingPrice','Expiration','Strike','Ask_Call','Bid_Call','Mid_Call']
puts.columns = ['UnderlyingPrice','Expiration','Strike','Ask_Put','Bid_Put','Mid_Put']

# join
result = pd.merge(calls, puts, on=['UnderlyingPrice','Expiration', 'Strike'])
del calls
del puts
del prices

result = result.apply(compute_expiration,axis=1)
# C-P = S0-Kexp(-rT)
# K = F -> C-P=0
result['cp'] = result['Mid_Call']-result['Mid_Put']
T = result.groupby('T')['cp'].apply(list).index
Strike = result.groupby('T')['Strike'].apply(list).values
cp = result.groupby('T')['cp'].apply(list).values
result['F'] = 0
for t in range(len(T)):
    a,b = plb.polyfit(Strike[t], cp[t], 1)
    result.loc[result['T']==T[t],'F'] = b
result = result.drop(['cp','Expiration'],axis=1)

result = result.apply(f,axis=1)
result.to_csv('imp_vol_yahoo_vix 1602.csv', sep=',', encoding='utf-8', index=False)
