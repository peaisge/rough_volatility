import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta

def bs_price(cp,S,K,T,r,v):
    d1 = (np.log(S/K)+(r+(v**2)/2)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if cp == 'c':
        price = S*stats.norm.cdf(d1)-K*np.exp(-r*T)*stats.norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*stats.norm.cdf(-d2)-S*stats.norm.cdf(-d1)
    return(price)

def bs_vega(cp,S,K,T,r,v):
    d1 = (np.log(S/K)+(r+(v**2)/2)*T)/(v*np.sqrt(T))
    return(S*np.sqrt(T)*stats.norm.pdf(d1))

def find_vol(target_value, call_put, S, K, T, r):
    MAX_ITERATIONS = 50000
    PRECISION = 0.02

    sigma = 0.35
    for i in range(MAX_ITERATIONS):
        print(i)
        print("sigma = %.3f"%sigma)
        price = bs_price(call_put, S, K, T, r, sigma)
        print("price = %.3f"%price)
        vega = bs_vega(call_put, S, K, T, r, sigma)
        print("vega = %.18f"%vega)
        price = price
        diff = target_value - price
        print("target_value = %.8f"%target_value)

        if (abs(diff) < PRECISION):
            return(sigma)
        sigma = sigma+(1/(target_value+30)**2)*(diff*(diff<1)+((diff>1)*2*(diff>0)-1)*diff**2)/vega # f(x) / f'(x)
#        sigma = sigma+0.08*diff/vega # f(x) / f'(x)
        if (np.isnan(sigma)):
            return(sigma)
    print(sigma)
    return(sigma)


rs = [0.29,0.23,0.24,0.23,0.24,0.24,0.25,0.24,0.21,0.21,0.22,0.22,0.22,0.22,0.22,0.23,0.23,0.27,0.31,0.31,0.32]

import os.path
from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())

def f(row):
    row['T'] = (row['Expiration']-pd.to_datetime(str(j),format="%Y%m%d"))/ timedelta (days=365)
    row['Imp_vol_bid'] = find_vol(row['Bid'],'c',row['UnderlyingPrice'],row['Strike'],row['T'],r)
    row['Imp_vol_ask'] = find_vol(row['Ask'],'c',row['UnderlyingPrice'],row['Strike'],row['T'],r)
    row['Imp_vol_mid'] = find_vol(row['Mid'],'c',row['UnderlyingPrice'],row['Strike'],row['T'],r)
    return(row)

for j in range(20151001,20151002):
    if os.path.isfile('month\options_{}.csv'.format(str(j))):
        print("Starting %i"%j)
        calls = pd.read_csv('month\options_{}.csv'.format(str(j)))
        calls = calls.loc[calls['UnderlyingSymbol'] == 'SPX']
        calls = calls[['UnderlyingPrice','Type','Expiration','Strike','Ask','Bid']]
        calls = calls[~(calls.Type=='put')]
        calls['Expiration'] = pd.to_datetime(calls['Expiration'], format="%m/%d/%Y")
        calls['Mid'] = (calls['Bid']+calls['Ask'])/2
        calls = calls[['UnderlyingPrice','Expiration','Strike','Ask','Bid','Mid']]
        calls = calls.drop_duplicates()
        calls = calls[::5]
        r = rs[j-20151001]
        calls = calls.progress_apply(f,axis=1)
        calls[['UnderlyingPrice','Strike','T','Imp_vol_bid','Imp_vol_ask','Imp_vol_mid']].to_csv('month\imp_vol_spx_{}.csv'.format(str(j)), sep=',', encoding='utf-8', index=False)
