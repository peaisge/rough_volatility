import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta

#calls = pd.ExcelFile('calls 2101.xlsx').parse('Feuil1',thousands=',') # S&P
#calls['T'] = calls['Maturity'].apply(lambda d: (d-pd.to_datetime('2018-01-19'))/ timedelta (days=365))

calls = pd.read_csv('spx 20160105.csv')
calls = calls[['UnderlyingPrice','Type','Expiration','Strike','Ask','Bid']]
calls = calls[~(calls.Type=='put')]
calls['Expiration'] = pd.to_datetime(calls['Expiration'])
calls['T'] = calls['Expiration'].apply(lambda d: (d-pd.to_datetime('2016-01-05'))/ timedelta (days=365))

calls['Mid'] = (calls['Bid']+calls['Ask'])/2
#%%
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
    PRECISION = 0.01

    sigma = 0.2
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
#        sigma = sigma+(1/(i+1))*diff/vega # f(x) / f'(x)
        sigma = sigma+0.0005*diff/vega # f(x) / f'(x)
        if (np.isnan(sigma)):
            return(sigma)
    print(sigma)
    return(sigma)

#S = 2810.30 #s&p
S = 2016.52 #s&p
#r = 0.0179 # T-bill 1-year interest rate
r = 0.0068

calls['Imp_vol_bid'] = calls.apply(lambda row: find_vol(row['Bid'],'c',S,row['Strike'],row['T'],r), axis=1)
calls['Imp_vol_ask'] = calls.apply(lambda row: find_vol(row['Ask'],'c',S,row['Strike'],row['T'],r), axis=1)
calls['Imp_vol_mid'] = calls.apply(lambda row: find_vol(row['Mid'],'c',S,row['Strike'],row['T'],r), axis=1)

calls[['Strike','T','Imp_vol_bid','Imp_vol_ask','Imp_vol_mid']].to_csv('imp_vol_yahoo_spx_20160105.csv', sep=',', encoding='utf-8', index=False)
