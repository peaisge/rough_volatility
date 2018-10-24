import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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


import os.path

def f(row):
    row['Imp_vol_bid'] = find_vol(row['Bid_Call'],row['F'],row['Strike'],row['UnderlyingPrice'],row['T'])
    row['Imp_vol_ask'] = find_vol(row['Ask_Call'],row['F'],row['Strike'],row['UnderlyingPrice'],row['T'])
    row['Imp_vol_mid'] = find_vol(row['Mid_Call'],row['F'],row['Strike'],row['UnderlyingPrice'],row['T'])
    return(row)

for j in range(20151001,20151032):
    if os.path.isfile('month\options_{}.csv'.format(str(j))):
        print("Starting %i"%j)
        prices = pd.read_csv('month\mydata\options_{}.csv'.format(str(j)))
        prices = prices.apply(f,axis=1)
        prices.to_csv('month\mydata\imp_vol_spx_{}.csv'.format(str(j)), sep=',', encoding='utf-8', index=False)
