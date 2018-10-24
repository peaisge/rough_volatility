import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from scipy import stats
from datetime import datetime, timedelta
import os.path

#%%

def compute_expiration(row):
    row['T'] = (row['Expiration']-pd.to_datetime(str(j),format="%Y%m%d"))/ timedelta (days=365)
    return(row)

for j in range(20151001,20151032):
    if os.path.isfile('month\options_{}.csv'.format(str(j))):
        prices = pd.read_csv('month\options_{}.csv'.format(str(j)))
        prices = prices.loc[prices['UnderlyingSymbol'] == 'SPX']
        prices = prices[['UnderlyingPrice','Type','Expiration','Strike','Ask','Bid']]

        prices['Expiration'] = pd.to_datetime(prices['Expiration'], format="%m/%d/%Y")
        prices['Mid'] = (prices['Bid']+prices['Ask'])/2

        # seperate calls and puts
        calls = prices[~(prices.Type=='put')]
        puts = prices[~(prices.Type=='call')]
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
        result['cpab'] = result['Ask_Call']-result['Bid_Put']
        result['cpba'] = result['Bid_Call']-result['Ask_Put']
        T = result.groupby('T')['cp'].apply(list).index
        Strike = result.groupby('T')['Strike'].apply(list).values
        cp = result.groupby('T')['cp'].apply(list).values
        cpab = result.groupby('T')['cpab'].apply(list).values
        cpba = result.groupby('T')['cpba'].apply(list).values
        result['F'] = 0
        result['Fab'] = 0
        result['Fba'] = 0
        for t in range(len(T)):
            a,b = plb.polyfit(Strike[t], cp[t], 1)
            result.loc[result['T']==T[t],'F'] = b
            a,b = plb.polyfit(Strike[t], cpab[t], 1)
            result.loc[result['T']==T[t],'Fab'] = b
            a,b = plb.polyfit(Strike[t], cpba[t], 1)
            result.loc[result['T']==T[t],'Fba'] = b
        result = result.drop(['cp','Expiration'],axis=1)
        result.to_csv('month\mydata\options2_{}.csv'.format(str(j)))
