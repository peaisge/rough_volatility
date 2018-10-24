import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from scipy import stats
from datetime import datetime, timedelta
import os.path

#%%

prices = pd.read_csv('month\mydata\options_20151001.csv')

T = prices.groupby('T')['Strike'].apply(list).index
Strike = prices.groupby('T')['Strike'].apply(list).values
Ask_Call = prices.groupby('T')['Ask_Call'].apply(list).values
Ask_Put = prices.groupby('T')['Ask_Put'].apply(list).values
Bid_Call = prices.groupby('T')['Bid_Call'].apply(list).values
Bid_Put = prices.groupby('T')['Bid_Put'].apply(list).values

plt.figure()
plt.plot(Strike[10],Ask_Call[10],label='Call Ask')
plt.plot(Strike[10],Ask_Put[10],label='Put Ask')
plt.plot(Strike[10],Bid_Call[10],label='Call Bid')
plt.plot(Strike[10],Bid_Put[10],label='Put Bid')
plt.title("Bid and Ask Option prices")
plt.xlabel('Strike')
plt.legend()
plt.show()


#%%

plt.figure()
plt.plot(Strike[10],np.asarray(Ask_Call[10])-np.asarray(Bid_Call[10]),'b')
plt.title("Bid-Ask Spread on call options")
plt.xlabel('Strike')
plt.show()

plt.figure()
plt.plot(Strike[10],np.asarray(Ask_Put[10])-np.asarray(Bid_Put[10]),'b')
plt.title("Bid-Ask Spread on put options")
plt.xlabel('Strike')
plt.show()

#%%
