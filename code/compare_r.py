import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('month\mydata\options2_20151001.csv')

T = data.groupby('T')['F'].apply(list).index
UnderlyingPrice = data.groupby('T')['UnderlyingPrice'].apply(list).values
F = data.groupby('T')['F'].apply(list).values
Fab = data.groupby('T')['Fab'].apply(list).values
Fba = data.groupby('T')['Fba'].apply(list).values

S0 = UnderlyingPrice[0][0]
F_list = [x[0] for x in F]
Fab_list = [x[0] for x in Fab]
Fba_list = [x[0] for x in Fba]

plt.figure()
plt.plot(T,F_list,T,Fab_list,T,Fba_list)
plt.show()


plt.figure()
plt.plot(T,np.log(F_list/S0)/T,label='CallMid/PutMid')
plt.plot(T,np.log(Fab_list/S0)/T,label='CallAsk/PutBid')
plt.plot(T,np.log(Fba_list/S0)/T,label='CallBid/PutAsk')
plt.ylim(-0.1,0.03)
plt.legend()
plt.xlabel('T')
plt.ylabel('r')
plt.show()
