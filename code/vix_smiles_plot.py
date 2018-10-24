import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

smiles = pd.read_csv('imp_vol_yahoo_vix 1602.csv')
smiles = smiles.drop_duplicates()
smiles = smiles.dropna(axis=0, how='any')


def svi(x, params): # params=(a,b,rho,m,sigma) x=log-moneyness
    return(params[0]+params[1]*(params[2]*(x-params[3])+np.sqrt((x-params[3])**2+params[4]**2)))

def error(params, x, v): # params=(a,b,rho,m,sigma) x=log-moneyness v=variance
    return(svi(x,params)-v)

params0 = [1,2,0.5,1,5]

smiles = smiles.loc[smiles.Imp_vol_bid<10]
T = smiles.groupby('T')['Imp_vol_bid'].apply(list).index
Strike = smiles.groupby('T')['Strike'].apply(list).values
smiles_bid_list = smiles.groupby('T')['Imp_vol_bid'].apply(list).values
smiles_ask_list = smiles.groupby('T')['Imp_vol_ask'].apply(list).values
smiles_mid_list = smiles.groupby('T')['Imp_vol_mid'].apply(list).values
UnderlyingPrice = smiles.groupby('T')['UnderlyingPrice'].apply(list).values
F = smiles.groupby('T')['F'].apply(list).values

fig, axes = plt.subplots(3,3)
fig.tight_layout(w_pad=0)
for i in range(9):
    log_moneyness = np.log(np.asarray(Strike[i])/np.asarray(F[i]))
    pts = np.where(np.logical_and(np.logical_not(np.isnan(smiles_mid_list[i])),np.asarray(smiles_mid_list[i])>0.05))[0]
    if (len(smiles_mid_list[i])>15):
        pts = pts[2:-2]
    best_params = least_squares(error, params0, args=(log_moneyness[pts], (np.asarray(smiles_mid_list[i])**2)[pts]),bounds=([0,0,-1,-1,0],[5,20,1,1,10]),method='trf').x
    if (i<28):
        axes[i//3,i%3].plot(log_moneyness,smiles_bid_list[i],'b*')
        axes[i//3,i%3].plot(log_moneyness,smiles_ask_list[i],'r*')
        axes[i//3,i%3].plot(log_moneyness,[np.sqrt(svi(x,best_params)) for x in log_moneyness],color="orange")
        axes[i//3,i%3].set_title("T = %.3f"%T[i])
        axes[i//3,i%3].set_xlim(-0.5,1.5)
        axes[i//3,i%3].locator_params(nbins=5, axis='x')

plt.show()
