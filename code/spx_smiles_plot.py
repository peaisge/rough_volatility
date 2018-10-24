import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy import stats

smiles = pd.read_csv('imp_vol_yahoo_spx_20160105.csv')
smiles = smiles.drop_duplicates()
smiles = smiles.dropna(axis=0, how='any')
T = smiles.groupby('T')['Imp_vol_bid'].apply(list).index
Strike = smiles.groupby('T')['Strike'].apply(list).values
smiles_bid_list = smiles.groupby('T')['Imp_vol_bid'].apply(list).values
smiles_ask_list = smiles.groupby('T')['Imp_vol_ask'].apply(list).values
smiles_mid_list = smiles.groupby('T')['Imp_vol_mid'].apply(list).values

#%%

def bs_price(cp,S,K,T,r,v):
    d1 = (np.log(S/K)+(r+(v**2)/2)*T)/(v*np.sqrt(T))
    d2 = d1-v*np.sqrt(T)
    if cp == 'c':
        price = S*stats.norm.cdf(d1)-K*np.exp(-r*T)*stats.norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*stats.norm.cdf(-d2)-S*stats.norm.cdf(-d1)
    return(price)

#S = 2810.30 #2016/01/20
S = 2016.52 #2016/01/05
#r = 0.0179 # T-bill 1-year interest rate 2016/01/20
r = 0.0068 # T-bill 1-year interest rate 2016/01/05

def svi(x, params):# params=(a,b,rho,m,sigma) x=log-moneyness
    return(params[0]+params[1]*(params[2]*(x-params[3])+np.sqrt((x-params[3])**2+params[4]**2)))

def error(params, x, v): # params=(a,b,rho,m,sigma) x=log-moneyness v=variance
    return(svi(x,params)-v)

params0 = [2,2,0.05,0.1,5]
params_list = []

fig, axes = plt.subplots(3,3)
for i in range(9):
    log_moneyness = np.log(np.asarray(Strike[i])/(S*np.exp(r*T[i])))
    pts = np.where(np.logical_and(np.logical_not(np.isnan(smiles_mid_list[i])),np.asarray(smiles_mid_list[i])>0.05))[0]
    if (len(smiles_mid_list[i])>15):
        pts = pts[5:-2]
    best_params = least_squares(error, params0, args=(log_moneyness[pts], (np.asarray(smiles_mid_list[i])**2)[pts]),bounds=([0,0,-1,-1,0],[5,20,1,1,10]),method='trf').x
    params_list.append(best_params)
    axes[i//3,i%3].plot(log_moneyness,smiles_bid_list[i],'b*')
    axes[i//3,i%3].plot(log_moneyness,smiles_ask_list[i],'r*')
    axes[i//3,i%3].plot(log_moneyness,[np.sqrt(svi(x,best_params)) for x in log_moneyness],color="orange")
    axes[i//3,i%3].set_title("T = %.3f"%T[i])
    axes[i//3,i%3].set_ylim(0, 0.3)
plt.show()

#%%

times = []
swaps = []
xmin = -0.2
xmax = 0.2
nb_points = 100
x1 = np.linspace(xmin,0,nb_points)
x2 = np.linspace(0,xmax,nb_points)

for i in range(9):
    if (i!=7):
        times.append(T[i])
        forward_price = S/np.exp(r*T[i])
        int_1 = (-xmin/nb_points)*np.asarray([bs_price('p',S,forward_price*np.exp(x),T[i],r,np.sqrt(svi(x,params_list[i]))) for x in x1])/np.exp(x1)
        int_2 = (xmax/nb_points)*np.asarray([bs_price('c',S,forward_price*np.exp(x),T[i],r,np.sqrt(svi(x,params_list[i]))) for x in x2])/np.exp(x2)
        swaps.append((2/(T[i]*S))*(np.sum(int_1)+np.sum(int_2)))

plt.figure()
plt.plot(times,swaps,'b*')
plt.show()
