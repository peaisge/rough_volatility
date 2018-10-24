import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from scipy.optimize import least_squares
from scipy import stats


def bs_price(call_put,F,K,S,T,sigma):
    d1 = (np.log(F/K)+((sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    if (call_put=='c'):
        price = (F*stats.norm.cdf(d1)-K*stats.norm.cdf(d2))*S/F
    else:
        price = (K*stats.norm.cdf(-d2)-F*stats.norm.cdf(-d1))*S/F
    return(price)

def svi(x, params):# params=(a,b,rho,m,sigma) x=log-moneyness
    return(params[0]+params[1]*(params[2]*(x-params[3])+np.sqrt((x-params[3])**2+params[4]**2)))

def svi_prime(x, params):
    return(params[1]*(params[2]+(x-params[3])/np.sqrt((x-params[3])**2+params[4]**2)))

def error(params, x, v): # params=(a,b,rho,m,sigma) x=log-moneyness v=variance
    return(svi(x,params)-v)

def power_law(t,params):
    return(params[0]*(t**(-params[1])))

def error_skew(params,t,skew):
    return(power_law(t,params)-skew)

params0 = [2,2,0.05,0.1,5]
params_list = []


for j in range(20151001,20151002):
    if os.path.isfile('month\mydata\imp_vol\imp_vol_spx_{}.csv'.format(str(j))):
        smiles = pd.read_csv('month\mydata\imp_vol\imp_vol_spx_{}.csv'.format(str(j)))
        smiles = smiles.loc[smiles.Imp_vol_bid<10]
        T = smiles.groupby('T')['Imp_vol_bid'].apply(list).index
        Strike = smiles.groupby('T')['Strike'].apply(list).values
        smiles_bid_list = smiles.groupby('T')['Imp_vol_bid'].apply(list).values
        smiles_ask_list = smiles.groupby('T')['Imp_vol_ask'].apply(list).values
        smiles_mid_list = smiles.groupby('T')['Imp_vol_mid'].apply(list).values
        UnderlyingPrice = smiles.groupby('T')['UnderlyingPrice'].apply(list).values
        F = smiles.groupby('T')['F'].apply(list).values
        fig, axes = plt.subplots(4,7)
        fig.tight_layout(w_pad=0)
        ATM_skews = []
        for i in range(len(T)):
            log_moneyness = np.log(np.asarray(Strike[i])/np.asarray(F[i]))
            pts = np.where(np.logical_and(np.logical_not(np.isnan(smiles_mid_list[i])),np.asarray(smiles_mid_list[i])>0.05))[0]
            if (len(smiles_mid_list[i])>15):
                pts = pts[2:-2]
            best_params = least_squares(error, params0, args=(log_moneyness[pts], (np.asarray(smiles_mid_list[i])**2)[pts]),bounds=([0,0,-1,-1,0],[5,20,1,1,10]),method='trf').x
            params_list.append([best_params])
            if (i<28):
                axes[i//7,i%7].plot(log_moneyness[::4],smiles_bid_list[i][::4],'b*')
                axes[i//7,i%7].plot(log_moneyness[::4],smiles_ask_list[i][::4],'r*')
                axes[i//7,i%7].plot(log_moneyness,[np.sqrt(svi(x,best_params)) for x in log_moneyness],color="orange")
                axes[i//7,i%7].set_title("T = %.3f"%T[i])
                axes[i//7,i%7].set_ylim(0, 0.5)
                axes[i//7,i%7].locator_params(nbins=5, axis='x')
            print(svi_prime(0,best_params))
            ATM_skews.append(np.abs(svi_prime(0,best_params)))
#        plt.savefig('img\smiles_{}.png'.format(str(j)))
        plt.show()
        times = []
        swaps = []
        xmin = -4
        xmax = 4
        nb_points = 200
        x1 = np.linspace(xmin,0,nb_points)
        x2 = np.linspace(0,xmax,nb_points)
        for i in range(len(T)):
            times.append(T[i])
            forward_price = F[i][0]
            S = UnderlyingPrice[i][0]
            int_1 = (-xmin/nb_points)*np.asarray([bs_price('p',forward_price,forward_price*np.exp(x),S,T[i],np.sqrt(svi(x,params_list[i][0]))) for x in x1])/np.exp(x1)
            int_2 = (xmax/nb_points)*np.asarray([bs_price('c',forward_price,forward_price*np.exp(x),S,T[i],np.sqrt(svi(x,params_list[i][0]))) for x in x2])/np.exp(x2)
            swaps.append((2/(T[i]*S))*(np.sum(int_1)+np.sum(int_2)))
        df = pd.DataFrame(data={"T": times, "swap": swaps})
        df.to_csv("month\mydata\swaps\swaps_{}.csv".format(str(j)), sep=',',index=False)

#
        param_skew0 = [0.5,0.5]
        params_skews = least_squares(error_skew, param_skew0, args=(T, np.asarray(ATM_skews)),bounds=([0,0],[10,2]),method='trf').x
        plt.figure()
        plt.plot(T,ATM_skews,'k+')
        plt.plot(T,power_law(T,params_skews),'r')
        plt.savefig('img\skews_{}_{}.png'.format(str(j),params_skews[1]))
