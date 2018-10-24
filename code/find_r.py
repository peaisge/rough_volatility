import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.linalg import sqrtm
from scipy.special import hyp2f1
from time import time

#%%


smiles = pd.read_csv('month\mydata\imp_vol\imp_vol_spx_20151001.csv')
smiles = smiles.loc[smiles.Imp_vol_bid<10]
T = smiles.groupby('T')['Imp_vol_bid'].apply(list).index
UnderlyingPrice = smiles.groupby('T')['UnderlyingPrice'].apply(list).values
K = smiles.groupby('T')['Strike'].apply(list).values
F = smiles.groupby('T')['F'].apply(list).values
F = smiles.groupby('T')['F'].apply(list).values
F = smiles.groupby('T')['F'].apply(list).values
smiles_bid_list = smiles.groupby('T')['Imp_vol_bid'].apply(list).values
smiles_ask_list = smiles.groupby('T')['Imp_vol_ask'].apply(list).values
smiles_mid_list = smiles.groupby('T')['Imp_vol_mid'].apply(list).values

S0 = UnderlyingPrice[0][0]
F_list = [x[0] for x in F]

plt.figure()
plt.plot(T,F_list)
plt.show()


plt.figure()
plt.plot(T,np.log(F_list/S0)/T)
plt.show()

r=0
#%% Parameters

H = .07
gamma = .5 - H
rho = -.9
eta = 1.9
m = 4
times = np.arange(1, 914) / 365 # intervalles de temps
times_discretized = np.arange(1, 913*m +1) / (365 * m) # discrÃ©tisation des intervalles de temps
N = len(times)
N_d = len(times_discretized)
S0 = 1921.77
discount_rate = -.023 # taux d'intÃ©rÃªt
xi_0 = .02

#%% Simulation of W_tilde and Z

def covW_fun_aux(x):
    assert x <= 1
    return ((1 - 2 * gamma) / (1 - gamma)) * (x**(gamma)) * hyp2f1(1, gamma, 2 - gamma, x)

def covW_fun(u, v):
    if u < v:
        return covW_fun(v, u)
    return v**(2*H) * covW_fun_aux(v/u)

def covWZ_fun(u, v):
    H_tilde = H + .5
    D = np.sqrt(2*H) / H_tilde
    return rho * D * (u ** H_tilde - (u - min(u, v)) ** H_tilde)

fWW = np.vectorize(lambda i, j: covW_fun(times_discretized[i], times_discretized[j]))
fZZ = np.vectorize(lambda i, j: min(times_discretized[i], times_discretized[j]))
fWZ = np.vectorize(lambda i, j: covWZ_fun(times_discretized[i], times_discretized[j]))
fWW_ufunc = np.frompyfunc(fWW, 2, 1)
fZZ_ufunc = np.frompyfunc(fZZ, 2, 1)
fWZ_ufunc = np.frompyfunc(fWZ, 2, 1)

integersNd = np.arange(N_d)
covWW = fWW_ufunc.outer(integersNd, integersNd)
covZZ = fZZ_ufunc.outer(integersNd, integersNd)
covWZ = fWZ_ufunc.outer(integersNd, integersNd)

covWW2 = np.zeros((N_d, N_d))
for i in range(N_d):
    for j in range(N_d):
        covWW2[i, j] = fWW(i, j)

del covWW

covWZ2 = np.zeros((N_d, N_d))
for i in range(N_d):
    for j in range(N_d):
        covWZ2[i, j] = fWZ(i, j)

del covWZ

covZZ2 = np.zeros((N_d, N_d))
for i in range(N_d):
    for j in range(N_d):
        covZZ2[i, j] = fZZ(i, j)

del covZZ

cov_matrix = np.bmat([[covWW2, covWZ2], [covWZ2.T, covZZ2]]) # matrice des covariances du vecteur (W, Z)

T_simul = np.union1d(T, times[:20])
T_simul = np.unique(T_simul)

np.savetxt("cov.csv", cov_matrix, delimiter=",")


#%%

#cov_matrix = np.genfromtxt('cov.csv',delimiter=',')

#%% Simulation of S and v
times = np.array(times,dtype='float32')
T_simul = np.array(T_simul,dtype='float32')
T_simul = np.unique(T_simul)

tt = [(times[k] in T_simul) for k in range(len(times))]

def simul(xi=xi_0):
    G = np.random.randn(2 * N_d) # gÃ©nÃ©ration d'une gaussienne centrÃ©e rÃ©duite
    B = sqrtm(cov_matrix) # racine carrÃ©e de la matrice de covariances
    WZ_sample = np.dot(B, G) # vecteur de mÃªme loi que (W, Z)
    W_sample, Z_sample = WZ_sample[:N_d], WZ_sample[N_d:]
    W_sample = np.insert(W_sample,0,0)
    Z_sample = np.insert(Z_sample,0,0)

    dt = (times_discretized[1] - times_discretized[0])
    my_times = np.insert(times_discretized,0,0)

    # Simulation of v
    integrande_sample = np.zeros_like(W_sample)
    for k in range(1,N_d+1):
        integrande_sample[k] = np.sum((1/(my_times[k]-my_times[:k])**(2*gamma))*dt)

    v_sample = xi_0 * np.exp(eta * W_sample - H * (eta**2) * integrande_sample)

    # Simulation of S
    int_sqrtv_dZ = np.cumsum(np.sqrt(v_sample[:-1]) * (Z_sample[1:] - Z_sample[:-1]))
    int_sqrtv_dZ = np.insert(int_sqrtv_dZ,0,0)
    v_sample[0] = 0
    int_v_dt = np.cumsum(v_sample* dt)
    S = S0*np.exp(int_sqrtv_dZ - .5 * int_v_dt)
    S = S[::m]
    S = S[1:]
    return S[tt]


#%%


n_mc = 150

Ss = np.zeros((n_mc,len(T_simul)))
for i in range(n_mc):
    Ss[i] = simul()


x = np.linspace(-0.4,0.4,200) # log(K/F)

calls = np.zeros((len(T_simul),len(x)))

for i in range(len(T_simul)):
    for j in range(len(x)):
        calls[i,j] = np.exp(-r*T_simul[i])*np.mean((Ss[:,i]>S0*np.exp(r*T_simul[i])*np.exp(x[j]))*(Ss[:,i]-S0*np.exp(r*T_simul[i])*np.exp(x[j])))

del Ss

def bs_price(call_put,F,K,S,T,sigma):
    d1 = (np.log(F/K)+((sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    if (call_put=='c'):
        price = (F*stats.norm.cdf(d1)-K*stats.norm.cdf(d2))*S/F
    else:
        price = (K*stats.norm.cdf(-d2)-F*stats.norm.cdf(-d1))*S/F
    return(price)

def find_vol(target_value,F,K,S,T):
    lower, upper = 0.000001, 5
    tol = 0.001
    lowerP = bs_price('c',F,K,S,T,lower)
    upperP = bs_price('c',F,K,S,T,upper)
    while (upper-lower>tol):
        mid = (lower+upper)/2
        midP = bs_price('c',F,K,S,T,mid)
        if ((midP-target_value)*(lowerP-target_value)<0):
            upper = mid
            upperP = midP
        else:
            lower = mid
            lowerP = midP
    return(lower)

def svi(z, params):# params=(a,b,rho,m,sigma) x=log-moneyness
    return(params[0]+params[1]*(params[2]*(z-params[3])+np.sqrt((z-params[3])**2+params[4]**2)))

def svi_prime(x, params):
    return(params[1]*(params[2]+(x-params[3])/np.sqrt((x-params[3])**2+params[4]**2)))

def error(params, z, v): # params=(a,b,rho,m,sigma) x=log-moneyness v=variance
    return(svi(z,params)-v)

params0 = [2,2,0.05,0.1,5]
params_list = []
ATM_skews_simul = []

imp_vols = np.zeros_like(calls)
for i in range(len(T_simul)):
    for j in range(len(x)):
        imp_vols[i,j] = find_vol(calls[i,j],S0*np.exp(r*T_simul[i]),S0*np.exp(r*T_simul[i])*np.exp(x[j]),S0,T_simul[i])
    pts = np.where(np.logical_and(np.logical_not(np.isnan(imp_vols[i])),np.asarray(imp_vols[i])<3))[0]
    best_params = least_squares(error, params0, args=(x[pts], np.asarray((imp_vols[i])**2)[pts]),bounds=([0,0,-1,-1,0],[5,20,1,1,10]),method='trf').x
    params_list.append([best_params])
    ATM_skews_simul.append(np.abs(svi_prime(0,best_params)))



#%% comparing

fig, axes = plt.subplots(4,7)
fig.tight_layout(w_pad=0)
ATM_skews_emp = []
for i in range(len(T)):
    index = next(j for j, x_j in enumerate(T_simul) if x_j == np.array(T[i],dtype='float32'))
    log_moneyness = np.log(np.asarray(K[i])/np.asarray(F[i]))
    if (i<28):
        axes[i//7,i%7].plot(log_moneyness[::4],smiles_bid_list[i][::4],'b*')
        axes[i//7,i%7].plot(log_moneyness[::4],smiles_ask_list[i][::4],'r*')
        best_params = least_squares(error, params0, args=(log_moneyness, (np.asarray(smiles_mid_list[i])**2)),bounds=([0,0,-1,-1,0],[5,20,1,1,10]),method='trf').x
        axes[i//7,i%7].plot(x,[np.sqrt(svi(y,params_list[index][0])) for y in x],color="orange")
        axes[i//7,i%7].set_title("T = %.3f"%T[i])
        axes[i//7,i%7].set_ylim(0, 1)
        axes[i//7,i%7].locator_params(nbins=5, axis='x')
#        plt.savefig('img\smiles_{}.png'.format(str(j)))
    ATM_skews_emp.append(np.abs(svi_prime(0,best_params)))
plt.show()


plt.figure()
pts_sk = np.where(np.asarray(ATM_skews_simul)<5)[0]
plt.plot(T,ATM_skews_emp,'k+')
plt.plot(T_simul[pts_sk],np.asarray(ATM_skews_simul)[pts_sk],'r')
plt.show()
#plt.savefig('img\skews_{}_{}.png'.format(str(j),params_skews[1]))

#%% get xi

swaps = []
xmin = -1
xmax = 1
nb_points = 100
x1 = np.linspace(xmin,0,nb_points)
x2 = np.linspace(0,xmax,nb_points)
for i in range(len(T_simul)):
    forward_price = S0*np.exp(r*T_simul[i])
    int_1 = (-xmin/nb_points)*np.asarray([bs_price('p',forward_price,forward_price*np.exp(y),S0,T_simul[i],np.sqrt(svi(y,params_list[i][0]))) for y in x1])/np.exp(x1)
    int_2 = (xmax/nb_points)*np.asarray([bs_price('c',forward_price,forward_price*np.exp(y),S0,T_simul[i],np.sqrt(svi(y,params_list[i][0]))) for y in x2])/np.exp(x2)
    swaps.append((2/(T_simul[i]*S0))*(np.sum(int_1)+np.sum(int_2)))
xi = np.zeros_like(swaps)
xi[:-1] = ((T_simul*swaps)[1:] -(T_simul*swaps)[:-1])/(T_simul[1]-T_simul[0])
xi[-1] = xi[-2]

plt.figure()
plt.plot(T_simul,xi)
plt.show()

#        df.to_csv("month\mydata\swaps\swaps_{}.csv".format(str(j)), sep=',',index=False)
