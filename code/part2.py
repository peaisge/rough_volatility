#%% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from datetime import datetime, timedelta
from scipy.stats import norm
from fbm import FBM
#%% import data

vol = pd.read_csv("OxfordManRealizedVolatilityIndices.csv", header=2)
vol5Nq_ = np.sqrt(np.array(vol.loc[:,"IXIC2.rv5ss"],dtype=float))
vol5Sp_ = np.sqrt(np.array(vol.loc[:,"SPX2.rv"],dtype=float))
T = np.array(vol.loc[:,"DateID"],dtype=str)
T = np.array([datetime.strptime(t, '%Y%m%d') for t in T])
del vol

#%%
plt.figure()
plt.plot(T,vol5Sp_)
plt.show()
#%% cleaning up
vol5Nq = vol5Nq_[~np.logical_or(np.isnan(vol5Nq_),vol5Nq_==0)]
vol5Sp = vol5Sp_[~np.logical_or(np.isnan(vol5Sp_),vol5Sp_==0)]

indice = vol5Sp

#%% logm(q,∆) as a function of log ∆ and estimation of H

Delta = np.arange(1,50)
Q = [0.5,1,1.5,2,3]

def m_function(q,delta, indice):
    return(np.mean(np.abs(np.log(indice)[delta:]-np.log(indice)[:-delta]) ** q))


M_Nq = [[m_function(q,delta,indice) for delta in Delta] for q in Q]
slopes=[]
plt.figure()
for _q in range(len(Q)):
    plt.plot(np.log(Delta),np.log(np.array(M_Nq[_q])),'*', label='q=%.1f'%Q[_q])
    a,b = plb.polyfit(np.log(Delta), np.log(np.array(M_Nq[_q])), 1)
    plt.plot(np.log(Delta), a*np.log(Delta)+b,'r')
    slopes.append(a)
plt.legend(loc='lower right')
plt.show()

#H = 0.132 ##Nasdaq
H = 0.130 ##S&P

plt.figure()
plt.plot(Q,slopes,'b',Q,[H*q for q in Q],'g')
plt.show()

#%% Histograms for various lags ∆ of the (overlapping) increments logσt+∆−logσtof
# the S&P log-volatility; normal fits in red; normal fit for ∆ = 1 day rescaled by ∆**H in blue


Deltas=[1,5,25,125]
X = np.linspace(-4,4,100)
f, axes = plt.subplots(2,2)
moy0=0
sd0=0
for i in range(len(Deltas)):
    inc_vol = np.log(indice[Deltas[i]:])-np.log(indice[:-Deltas[i]])
    axes[i//2,i%2].hist(inc_vol, normed=True, range=(-4,4), bins=60, color='w')
    moy = np.mean(inc_vol)
    sd = np.std(inc_vol)
    axes[i//2,i%2].plot(X, norm(moy,sd).pdf(X), color='r')
    axes[i//2,i%2].set_title("Delta = %i"%Deltas[i])
    if i==0:
        moy0=moy
        sd0=sd
    else:
        axes[i//2,i%2].plot(X, norm(moy0,sd0*(Deltas[i]**0.135)).pdf(X), 'b--')
plt.show()

#%% Empirical  counterpart  of  Cov[σt+∆,σt]  as  a  function  of ∆**2H
Delta = np.arange(1,50)
cov_delta = []
for delta in Delta:
    cov_delta.append(np.cov(np.log(indice[delta:]),np.log(indice[:-delta]))[0,1])
plt.figure()
plt.plot(Delta**(2*H),cov_delta,'b*')
a,b = plb.polyfit(Delta**(2*H), cov_delta, 1)
plt.plot(Delta**(2*H), a*Delta**(2*H)+b,'r')
plt.xlim([0,3])
plt.xlabel("∆**2H")
plt.ylabel("Cov[log(σt+∆),log(σt)]")
plt.show()

#%% Empirical  counterpart  of   log(E[σt+∆σt])  as  a  function  of ∆**2H
Delta = np.arange(1,300)
prod_delta = []
for delta in Delta:
    prod_delta.append(np.log(np.mean(indice[delta:]*indice[:-delta])))
plt.figure()
plt.plot(Delta**(2*H),prod_delta,'b*')
a,b = plb.polyfit(Delta**(2*H), prod_delta, 1)
plt.plot(Delta**(2*H), a*Delta**(2*H)+b,'r')
plt.xlim([1,6])
plt.xlabel("∆**2H")
plt.ylabel("log(E[σt+∆σt])")
plt.show()

#%% Empirical  counterpart  of  log(Cov[σt+∆,σt])  as  a  function  of log(∆)
Delta = np.arange(1,300)
prod_delta = []
for delta in Delta:
    prod_delta.append(np.log(np.mean(indice[delta:]*indice[:-delta])))
plt.figure()
plt.plot(np.log(Delta),prod_delta,'b*')
a,b = plb.polyfit(np.log(Delta), np.array(prod_delta), 1)
plt.plot(np.log(Delta), a*np.log(Delta)+b,'r')
plt.xlim([1,4])
plt.xlabel("log(∆)")
plt.ylabel("log(Cov[σt+∆,σt])")
plt.show()

#%%
Delta = np.arange(1,50)
X0 = -5
m = -5
days = T.size
points_per_5min = 1
points_per_day = points_per_5min*12*24
delta = 1/points_per_day

alpha_rough = 0
nu_rough = 0.72
alpha_fsv = 100
nu_fsv = 5.2

#alpha_rough = 0
#nu_rough = 0.65
#alpha_fsv = 100
#nu_fsv = 5

f_rough = FBM(n=days*points_per_day, hurst=0.13, length=days, method='daviesharte').fbm()
f_fsv = FBM(n=days*points_per_day, hurst=0.53, length=days, method='daviesharte').fbm()
X_rough = [X0]
X_fsv = [X0]

for n in range(days*points_per_day-1):
    X_rough.append(X_rough[-1]+(f_rough[n+1]-f_rough[n])*nu_rough+alpha_rough*delta*(m-X_rough[-1]))
    X_fsv.append(X_fsv[-1]+(f_fsv[n+1]-f_fsv[n])*nu_fsv+alpha_fsv*delta*(m-X_fsv[-1]))

M_Nq_data = [m_function(2,delta,indice) for delta in Delta]
M_Nq_rough = [m_function(2,delta,np.exp(np.asarray(X_rough))) for delta in Delta]
M_Nq_fsv = [m_function(2,delta,np.exp(np.asarray(X_fsv))) for delta in Delta]

plt.figure()

plt.plot(np.log(Delta),np.log(np.asarray(M_Nq_data)),'*')
plt.plot(np.log(Delta),np.log(np.asarray(M_Nq_rough)),'y')
plt.plot(np.log(Delta),np.log(np.asarray(M_Nq_fsv)),'b')

plt.show()


#%%  log(m(q,∆)) as a function of log(∆), simulated data, with realized variance estimators


nu = 0.3
m = -5
X0 = -5
P0 = 1
alpha = 0.0005
H = 0.13
days = T.size
points_per_5min = 5
points_per_day = points_per_5min*12*24
delta = 1/points_per_day

f = FBM(n=days*points_per_day, hurst=H, length=days, method='daviesharte').fbm()
X = [X0]
P = [P0]

for n in range(days*points_per_day-1):
    X.append(X[-1]+(f[n+1]-f[n])*nu+alpha*delta*(m-X[-1]))
    P.append(P[-1]*(1+np.exp(X[-2])*np.sqrt(delta)*np.random.randn()))

P = np.asarray(P)

# 5-minutes realized variance
RV = [np.sum((np.log(M[9*12*points_per_5min+points_per_5min:17*12*points_per_5min:points_per_5min]
/M[9*12*points_per_5min:17*12*points_per_5min-points_per_5min:points_per_5min]))**2) for M in np.split(P,days)]

# uncertainty zones
tick = 0.00005
eta = 0.25

P_tick = np.round(P/tick)*tick
P_tau = [P0]
for n in range(days*points_per_day-1):
    P_tau.append(P_tick[n+1]-(0.5-eta)*tick*(2*(P_tick[n+1]>P_tick[n])-1))
P_tau = np.asarray(P_tau)
IV = [24*np.sum(((M[10*12*points_per_5min+points_per_5min+1:11*12*points_per_5min]-M[10*12*points_per_5min+points_per_5min:11*12*points_per_5min-1])/M[10*12*points_per_5min+points_per_5min:11*12*points_per_5min-1])**2) for M in np.split(P_tau,days)]

Delta = np.arange(1,50)
M_Nq_sim_RV = [[m_function(q,delta,np.sqrt(RV)) for delta in Delta] for q in Q]
M_Nq_sim_IV = [[m_function(q,delta,np.sqrt(IV)[np.where(np.asarray(IV)!=0)]) for delta in Delta] for q in Q]
slopes_sim_RV=[]
slopes_sim_IV=[]
plt.figure()
for _q in range(len(Q)):
    plt.plot(np.log(Delta),np.log(np.array(M_Nq_sim_RV[_q])),'*', label='RV q=%.1f'%Q[_q])
    a,b = plb.polyfit(np.log(Delta), np.log(np.array(M_Nq_sim_RV[_q])), 1)
    plt.plot(np.log(Delta), a*np.log(Delta)+b,'r')
    slopes_sim_RV.append(a)
    plt.plot(np.log(Delta),np.log(np.array(M_Nq_sim_IV[_q])),'o', label='UZ q=%.1f'%Q[_q])
    a,b = plb.polyfit(np.log(Delta), np.log(np.array(M_Nq_sim_IV[_q])), 1)
    plt.plot(np.log(Delta), a*np.log(Delta)+b,'r')
    slopes_sim_IV.append(a)
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.plot(Q,slopes_sim_RV,'b',Q,[0.165*q for q in Q],'g')
plt.show()

plt.figure()
plt.plot(Q,slopes_sim_IV,'b',Q,[0.134*q for q in Q],'g')
plt.show()

#%% Volatility of the index (above) and of the model (below)

fig, (ax0,ax1) = plt.subplots(2,1)
ax0.plot(T,vol5Nq_)
ax1.plot(T,np.exp(X)[1::points_per_day])
ax1.set_ylim([0,0.07])
plt.show()

#%%
plt.figure()
plt.plot(np.linspace(0,days,len(P)),P)
plt.show()

#%%
Delta = np.arange(2,150)
V_emp = []
V_sim = []

for delta in Delta:
    V_emp.append(np.var(np.asarray([np.sum(M**2) for M in np.split(np.asarray(indice)[:(len(indice)//delta)*delta],delta)])))
    V_sim.append(np.var(np.asarray([np.sum(M) for M in np.split(np.asarray(RV)[:(len(RV)//delta)*delta],delta)])))

fig, (ax0,ax1) = plt.subplots(2,1)
ax0.plot(np.log(Delta),np.log(np.asarray(V_emp)), 'g*')
a,b = plb.polyfit(np.log(Delta), np.log(np.asarray(V_emp)), 1)
ax0.plot(np.log(Delta), a*np.log(Delta)+b,'r')
ax0.set_title('Data')
ax1.plot(np.log(Delta),np.log(np.asarray(V_sim)), 'g*')
a,b = plb.polyfit(np.log(Delta), np.log(np.asarray(V_sim)), 1)
ax1.plot(np.log(Delta), a*np.log(Delta)+b,'r')
ax1.set_title('model')
plt.show()

#%%

nu = 0.3
m = -5
X0 = -5
P0 = 1
alpha = 0.0005
H_vec = np.arange(0.01,0.4,0.01)
days = T.size
points_per_5min = 5
points_per_day = points_per_5min*12*24
delta = 1/points_per_day
slopes_RV=[]
slopes_UZ=[]


for H in H_vec:
    print(H)
    f = FBM(n=days*points_per_day, hurst=H, length=days, method='daviesharte').fbm()
    X = [X0]
    P = [P0]

    for n in range(days*points_per_day-1):
        X.append(X[-1]+(f[n+1]-f[n])*nu+alpha*delta*(m-X[-1]))
        P.append(P[-1]*(1+np.exp(X[-2])*np.sqrt(delta)*np.random.randn()))

    P = np.asarray(P)

    # 5-minutes realized variance
    RV = [np.sum((np.log(M[9*12*points_per_5min+points_per_5min:17*12*points_per_5min:points_per_5min]
    /M[9*12*points_per_5min:17*12*points_per_5min-points_per_5min:points_per_5min]))**2) for M in np.split(P,days)]

    # uncertainty zones
    tick = 0.00005
    eta = 0.25

    P_tick = np.round(P/tick)*tick
    P_tau = [P0]
    for n in range(days*points_per_day-1):
        P_tau.append(P_tick[n+1]-(0.5-eta)*tick*(2*(P_tick[n+1]>P_tick[n])-1))
    P_tau = np.asarray(P_tau)
    IV = [24*np.sum(((M[10*12*points_per_5min+points_per_5min+1:11*12*points_per_5min]-M[10*12*points_per_5min+points_per_5min:11*12*points_per_5min-1])/M[10*12*points_per_5min+points_per_5min:11*12*points_per_5min-1])**2) for M in np.split(P_tau,days)]

    Delta = np.arange(1,50)
    M_Nq_sim_RV = [[m_function(q,delta,np.sqrt(RV)) for delta in Delta] for q in Q]
    M_Nq_sim_IV = [[m_function(q,delta,np.sqrt(IV)[np.where(np.asarray(IV)!=0)]) for delta in Delta] for q in Q]
    slopes_sim_RV=[]
    slopes_sim_IV=[]
    plt.figure()
    for _q in range(len(Q)):
        a,b = plb.polyfit(np.log(Delta), np.log(np.array(M_Nq_sim_RV[_q])), 1)
        slopes_sim_RV.append(a)
        a,b = plb.polyfit(np.log(Delta), np.log(np.array(M_Nq_sim_IV[_q])), 1)
        slopes_sim_IV.append(a)
    a,b = plb.polyfit(Q, slopes_sim_RV, 1)
    slopes_RV.append(a)
    a,b = plb.polyfit(Q, slopes_sim_IV, 1)
    slopes_UZ.append(a)

plt.figure()
plt.plot(H_vec,H_vec,'r',H_vec,slopes_RV,'g',H_vec,slopes_UZ,'b')
plt.show()
