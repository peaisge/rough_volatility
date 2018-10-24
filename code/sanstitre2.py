import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as scsp
from datetime import datetime, timedelta

#%% import data

vol = pd.read_csv("OxfordManRealizedVolatilityIndices.csv", header=2)
vol = vol[~np.isnan(vol['SPX2.rv'])]
var_data = np.array(vol.loc[:,"SPX2.rv"],dtype=float)

T1 = np.array(vol.loc[:,"DateID"],dtype=str)
T1 = np.array([datetime.strptime(t, '%Y%m%d') for t in T1])

#%% Predict vol using formula

H = 0.13
nu = 0.3

# predict var at t+delta using data up to t
def predict_delta(t, n, delta):
    if (delta==0):
        return(var_data[t])
    s = np.arange(n)
    fraction = 1/(((s+0.5)**(H+0.5))*(s+0.5+delta))
    E_log = np.sum(np.log(var_data[t-n:t])*fraction[::-1])/np.sum(fraction)
    c = scsp.gamma(1.5-H)/(scsp.gamma(H+0.5)*scsp.gamma(2-2*H))
    return(np.exp(E_log+2*c*((nu/252)**2)*(delta**(2*H))))

# test: predict tomorrow's vol
delta = 50
n = 500
RV_predict = [predict_delta(t,n,delta) for t in range(n,len(var_data)-delta)]
RV_real = var_data[n:-delta]

plt.figure()
plt.plot(np.sqrt(RV_real), "r")
plt.plot(np.sqrt(RV_predict), "b")
plt.show()

#%% Predict whole variance swap curve

def varSwapCurve(t, days):
  pts = np.arange(days+1)
  rv_predict = [predict_delta(t,1000,s) for s in pts]
  print(rv_predict)
  varcurve = np.array([rv_predict[0],]+list(np.cumsum(np.array(rv_predict[1:]))/np.array(pts[1:])))
  return(varcurve*252)

threshold = 0.002

from scipy.interpolate import splrep, splev
plt.figure()

swaps1 = pd.read_csv("month\mydata\swaps\swaps_20151001.csv")
swaps1['med'] = (swaps1['swap']*np.exp(0.024*swaps1['T'])).rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill')
outlier_idx1 = np.abs(swaps1['swap']-swaps1['med']) < threshold
f1 = splrep(swaps1['T'][outlier_idx1],swaps1['swap'][outlier_idx1],k=3,s=0.7)


swaps2 = pd.read_csv("month\mydata\swaps\swaps_20151019.csv")
swaps2['med'] = swaps2['swap'].rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill')
outlier_idx2 = np.abs(swaps2['swap']-swaps2['med']) < threshold
f2 = splrep(swaps2['T'][outlier_idx2],swaps2['swap'][outlier_idx2],k=3,s=0.7)

plt.plot(np.arange(252*2.5+1)/252, varSwapCurve(4118,252*2.5)*14/(1.29*1.4),'g--',
         swaps1['T'][outlier_idx1],swaps1['swap'][outlier_idx1],'g',
         swaps1['T'][outlier_idx1],splev(swaps1['T'][outlier_idx1],f1),'g')
plt.plot(np.arange(252*2.5+1)/252, varSwapCurve(4126,252*2.5)*14/(1.29*1.4),'r--',
         swaps2['T'][outlier_idx2],swaps2['swap'][outlier_idx2],'r',
         swaps2['T'][outlier_idx2],splev(swaps2['T'][outlier_idx2],f2),'r')
plt.xlim([0,2])
plt.show()

#%%

plt.figure()
pts = np.arange(500)
plt.plot(pts/252, [*252*predict_delta(4114,1000,s) for s in pts],'g--')
plt.show()

f = splrep(np.arange(252*2.5+1)/252,varSwapCurve(4114,252*2.5)*np.arange(252*2.5+1)/np.sqrt(252),k=5,s=4)
plt.figure()
#plt.plot(np.arange(252*2.5+1)/252, varSwapCurve(4114,252*2.5)*np.arange(252*2.5+1)/252, label="noisy data")
#plt.plot(np.arange(252*2.5+1)/252, splev(np.arange(252*2.5+1)/252,f), label="fitted")
plt.plot(np.arange(252*2.5+1)/252, splev(np.arange(252*2.5+1)/252,f,der=1)/10, label="1st derivative")
plt.show()

f = splrep(swaps1['T'],swaps1['swap']*np.exp(0.023*swaps1['T'])*swaps1['T'],k=5,s=4)

plt.figure()
plt.plot(swaps1['T'], 2*splev(swaps1['T'],f,der=1)/10,'g--')
plt.show()

f = splrep(swaps1['T'],2*swaps1['swap']*np.exp(0.023*swaps1['T']),k=5,s=4)

swaps2 = pd.read_csv("month\mydata\swaps\swaps_20151027.csv")

plt.figure()
plt.plot(swaps1['T'], 2*swaps1['swap']*np.exp(0.03*swaps1['T']),'r--',np.arange(252*2.5+1)/252, varSwapCurve(4114,252*2.5)*np.sqrt(252),'r')
plt.plot(swaps2['T'], 2*swaps2['swap']*np.exp(0.03*swaps2['T']),'g--',np.arange(252*2.5+1)/252, varSwapCurve(4124,252*2.5)*np.sqrt(252),'g')
plt.show()
