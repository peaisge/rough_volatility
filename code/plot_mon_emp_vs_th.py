import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.special as scsp


three_m_swap_emp = []

for j in range(20151001,20151031):
    data = pd.read_csv("month/swaps_{}.csv".format(str(j)), sep=',')
    three_m_swap_emp.append(interp1d(data['times'], data['swaps'])(0.25/252))


H = 0.14
nu = 0.65

# predict var at t+delta using data up to t
def predict_delta(t, n, delta):
    s = np.arange(n)
    fraction = 1/(((s+0.5)**(H+0.5))*(s+0.5+delta))
    E_log = np.sum(np.log(var_data[t-n:t])*fraction[::-1])/np.sum(fraction)
    c = scsp.gamma(1.5-H)/(scsp.gamma(H+0.5)*scsp.gamma(2-2*H))
    return(np.exp(E_log+2*c*((nu/252)**2)*(delta**(2*H))))

def varSwapCurve(t, days):
  pts = np.arange(days+1)
  rv_predict = [predict_delta(t,500,s) for s in pts]
  varcurve = np.array([rv_predict[0],]+list(np.cumsum(np.array(rv_predict[1:]))/np.array(pts[1:])))
  return(varcurve)

vol = pd.read_csv("OxfordManRealizedVolatilityIndices.csv", header=2)
vol = vol[~np.isnan(vol['SPX2.rv'])]
var_data = np.array(vol.loc[:,"SPX2.rv"],dtype=float)

t =
three_m_swap_th = []
for j in range(30):
    three_m_swap_th.append(varSwapCurve(t+j,63)[-1])

plt.figure()
plt.plot(range(20151001,20151031),three_m_swap,'b')
plt.plot(range(20151001,20151031),three_m_swap,'r')
plt.show()
