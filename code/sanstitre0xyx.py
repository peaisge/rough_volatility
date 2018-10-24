import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#%% 1. Moindres carrés classiques

# 1.a
n = 10
sigma_mes = 0.1

X = np.linspace(0,1,n)
Y_reel = X
Y_obs = X + np.random.randn(n)*sigma_mes

# 1.b
H = np.zeros((n,2))
for j in range(n):
    for i in range(2):
        H[j,i] = X[j]**i

# 1.c
beta_chapeau = np.linalg.inv(np.transpose(H).dot(H)).dot(np.transpose(H)).dot(Y_obs)
beta_vrai = np.array([0,1])

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

plt.figure()
plt.plot(beta_vrai[0],beta_vrai[1],'r+')
ax = plt.gca()
cov = 6*(sigma_mes**2)*np.linalg.inv(np.transpose(H).dot(H))
vals, vecs = eigsorted(cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

width, height = 2*np.sqrt(vals)
ellip = Ellipse(xy=beta_chapeau, width=width, height=height, angle=theta, fill=False, color='b')

ax.add_artist(ellip)
plt.xlim([-0.5,0.5])
plt.ylim([0.5,1.5])
plt.xlabel('beta1')
plt.ylabel('beta2')
plt.show()

# 1.d
Y_chapeau = np.zeros((n))
q_pred = np.zeros((n))

for j in range(n):
    Y_chapeau[j] = np.transpose(H[j]).dot(beta_chapeau)
    q_pred[j] = np.sqrt(np.transpose(H[j]).dot(np.linalg.inv(np.transpose(H).dot(H)).dot(H[j])))

tube_bas_reel = Y_chapeau-2*sigma_mes*q_pred
tube_haut_reel = Y_chapeau+2*sigma_mes*q_pred
tube_bas_obs = Y_chapeau-2*sigma_mes*np.sqrt(1+q_pred**2)
tube_haut_obs = Y_chapeau+2*sigma_mes*np.sqrt(1+q_pred**2)

plt.figure()
plt.plot(X,Y_reel,'r',linewidth=3.3)
plt.plot(X,Y_chapeau,'b')
plt.plot(X,Y_obs,'k+')
plt.plot(X,tube_bas_reel,'b--')
plt.plot(X,tube_haut_reel,'b--')
plt.plot(X,tube_bas_obs,'g--')
plt.plot(X,tube_haut_obs,'g--')
plt.xlim([-0.2,1.2])
plt.ylim([-0.2,1.2])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
