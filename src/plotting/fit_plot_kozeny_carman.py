#! /usr/bin/env python


import sys
import numpy as np

from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="darkgrid")

def kozeny_carman(x, x_c, gamma, C):
    dx = x - x_c
    y = np.log10(C) + gamma * np.log10(dx) + 2.0 * np.log10(x/(1-x))
    return y


def mse(x, logk, x_c, gamma, C):
    N = 0
    mse = 0.0
    for i in range(len(x)):
        logk_fit = kozeny_carman(x[i], x_c, gamma, C)
        mse += (logk[i] - logk_fit)**2
        N += 1
    mse /= N
    return mse

data = np.loadtxt(sys.argv[1])
phi, perc, k, t = data.T

indcs = perc == 1
phi=phi[indcs]
k=k[indcs]

indcs = phi <=0.95
phi=phi[indcs]
k=k[indcs]

log10k = np.log10(k)

popt, pcov = curve_fit(kozeny_carman, phi, log10k, bounds=(0, [0.6, np.inf, np.inf]))

x = np.linspace(popt[0]+0.01,np.max(phi), 100)
log_k_fit = kozeny_carman(x, popt[0], popt[1], popt[2]) 
k_fit = 10**log_k_fit

print "MSE=",mse(phi, log10k, popt[0], popt[1], popt[2])

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(7,7) )
ax.set_yscale("log", nonposy='clip')

plt.plot(phi, k, '+', color='green')
plt.plot(x,k_fit,'--',color='darkblue',lw=3)
plt.xlabel('Porosity', fontsize=20)
plt.ylabel(r'Permeability', fontsize=20, labelpad=0)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.show()


