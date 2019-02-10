#! /usr/bin/env python


import sys
import numpy as np

from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="darkgrid")

data = np.loadtxt(sys.argv[1])
kappa_LB, kappa_CNN = data.T
kappa_LB = 10.**kappa_LB
kappa_CNN = 10.**kappa_CNN

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(7,7) )
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.plot(kappa_LB, kappa_CNN, '+', color='green')
#plt.plot([1e-3,1e3],[1e-3,1e3],'--', lw=3, color='darkblue')

plt.xlabel('lattice-Boltzmann', fontsize=20)
plt.ylabel('ConvNet', fontsize=20, labelpad=-8)

plt.show()

