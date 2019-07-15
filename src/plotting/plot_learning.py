#! /usr/bin/env python
#
# Usage:
#
#

import sys
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

data1 = np.loadtxt(sys.argv[1])
data2 = np.loadtxt(sys.argv[2])
ep1, train1, test1 = data1.T
ep2, train2, test2 = data2.T

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(7, 7))
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.plot(ep1, train1, '--', color='orange', lw=2, label='Train: no pbc')
plt.plot(ep1, test1, '--', color='green', lw=2, label='Test: no pbc')

plt.plot(ep2, train2, '-', color='darkorange', lw=2, label='Train: pbc')
plt.plot(ep2, test2, '-', color='darkgreen', lw=2, label='Test: pbc')

plt.plot([0, 1.5 * ep1.max()], [0.01795, 0.01795], '--', lw=3, color='gray', label='Kozeny-Carman fit')

plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Mean Squared Error', fontsize=20, labelpad=0)

plt.legend(loc=0)
plt.show()
