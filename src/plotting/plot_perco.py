#! /usr/bin/env python
#
# Usage:
# python plot_perco.py ../../output/CNN/values.txt
#

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc


def read_file(fn):
    data = np.loadtxt(fn)
    phi, perc, k, t = data.T
    return phi, perc


def sigmoid(x, x0, k):
    y = erfc(k * (x - x0)) / 2
    return y


def resample(x, y):
    N = len(x)
    indices = np.random.randint(0, N, N)
    x_new = []
    y_new = []
    for i in range(N):
        x_new.append(x[indices[i]])
        y_new.append(y[indices[i]])
    return x_new, y_new


def estimate_params(x, y, N=1000):
    ks = []
    x0 = []
    for i in range(N):
        phi, perc = resample(x, y)
        popt, pcov = curve_fit(sigmoid, phi, perc)
        x0.append(popt[0])
        ks.append(popt[1])

    return np.mean(x0), np.std(x0), np.mean(ks), np.std(ks)


phi, perc = read_file(sys.argv[1])

x0, x0_std, k0, k0_std = estimate_params(phi, perc, N=1000)
phi_ = np.linspace(0.001, 0.999, 100)
perc_ = sigmoid(phi_, x0, k0)

fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(6, 2))
ax1.plot(phi, perc, 'o')
ax1.plot(phi_, perc_, "--")

plt.show()
