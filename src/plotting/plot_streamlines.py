#! /usr/bin/env python
#
# Usage:
# ./plot_streamlines.py file_name lattice_size
#

import sys
import numpy
import numpy as np
import matplotlib.pyplot as plt


def format_axes(ax):
    ax.set_aspect('equal')
    ax.figure.subplots_adjust(bottom=0, top=1, left=0, right=1)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    for spine in ax.spines.itervalues():
        spine.set_visible(False)


if __name__ == "__main__":
    data = np.loadtxt(sys.argv[1])

    Xo, Yo, UX, UY, U = data.T

    Xo += 0.5
    Yo += 0.5
    LX = int(sys.argv[2])
    LY = 3 * LX

    Y, X = np.mgrid[0:LX, 0:LX]

    U = np.zeros([LX, LX])
    V = np.zeros([LX, LX])

    mask = np.zeros(U.shape, dtype=bool)

    for i in range(len(Xo)):
        xi = int(Xo[i])
        yi = int(Yo[i])
        if xi >= LX and xi < 2 * LX:
            U[yi][xi - LX] = UX[i]
            V[yi][xi - LX] = UY[i]

        if UX[i] == 0 and UY[i] == 0:
            mask[(LX - 1) - yi][xi - LX] = True
            mask[(LX - 1) - yi][xi - LX] = True

    speed = np.sqrt(U * U + V * V)
    lw = 5 * speed / speed.max()

    zero_v = lw < 0.15
    U[zero_v] = 0.0
    V[zero_v] = 0.0

    fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(6, 6))
    format_axes(ax1)

    ax1.streamplot(X, Y, U, V, density=5, arrowsize=0.01, linewidth=lw)

    ax1.imshow(~mask, extent=(0 - 0.5, LX - 0.5, 0 - 0.5, LX - 0.5), alpha=0.45,
               interpolation='nearest', cmap='gray', aspect='auto')
    ax1.set_aspect('equal')
    plt.show()
