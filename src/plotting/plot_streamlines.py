#! /usr/bin/env python



import sys
import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


data = np.loadtxt(sys.argv[1])

Xo, Yo, UX, UY = data.T

print UX[0], Xo[0]
print len(UX)

LX = 64
LY = 192

Y, X = np.mgrid[0:LX, 0:LY]

U = np.zeros( [LX, LY] )
V = np.zeros( [LX, LY] )

mask = np.zeros(U.shape, dtype=bool)

print U.shape
for i in range( len( Xo ) ):
    xi = int(Xo[i])
    yi = int(Yo[i])
    U[yi][xi] = UX[i]
    V[yi][xi] = UY[i]

    if UX[i] == 0 and UY[i] == 0:
        mask[63-yi][xi] = True

print type(Y)
print type(U)
print X.shape
print Y.shape
print U.shape
print V.shape

speed = np.sqrt(U*U + V*V)
lw = 5*speed / speed.max()

zero_v = lw < 0.5
U[zero_v] = 0.0
V[zero_v] = 0.0

fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(6,2) )
ax1.streamplot(X, Y, U, V, density=[10,10]) #, linewidth=lw)

ax1.set_title('Varying Density')
ax1.scatter([10,10],[10,10],color='red')

w=2
ax1.imshow(~mask, extent=(0, 192, 0, 64), alpha=0.5,
          interpolation='nearest', cmap='gray', aspect='auto')
ax1.set_aspect('equal')

plt.tight_layout()
plt.show()
