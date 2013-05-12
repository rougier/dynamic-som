#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    n = 50
    elasticity = 2.0
    Z = np.zeros((n,n))

    X = np.arange(0, 1, 0.005)
    Y = np.arange(0, 1, 0.005)
    X,Y = np.meshgrid(X, Y)    
    Z = np.nan_to_num (np.sqrt(np.exp(-X/Y**2 * 1.0/(elasticity**2))))
    fig = plt.figure(figsize=(12,9))
    fig.patch.set_alpha(0.0)

    plt.imshow(Z, extent=[0,1,0,1], interpolation='bicubic', origin='lower',
               cmap=plt.cm.PuOr, vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label('Learning rate modulation')
    CS = plt.contour(X, Y, Z, linewidths=2, colors='w')
    plt.clabel(CS, inline=1, fontsize=12, colors='w')
    plt.title('''Learning rate modulation as a function of a\n data $\mathbf{v}$, '''
              '''a neuron $i$ and a winner $s$''', fontsize=20)
    plt.xlabel(r'$|| \mathbf{p}_i - \mathbf{p}_s ||$', fontsize=20)
    plt.ylabel(r'$|| \mathbf{w}_s - \mathbf{v} ||$', fontsize=20)
    fig.savefig('learning-rate.pdf', transparent=True)
