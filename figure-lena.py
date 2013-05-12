#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either  version 3 of the  License, or (at your  option)
# any later version.
# 
# This program is  distributed in the hope that it will  be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public 
# License for  more details.
# 
# You should have received a copy  of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
# 
# Contact:  CORTEX Project - INRIA
#           INRIA Lorraine, 
#           Campus Scientifique, BP 239
#           54506 VANDOEUVRE-LES-NANCY CEDEX 
#           FRANCE


def plot(net, n, p):

    classname = net.__class__.__name__

    axes.set_xticks([])
    axes.set_yticks([])
    divider = make_axes_locatable(axes)
    subaxes = divider.new_vertical(1.0, pad=0.4, sharex=axes)
    fig.add_axes(subaxes)
    subaxes.set_xticks([])
    subaxes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
    subaxes.yaxis.set_ticks_position('right')
    subaxes.set_ylabel('Distortion')
    subaxes.set_xlabel('Time')

    Y = net.distortion[::1]
    X = np.arange(len(Y))/float(len(Y)-1)
    subaxes.plot(X,Y)

    if classname == 'NG':
        plt.title('Neural Gas', fontsize=20)
    elif classname == 'SOM':
        plt.title('Self-Organizing Map', fontsize=20)
    elif classname == 'DSOM':
        plt.title('Dynamic Self-Organizing Map', fontsize=20)

    axes.axis([0,1,0,1])
    axes.set_aspect(1)

    bounds = divider.locate(0,0).bounds
    grid = AxesGrid(fig, bounds, nrows_ncols = (n,n), axes_pad = 0.05, label_mode = "1")
    for row in range(n):
        for col in range(n):
            index = row*n+col
            Z = net.codebook[row,col].reshape(p,p) 
            im = grid[index].imshow(Z, interpolation = 'nearest', vmin=0, vmax=1, cmap=plt.cm.gray)
            grid[index].set_yticks([])
            grid[index].set_xticks([])

    classname = net.__class__.__name__
    if classname == 'NG':
        axes.text(0.5, -0.01,
                  r'$\lambda_i = %.3f,\lambda_f = %.3f, \varepsilon_i=%.3f, \varepsilon_f=%.3f$' % (
                net.sigma_i, net.sigma_f, net.lrate_i, net.lrate_f),
                  fontsize=16, 
                  horizontalalignment='center',
                  verticalalignment='top',
                  transform = axes.transAxes)
    if classname == 'SOM':
        axes.text(0.5, -0.01,
                  r'$\sigma_i = %.3f,\sigma_f = %.3f, \varepsilon_i=%.3f, \varepsilon_f=%.3f$' % (
                net.sigma_i, net.sigma_f, net.lrate_i, net.lrate_f),
                  fontsize=16, 
                  horizontalalignment='center',
                  verticalalignment='top',
                  transform = axes.transAxes)
    elif classname == 'DSOM':
        axes.text(0.5, -0.01,
                  r'$elasticity = %.2f$, $\varepsilon = %.3f$' % (net.elasticity, net.lrate),
                  fontsize=16, 
                  horizontalalignment='center',
                  verticalalignment='top',
                  transform = axes.transAxes)


if __name__ == '__main__':
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid import make_axes_locatable
    from mpl_toolkits.axes_grid import AxesGrid
    from network import NG,SOM,DSOM
    from distribution import uniform, normal, ring, image

    n,p = 8, 8
    epochs = 10000
    N = 5000

    np.random.seed(123)
    samples = image(filename='lena.png', shape=(p,p), n=N) 

    print 'Neural Gas'
    np.random.seed(123)
    ng = NG((n,n,p*p), init_method='fixed')
    ng.learn(samples,epochs)

    print 'Self-Organizing Map'
    np.random.seed(123)
    som = SOM((n,n,p*p), init_method='fixed')
    som.learn(samples,epochs)

    print 'Dynamic Self-Organizing Map'
    np.random.seed(123)
    dsom = DSOM((n,n,p*p), elasticity=0.5, init_method='fixed')
    dsom.learn(samples,epochs)

    # fig = plt.figure(figsize=(10,10))
    # axes = plt.subplot(111, frameon=False)
    # plot(dsom,n,p)

    fig = plt.figure(figsize=(21,8))
    fig.patch.set_alpha(0.0)
    axes = plt.subplot(131, frameon=False)
    plot(ng,n,p)
    axes = plt.subplot(132, frameon=False)
    plot(som,n,p)
    axes = plt.subplot(133, frameon=False)
    plot(dsom,n,p)
    fig.savefig('image.png',dpi=150)

    plt.show()


