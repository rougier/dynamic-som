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


def plot(axes, net):

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

    codebook = net.codebook
    axes.imshow(codebook, interpolation='nearest')
                         #interpolation='bicubic')

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
                  r'$elasticity = %.2f$' % (net.elasticity),
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

    n = 60
    epochs = 5000
    N = 5000

    np.random.seed(123)
    samples = np.zeros((N,3))
    samples[:,0] = np.random.uniform(low=0,high=1,size=N)
    samples[:,1] = np.random.uniform(low=0,high=1,size=N)
    samples[:,2] = np.random.uniform(low=0,high=1,size=N)

    samples[:,0] = np.random.randint(low=0,high=3,size=N)/2.
    samples[:,1] = np.random.randint(low=0,high=3,size=N)/2.
    samples[:,2] = np.random.randint(low=0,high=3,size=N)/2.

    print 'Dynamic Self-Organizing Map'
    np.random.seed(123)
    dsom = DSOM((n,n,3), elasticity=1.0, init_method='fixed')
    dsom.learn(samples, epochs) 

    fig = plt.figure(figsize=(8,8))
    axes = plt.subplot(111) 
    plot(axes,dsom)

    fig.savefig('color.png', dpi=150)
    #plt.show()
