#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
            im = grid[index].imshow(Z, interpolation = 'nearest', vmin=0, vmax=1, cmap=plt.cm.hot)
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
    import sys
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid import make_axes_locatable
    from mpl_toolkits.axes_grid import AxesGrid
    from network import NG,SOM,DSOM

    def gaussian(shape=(16,16), center=(0,0), sigma=(1,1), theta=0):
        A = 1
        x0, y0 = center
        sigma_x, sigma_y = sigma
        a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
        b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
        c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
        X,Y = np.meshgrid(np.arange(-5,+5,10./shape[0]),np.arange(-5,+5,10./shape[1]))
        Z = A*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
        return Z

    n,p = 8, 16
    epochs = 20000
    N = 1000

    np.random.seed(123)
    samples = np.zeros((N,p*p))
    T = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=N)
    for i in range(N):
        samples[i] = gaussian(shape=(p,p), sigma=(.5,2), theta=T[i]).flatten()

    print 'Neural Gas'
    np.random.seed(123)
    ng = NG((n,n,p*p), init_method='fixed')
    ng.learn(samples,epochs, noise=0.1)

    print 'Self-Organizing Map'
    np.random.seed(123)
    som = SOM((n,n,p*p), init_method='fixed')
    som.learn(samples,epochs, noise=0.1)

    print 'Dynamic Self-Organizing Map'
    np.random.seed(123)
    dsom = DSOM((n,n,p*p), elasticity=1.5, init_method='fixed')
    dsom.learn(samples,epochs, noise=0.1)

    fig = plt.figure(figsize=(21,8))
    fig.patch.set_alpha(0.0)

    axes = plt.subplot(131, frameon=False)
    plot(ng,n,p)
    axes = plt.subplot(132, frameon=False)
    plot(som,n,p)
    axes = plt.subplot(133, frameon=False)
    plot(dsom,n,p)
    fig.savefig('gaussian-filters.png',dpi=150)
    #plt.show()


