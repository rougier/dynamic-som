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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import make_axes_locatable
from mpl_toolkits.axes_grid import AxesGrid
from progress import ProgressBar, Percentage, Bar


def fromdistance(fn, shape, center=None, dtype=float):
    '''Construct an array by executing a function over a normalized distance.
    
    For a 2d shape, The resulting array therefore has a value
    ``fn(sqrt((x-x0)²+(y-y0)²))`` at coordinate  ``(x,y)`` where x,y ∈ [-1,+1]²
    
    Parameters
    ----------
    fn : callable
        The function is called with one parameter representing the normalized
        distance. `fn` must be capable of operating on arrays, and should
        return a scalar value.
    shape : (N,) tuple of ints
        Shape of the output array, which also determines the shape of
        the coordinate arrays passed to `fn`.
    center : (N,) tuple of ints
        Center
    dtype : data-type, optional
        Data-type of the coordinate arrays passed to `fn`.  By default,
        `dtype` is float.
    '''
    def distance(*args):
        d = 0
        for i in range(len(shape)):
            d += ((args[i]-center[i])/float(max(1,shape[i]-1)))**2
#            d += ((args[i]-center[i])/float(shape[i]))**2
        return np.sqrt(d)/np.sqrt(len(shape))
    if center == None:
        center = np.array(list(shape))//2
    return fn(np.fromfunction(distance,shape,dtype=dtype))



def Gaussian(shape,center,sigma=0.5):
    ''' Return a two-dimensional gaussian with given shape.

    :Parameters:
       `shape` : (int,int)
           Shape of the output array
       `center`: (int,int)
           Center of Gaussian
       `sigma` : float
           Width of Gaussian
    '''
    def g(x):
        return np.exp(-x**2/sigma**2)
    return fromdistance(g,shape,center)


def Identity(shape,center):
    ''' Return a two-dimensional gaussian with given shape.

    :Parameters:
       `shape` : (int,int)
           Shape of the output array
       `center`: (int,int)
           Center of Gaussian
       `sigma` : float
           Width of Gaussian
    '''
    def identity(x):
        return x
    return fromdistance(identity,shape,center)



# -----------------------------------------------------------------------------
class MAP(object):
    ''' Neural Map class '''

    def __init__(self, shape=(10,10,2),
                 sigma_i = 10.00, sigma_f = 0.010,
                 lrate_i = 0.500, lrate_f = 0.005,
                 lrate = 0.1, elasticity = 2.0, init_method='random'):
        ''' Build map '''
        # Fixed initialization
        if init_method == 'fixed':
            self.codebook = np.ones(shape)*0.5

        # Regular grid initialization
        elif init_method == 'regular':
            self.codebook = np.zeros(shape)
            for i in range(shape[0]):
                self.codebook[i,:,0] = np.linspace(0,1,shape[1])
                self.codebook[:,i,1] = np.linspace(0,1,shape[1])
                
        # Random initialization
        else:
            self.codebook = np.random.random(shape)

        self.max = 0
        self.elasticity = elasticity
        self.sigma_i = sigma_i # Initial neighborhood parameter
        self.sigma_f = sigma_f # Final neighborhood parameter
        self.lrate_i = lrate_i # Initial learning rate
        self.lrate_f = lrate_f # Final learning rate
        self.lrate   = lrate   # Constant learning rate
        self.entropy    = []
        self.distortion = []


    def learn(self, samples, epochs=25000, noise=0, test_samples=None, show_progress=True):
        ''' Learn given distribution using n data

        :Parameters:
            `samples` : [numpy array, ...]
                List of sample sets
            `epochs` : [int, ...]
                Number of epochs to be ran for each sample set
        '''

        # Check if samples is a list
        if type(samples) not in [tuple,list]:
            samples = (samples,)
            epochs = (epochs,)

        n = 0 # total number of epochs to be ran
        for j in range(len(samples)):
            n += epochs[j]
        self.entropy = []
        self.distortion = []

        if show_progress:
            bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=n).start()
        index = 0
        for j in range(len(samples)):
            self.samples = samples[j]
            I = np.random.randint(0,self.samples.shape[0],n)
            for i in range(epochs[j]):
                # Set sigma and learning rate according to current time
                t = index/float(n)
                lrate = self.lrate_i*(self.lrate_f/self.lrate_i)**t
                sigma = self.sigma_i*(self.sigma_f/self.sigma_i)**t
                C = self.codebook.copy()
                # Learn data

                S = self.samples[I[i]] + noise*(2*np.random.random(len(self.samples[I[i]]))-1)
                S = np.minimum(np.maximum(S,0),1)
                self.learn_data(S,lrate,sigma)

                #self.learn_data(self.samples[I[i]],lrate,sigma)
                if i%100 == 0:
                    self.entropy.append(((self.codebook-C)**2).sum())
                    if test_samples is not None:
                        distortion = self.compute_distortion(test_samples)
                    else:
                        distortion = self.compute_distortion(self.samples)
                    self.distortion.append(distortion)
                if show_progress:
                    bar.update(index+1)
                index = index+1
        if show_progress:
            bar.finish()


    def compute_distortion(self, samples):
        ''' '''
        distortion = 0
        for i in range(samples.shape[0]):
            data = samples[i]
            D = ((self.codebook-data)**2).sum(axis=-1)
            distortion += D.min()
        distortion /= float(samples.shape[0])
        return distortion


    def plot(self, axes):
        ''' Plot network on given axes

         :Parameters:
         `axes` : matploltlib Axes
             axes where to draw network
        '''

        classname = self.__class__.__name__

        # Plot samples
        axes.scatter(self.samples[:,0], self.samples[:,1], s=1.0, color='b', alpha=0.25)

        fig = plt.gcf()
        divider = make_axes_locatable(axes)

        # Plot network
        C = self.codebook
        Cx,Cy = C[...,0], C[...,1]
        if classname != 'NG':
            for i in range(C.shape[0]):
                axes.plot (Cx[i,:], Cy[i,:], 'k', alpha=0.85, lw=1.5)
            for i in range(C.shape[1]):
                axes.plot (Cx[:,i], Cy[:,i], 'k', alpha=0.85, lw=1.5)
        axes.scatter (Cx.flatten(), Cy.flatten(), s=50, c= 'w', edgecolors='k', zorder=10)
        axes.axis([0,1,0,1])
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_aspect(1)

        # Plot distortion
        subaxes = divider.new_vertical(1.0, pad=0.4, sharex=axes)
        fig.add_axes(subaxes)
        subaxes.set_xticks([])
        subaxes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
        subaxes.yaxis.set_ticks_position('right')
        subaxes.set_ylabel('Distortion')
        subaxes.set_xlabel('Time')
        #subaxes.axis([0,1,0,1])
        Y = self.distortion[::1]
        X = np.arange(len(Y))/float(len(Y)-1)
        subaxes.plot(X,Y)
        axes.axis([0,1,0,1])

        if classname == 'NG':
            plt.title('Neural Gas', fontsize=20)
        elif classname == 'SOM':
            plt.title('Self-Organizing Map', fontsize=20)
        elif classname == 'DSOM':
            plt.title('Dynamic Self-Organizing Map', fontsize=20)
        if classname == 'NG':
            axes.text(0.5, -0.01,
                      r'$\lambda_i = %.3f,\lambda_f = %.3f, \varepsilon_i=%.3f, \varepsilon_f=%.3f$' % (
                    self.sigma_i, self.sigma_f, self.lrate_i, self.lrate_f),
                      fontsize=16, 
                      horizontalalignment='center',
                      verticalalignment='top',
                      transform = axes.transAxes)
        if classname == 'SOM':
            axes.text(0.5, -0.01,
                      r'$\sigma_i = %.3f,\sigma_f = %.3f, \varepsilon_i=%.3f, \varepsilon_f=%.3f$' % (
                    self.sigma_i, self.sigma_f, self.lrate_i, self.lrate_f),
                      fontsize=16, 
                      horizontalalignment='center',
                      verticalalignment='top',
                      transform = axes.transAxes)
        elif classname == 'DSOM':
            axes.text(0.5, -0.01,
                      r'$elasticity = %.2f$, $\varepsilon = %.3f$' % (self.elasticity, self.lrate),
                      fontsize=16, 
                      horizontalalignment='center',
                      verticalalignment='top',
                      transform = axes.transAxes)




# -----------------------------------------------------------------------------
class SOM(MAP):
    ''' Self Organizing Map class '''

    def learn_data(self, data, lrate, sigma):
        ''' Learn a single data using lrate and sigma parameter

        :Parameters:
            `lrate` : float
                Learning rate
            `sigma` : float
                Neighborhood width
        '''

        # Compute distances to data 
        D = ((self.codebook-data)**2).sum(axis=-1)

        # Get index of nearest node (minimum distance)
        winner = np.unravel_index(np.argmin(D), D.shape)

        # Generate a Gaussian centered on winner
        G = Gaussian(D.shape, winner, sigma)
        G = np.nan_to_num(G)

        # Move nodes towards data according to Gaussian 
        delta = (self.codebook - data)
        for i in range(self.codebook.shape[-1]):
            self.codebook[...,i] -= lrate * G * delta[...,i]


# -----------------------------------------------------------------------------
class DSOM(MAP):
    ''' Dynamic Self Organizing Map class '''

    def learn_data(self, data, lrate=0, sigma=0):
        ''' Learn a single datum 

        :Parameters:
            `data` : numpy array
                Data to be learned
        '''
        # Compute distances to data 
        D = ((self.codebook-data)**2).sum(axis=-1)

        # Get index of nearest node (minimum distance)
        winner = np.unravel_index(np.argmin(D), D.shape)

        # Dynamic version
        self.max = max(D.max(), self.max)
        d = np.sqrt(D/self.max)
        sigma = self.elasticity*d[winner]

        # Generate a Gaussian centered on winner
        G = Gaussian(D.shape, winner, sigma)
        G = np.nan_to_num(G)

        # Move nodes towards data according to Gaussian 
        delta = (self.codebook - data)
        for i in range(self.codebook.shape[-1]):
            self.codebook[...,i] -= self.lrate*d*G*delta[...,i]


# -----------------------------------------------------------------------------
class NG(MAP):
    ''' Neural Gas class '''

    def learn_data(self, data, lrate, sigma):
        ''' Learn a single data using lrate and sigma parameter

        :Parameters:
            `lrate` : float
                Learning rate
            `sigma` : float
                Neighborhood width
        '''

        # Compute distances to data 
        D = ((self.codebook-data)**2).sum(axis=-1).flatten()

        # Ordered distance indices
        I = np.argsort(np.argsort(D))

        # Compute h(k/sigma)
        H = np.exp(-I/sigma).reshape(self.codebook.shape[:-1])

        # Move nodes towards data according to H
        delta = data - self.codebook
        for i in range(self.codebook.shape[-1]):
            self.codebook[...,i] += lrate * H * delta[...,i]
