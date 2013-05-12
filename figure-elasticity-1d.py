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


if __name__ == '__main__':
    import numpy as np
    import matplotlib
#    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from network import NG,SOM,DSOM
    from distribution import uniform, normal, ring

    n = 2
    epochs = 2500
    N = 2

    #np.random.seed(123)
    samples = uniform(n=N) 
    samples[:,0] = [0.0,1.0]
    samples[:,1] = [0.5,0.5]

    fig = plt.figure(figsize=(15,6))
    p = 120
    X,Y = np.zeros((p,)), np.zeros((p,))
    lrate = 0.1
    for s in [0.5,0.8,0.85,0.9,0.95]:
        for i in range(p):
            elasticity = 1.0 +i*(3.0/p)
            np.random.seed(123)
            dsom = DSOM((n,1,2), elasticity=elasticity, lrate=lrate)
            #dsom.codebook[...] = [0.5,0.5]
            dsom.codebook[0] = [s,0.5]
            dsom.codebook[1] = [1-s,0.5]
            dsom.learn(samples,epochs,show_progress=False)
            x1,x2 = dsom.codebook[0,0][0], dsom.codebook[-1,0][0]
            X[i],Y[i] = elasticity, 1-np.sqrt((x2-x1)**2)
            print s, elasticity, Y[i]
        plt.plot(X,Y,lw=2)
    plt.legend((r'$x_0 = 1-y_0 = 0.50$',
                r'$x_0 = 1-y_0 = 0.20$',
                r'$x_0 = 1-y_0 = 0.15$',
                r'$x_0 = 1-y_0 = 0.10$',
                r'$x_0 = 1-y_0 = 0.05$'), 'upper right')
    plt.xlabel('Elasticity', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.title('Error as a function of initial conditions and elasticity', fontsize=20)
    plt.show()

