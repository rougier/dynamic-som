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

    #n = 10
    epochs = 1000
    N = 2

    #np.random.seed(123)
    samples = uniform(n=N) 
    samples[:,0] = [0.0,1.0]
    samples[:,1] = [0.5,0.5]

    fig = plt.figure(figsize=(12,9))


    p = 200
    X,Y = np.zeros((p,)), np.zeros((p,))
    for n in [2,4,6,8,10]:
        for i in range(p):
            elasticity = 0.5 +i*2.0/p
            np.random.seed(123)
            dsom = DSOM((n,1,2), elasticity=elasticity)
            dsom.codebook[...] = [0.5,0.5]
            dsom.learn(samples,epochs,show_progress=False)
            x1,x2 = dsom.codebook[0,0][0], dsom.codebook[-1,0][0]
            X[i],Y[i] = elasticity, 1-np.sqrt((x2-x1)**2)
            print n, elasticity, Y[i]
        plt.plot(X,Y,lw=2)

    plt.legend(('n=2', 'n=4', 'n=6', 'n=8', 'n=10'), 'upper left')
    plt.xlabel('Elasticity', fontsize=20)
    plt.ylabel('Distortion', fontsize=20)
    plt.show()

