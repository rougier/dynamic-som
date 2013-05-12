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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from network import NG,SOM,DSOM
    from distribution import uniform, normal, ring

    n = 2
    epochs = 5000
    N = 5

    np.random.seed(2)
    samples = np.zeros((N,2))
    #samples[:,0] = np.array([0.1, 0.1, 0.5, 0.9])
    #samples[:,1] = np.array([0.1, 0.9, 0.5, 0.9])
    samples[:,0] = np.array([0.1, 0.1, 0.5, 0.9, 0.9])
    samples[:,1] = np.array([0.1, 0.9, 0.5, 0.1, 0.9])


    print 'Neural Gas'
    np.random.seed(2)
    ng = NG((n,n,2))
    ng.learn(samples,epochs)
    print 'Self-Organizing Map'
    np.random.seed(2)
    som = SOM((n,n,2))
    som.learn(samples,epochs)
    print 'Dynamic Self-Organizing Map'
    np.random.seed(2)
    dsom = DSOM((n,n,2), elasticity=1.0, lrate=0.1)
    #dsom = DSOM((n,n,2), elasticity=1.75)
    dsom.learn(samples,epochs)

    fig = plt.figure(figsize=(21,8))
    fig.patch.set_alpha(0.0)

    axes = plt.subplot(1,3,1)
    ng.plot(axes)
    axes = fig.add_subplot(1,3,2)
    som.plot(axes)
    axes = fig.add_subplot(1,3,3)
    dsom.plot(axes)
    fig.savefig('false.png',dpi=150)
    
