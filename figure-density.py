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

    n = 8
    epochs = 20000
    N = 10000
    np.random.seed(123)
    area_1 = np.pi*0.5**2 - np.pi*0.25**2
    area_2 = np.pi*0.25**2

    n1 = int(area_1*25000)
    n2 = int(area_2*25000)
    samples = np.zeros((n1+n2,2))
    samples[:n1] = ring(n=n1, radius=(0.25,0.50))
    samples[n1:] = ring(n=n2, radius=(0.00,0.25))
    print 'Dynamic Self-Organizing Map 1'
    np.random.seed(123)
    dsom1 = DSOM((n,n,2), elasticity=1.25, init_method='fixed')
    dsom1.learn(samples,epochs)

    n1 = int(area_1*40000)
    n2 = int(area_2*10000)
    samples = np.zeros((n1+n2,2))
    samples[:n1] = ring(n=n1, radius=(0.25,0.50))
    samples[n1:] = ring(n=n2, radius=(0.00,0.25))
    print 'Dynamic Self-Organizing Map 2'
    np.random.seed(123)
    dsom2 = DSOM((n,n,2), elasticity=1.25, init_method='fixed')
    dsom2.learn(samples,epochs)

    n1 = int(area_1*10000)
    n2 = int(area_2*40000)
    samples = np.zeros((n1+n2,2))
    samples[:n1] = ring(n=n1, radius=(0.25,0.50))
    samples[n1:] = ring(n=n2, radius=(0.00,0.25))
    print 'Dynamic Self-Organizing Map 3'
    np.random.seed(123)
    dsom3 = DSOM((n,n,2), elasticity=1.25, init_method='fixed')
    dsom3.learn(samples,epochs)

    fig = plt.figure(figsize=(21,8))
    fig.patch.set_alpha(0.0)

    axes = fig.add_subplot(1,3,1)
    dsom1.plot(axes)
    axes = fig.add_subplot(1,3,2)
    dsom2.plot(axes)
    axes = fig.add_subplot(1,3,3)
    dsom3.plot(axes)
    fig.savefig('density.png',dpi=150)
#    plt.show()
    
