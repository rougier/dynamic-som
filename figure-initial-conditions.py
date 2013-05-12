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
    samples = normal(n=N) 

    print 'Dynamic Self-Organizing Map, regular init'
    np.random.seed(123)
    dsom1 = DSOM((n,n,2), elasticity=1.75, init_method='regular')
    dsom1.learn(samples,epochs)

    print 'Dynamic Self-Organizing Map, fixed init'
    np.random.seed(123)
    dsom2 = DSOM((n,n,2), elasticity=1.75, init_method='fixed')
    dsom2.learn(samples,epochs)

    print 'Dynamic Self-Organizing Map, random init'
    np.random.seed(123)
    dsom3 = DSOM((n,n,2), elasticity=1.75, init_method='random')
    dsom3.learn(samples,epochs)

    fig = plt.figure(figsize=(21,8))
    axes = plt.subplot(1,3,1)
    dsom1.plot(axes)
    axes = fig.add_subplot(1,3,2)
    dsom2.plot(axes)
    axes = fig.add_subplot(1,3,3)
    dsom3.plot(axes)
    fig.savefig('initial-conditions.png',dpi=150)
    #plt.show()
    
