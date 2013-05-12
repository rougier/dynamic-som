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
    from network import NG,SOM,DSOM, fromdistance, Identity
    from distribution import uniform, normal, ring
#    Z = Identity((20,20), (0,0))
#    plt.imshow(Z)
#    plt.colorbar()
#    plt.show()

    
    x1, x2 = 0.1, 0.9
    elasticity = 2.5
    lrate = 0.01
    for i in range(10000):
        # Sample 0.0
        d = np.sqrt((x1-0)**2)*elasticity
        d1 = lrate*np.exp(-(0.0/np.sqrt(2))**2/d**2)*np.sqrt((x1-0)**2)*(x1-0)
        d2 = lrate*np.exp(-(1.0/np.sqrt(2))**2/d**2)*np.sqrt((x2-0)**2)*(x2-0)
        x1 -= d1
        x2 -= d2

        #print x1,x2, d1, d2
        # Sample 1.0
        d = np.sqrt((x2-1)**2)*elasticity
        d1_ = lrate*np.exp(-(1.0/np.sqrt(2))**2/d**2)*np.sqrt((x1-1)**2)*(x1-1)
        d2_ = lrate*np.exp(-(0.0/np.sqrt(2))**2/d**2)*np.sqrt((x2-1)**2)*(x2-1)
        x1 -= d1_
        x2 -= d2_
        #print x1,x2, d1_, d2_

        if (d1+d1_) < 1e-15 and (d2+d2_) < 1e-15:
            break

    print i,":",x1, x2

#     print
#     n = 2
#     samples1 = uniform(n=1) 
#     samples1[...] = [0.0,0.5]
#     samples2 = uniform(n=1) 
#     samples2[...] = [1.0,0.5]
#     dsom = DSOM((n,1,2), elasticity=elasticity, lrate_i=0.1)
#     dsom.codebook[...] = [0.5,0.5]
#     for i in range(20000):
#         dsom.learn(samples1,1,show_progress=False)
#         x1,x2 = dsom.codebook[0,0][0], dsom.codebook[-1,0][0]
# #        print x1, x2
#         dsom.learn(samples2,1,show_progress=False)
#         x1,x2 = dsom.codebook[0,0][0], dsom.codebook[-1,0][0]
#         print x1, x2
#     print x1, x2

