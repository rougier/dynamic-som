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

    size = 8
    epochs = 20000
    n = 10000

    np.random.seed(12345)

    samples_1 = uniform(n=n, center=(0.25,0.25), scale=(0.25,0.25))
    samples_2 = uniform(n=n, center=(0.75,0.75), scale=(0.25,0.25))
    samples_3 = uniform(n=n, center=(0.25,0.75), scale=(0.25,0.25))
    samples_4 = uniform(n=n, center=(0.75,0.25), scale=(0.25,0.25))


    print 'Neural gas'
    np.random.seed(12345)
    ng = NG((size,size,2))
    ng.learn([samples_1,   samples_2,   samples_3,   samples_4],
             [2*epochs//8, 2*epochs//8, 2*epochs//8, 2*epochs//8])

    print 'Self-Organizing Map'
    np.random.seed(12345)
    som = SOM((size,size,2))
    som.learn([samples_1,   samples_2,   samples_3,   samples_4],
              [2*epochs//8, 2*epochs//8, 2*epochs//8, 2*epochs//8])

    print 'Dynamic Self-Organizing Map'
    np.random.seed(12345)
    dsom = DSOM((size,size,2), elasticity=2.5)
    dsom.learn([samples_1,   samples_2,   samples_3,   samples_4],
               [2*epochs//8, 2*epochs//8, 2*epochs//8, 2*epochs//8])

    fig = plt.figure(figsize=(21,8))
    fig.patch.set_alpha(0.0)

    axes = plt.subplot(1,3,1)
    ng.plot(axes)
    axes = plt.subplot(1,3,2)
    som.plot(axes)
    axes = plt.subplot(1,3,3)
    dsom.plot(axes)
    fig.savefig('dynamic.png',dpi=150)
    #plt.show()
    
