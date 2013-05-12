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
    import os
    import numpy as np
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from network import NG,SOM,DSOM

    n = 32
    epochs = 10000
    N = 5000

    # Pick N points on a sphere uniformly
    np.random.seed(123)
    rho   = 0.5
    u = np.random.uniform(low=0, high=1, size=N)
    v = np.random.uniform(low=0, high=1, size=N)
    theta = 2*np.pi*u
    phi = np.arccos(2*v-1)
    samples = np.zeros((N,3))
    x = samples[:,0] = rho*np.cos(theta)*np.sin(phi)+.5
    y = samples[:,1] = rho*np.sin(theta)*np.sin(phi)+.5
    z = samples[:,2] = rho*np.cos(phi)+.5

    np.random.seed(123)
    dsom = DSOM((n,n,3), elasticity=1.0, init_method='fixed')
    dsom.learn(samples, epochs)

    fig = plt.figure(figsize=(10,10))
    axes = Axes3D(fig)
    axes.scatter(x,y,z)

    C = dsom.codebook
    Cx,Cy,Cz = C[...,0], C[...,1], C[...,2]
    for i in range(C.shape[0]):
        axes.plot (Cx[i,:], Cy[i,:], Cz[i,:], 'k', alpha=0.85, lw=1.5)
    for i in range(C.shape[1]):
        axes.plot (Cx[:,i], Cy[:,i], Cz[:,i], 'k', alpha=0.85, lw=1.5)
    axes.scatter (Cx.flatten(), Cy.flatten(), Cz.flatten(), s=50, c= 'w', edgecolors='k', zorder=10)

    file = open('sphere.plot', 'w')
    file.write('''set parametric\n''')
    file.write('''set hidden3d\n''')
    file.write('''unset key\n''')
    file.write('''set style line 2 lw 1 lc rgb "#000000"\n''')
    file.write('''set style line 1 lw 1 lc rgb "#999999"\n''')
    file.write('''set style increment user\n''')
    file.write('''set xrange [0:1]\n''')
    file.write('''set yrange [0:1]\n''')
    file.write('''set zrange [0:1]\n''')
    file.write('''set style data line\n''')
    file.write('''set ticslevel 0\n''')
    file.write('''set size ratio 1\n''')
    file.write('''set view 65,225\n''')
    file.write('''set terminal svg size 512,512\n''')
    file.write('''set output 'sphere.svg'\n''')
    file.write('''splot '-' using 1:2:3\n\n''')
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            file.write('%.3f %.3f %.3f\n' % (C[i,j,0],C[i,j,1],C[i,j,2]))
        file.write('''\n''')
    os.system('gnuplot sphere.plot\n')
    plt.show()
