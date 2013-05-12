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
    import os, subprocess
    import numpy as np
    from network import NG,SOM,DSOM
    from progress import ProgressBar, Percentage, Bar

    n = 32
    epochs = 20000
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
    net = DSOM((n,n,3), elasticity=1.0, init_method='fixed')
    I = np.random.randint(0,samples.shape[0], epochs)
    bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=epochs).start()
    plotfile = 'plot.txt'

    for i in range(epochs):
        net.learn_data(samples[I[i]])
        bar.update(i)
    bar.finish()

    rot_x = 65
    rot_z = 225
    file = open(plotfile, 'w')
    file.write('''set parametric\n''')
    file.write('''set hidden3d\n''')
    file.write('''unset key\n''')
#    file.write('''unset border\n''')
#    file.write('''unset xtics\n''')
#    file.write('''unset ytics\n''')
#    file.write('''unset ztics\n''')
    file.write('''set style line 2 lw 1 lc rgb "#000000"\n''')
    file.write('''set style line 1 lw 1 lc rgb "#999999"\n''')
    file.write('''set style increment user\n''')
    file.write('''set xrange [0:1]\n''')
    file.write('''set yrange [0:1]\n''')
    file.write('''set zrange [0:1]\n''')
    file.write('''set style data line\n''')
    file.write('''set ticslevel 0\n''')
    file.write('''set size ratio 1\n''')
    file.write('''set view %d,%d\n''' % (rot_x,rot_z))
    file.write('''splot '-' using 1:2:3\n''')
    C = net.codebook
    for x in range(C.shape[0]):
        for y in range(C.shape[1]):
            file.write('%.3f %.3f %.3f\n' % (C[x,y,0],C[x,y,1],C[x,y,2]))
        file.write('''\n''')
    file.close()


