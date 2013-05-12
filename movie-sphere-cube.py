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
    epochs = 5000
    N = 5000

    # Pick N points on a sphere uniformly
    np.random.seed(123)
    rho   = 0.5
    u = np.random.uniform(low=0, high=1, size=N)
    v = np.random.uniform(low=0, high=1, size=N)
    theta = 2*np.pi*u
    phi = np.arccos(2*v-1)
    sphere = np.zeros((N,3))
    sphere[:,0] = rho*np.cos(theta)*np.sin(phi)+.5
    sphere[:,1] = rho*np.sin(theta)*np.sin(phi)+.5
    sphere[:,2] = rho*np.cos(phi)+.5

    # Pick N points on a cube uniformly
    np.random.seed(123)
    cube = np.zeros((N,3))
    cube[:,0] = np.random.uniform(low=0,high=1,size=N)
    cube[:,1] = np.random.uniform(low=0,high=1,size=N)
    cube[:,2] = np.random.uniform(low=0,high=1,size=N)
    for i in range(N):
        index = int(np.random.random()*3)
        value = int(np.random.random()*2)
        cube[i,index] = value

    samples = sphere
    np.random.seed(123)
    net = DSOM((n,n,3), elasticity=1.0, init_method='fixed')
    I = np.random.randint(0,samples.shape[0], epochs)
    bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=epochs).start()
    plotfile = '/tmp/plot.txt'
    datafile = '/tmp/data.txt'
    rot_x, rot_z = 65,225


    for i in range(epochs):
        if i == epochs//2:
            samples = cube
            I = np.random.randint(0,samples.shape[0], epochs)
        if i%5 == 0:
            rot_x = 20+(1+np.cos(i/float(epochs)*4*np.pi))*45
            rot_z = (rot_z+1) % 360
            filename = '/tmp/image-%05d' % i
            file = open(plotfile, 'w')
            file.write('''set parametric\n''')
            file.write('''set hidden3d\n''')
            file.write('''unset key\n''')
            file.write('''unset border\n''')
            file.write('''unset xtics\n''')
            file.write('''unset ytics\n''')
            file.write('''unset ztics\n''')
            file.write('''set style line 3 lw 1 lc rgb "#0000ff"\n''')
            file.write('''set style line 2 lw 1 lc rgb "#000000"\n''')
            file.write('''set style line 1 lw 1 lc rgb "#999999"\n''')
            file.write('''set style increment user\n''')
            file.write('''set xrange [0:1]\n''')
            file.write('''set yrange [0:1]\n''')
            file.write('''set zrange [0:1]\n''')
            file.write('''set style data line\n''')
            file.write('''set ticslevel 0\n''')
            file.write('''set size ratio 1\n''')
            file.write('''set title "Dynamic Self-Organising Map\\nNicolas Rougier & Yann Boniface"\n''')
            file.write('''set label "Self-reorganisation from sphere to cube surface" at screen .5, screen .1 center\n''')
            file.write('''set label "(http://www.loria.fr/~rougier/)" at screen .5, screen .065 center textcolor lt 3\n''')
            file.write('''set view %d,%d\n''' % (rot_x,rot_z))
            file.write('''set terminal pngcairo size 512,512\n''')
            file.write('''set output '%s.png'\n''' % filename)
#            file.write('''splot '%s' using 1:2:3, '%s' with point pt 6 lw .1\n''' % (datafile,datafile))
            file.write('''splot '%s' using 1:2:3\n''' % (datafile))
            file.close()
            file = open(datafile, 'w')
            C = net.codebook
            for x in range(C.shape[0]):
                for y in range(C.shape[1]):
                    file.write('%.3f %.3f %.3f\n' % (C[x,y,0],C[x,y,1],C[x,y,2]))
                file.write('''\n''')
            file.close()
            subprocess.call(['/usr/bin/gnuplot', plotfile])
        net.learn_data(samples[I[i]])
        bar.update(i)
    bar.finish()
#    os.system('''mencoder 'mf:///tmp/sphere*.png' -mf type=png:fps=25  -Ovc lavc -lavcopts \
#                 vcodec=mpeg4:vbitrate=2500  -oac copy -o sphere.avi''')


