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

import numpy as np
from PIL import Image


def normal(center=(0.5,0.5), scale=(0.125,0.125), bounds=(0,1,0,1), n=1):
    ''' Return n points drawn from a 2 dimensional normal distribution.
    
    :Parameters:
        `center` : (float,float)
            Center of disribution
        `scale` : (float, float)
            Scale of distribution
        `bounds` : (float,float,float,float)
            Bounds of distribution as (xmin,xmax,ymin,ymax)
        `n` : int
            Number of sample to generate
    '''
    Z = np.zeros((n,2))
    Z[:,0] = np.random.normal(center[0], scale[0], n)
    Z[:,0] = np.maximum(np.minimum(Z[:,0],bounds[1]),bounds[0])
    Z[:,1] = np.random.normal(center[1], scale[1], n)
    Z[:,1] = np.maximum(np.minimum(Z[:,1],bounds[3]),bounds[2])
    return Z

def uniform(center=(0.5,0.5), scale=(0.5,0.5), n=1):
    ''' Return n points drawn from a 2 dimensional normal distribution.
    
    :Parameters:
        `center` : (float,float)
            Center of disribution
        `scale` : (float, float)
            Scale of distribution
        `n` : int
            Number of sample to generate
    '''
    Z = np.zeros((n,2))
    Z[:,0] = np.random.uniform(center[0]-scale[0],center[0]+scale[0],n)
    Z[:,1] = np.random.uniform(center[1]-scale[1],center[1]+scale[1],n)
    return Z


def ring(center=(0.5,0.5), radius=(0.0,0.5), n=1):
    ''' Return n points drawn from a 2 dimensional normal distribution.
    
    :Parameters:
        `center` : (float,float)
            Center of disribution
        `radius` : (float, float)
            Inner/Outer radius
        `n` : int
            Number of sample to generate
    '''
    Z = np.zeros((n,2))
    rmin,rmax = radius
    xc,yc = center
    for i in range(Z.shape[0]):
        r = -1
        while r < rmin or r > rmax:
            x,y = np.random.random(), np.random.random()
            r = np.sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc))
        Z[i,:] = x,y
    return Z


def image(filename, shape=(8,8), n=1):
    image = np.array(Image.open(filename),dtype=float)/256.0
    Z = np.zeros((n,shape[0]*shape[1]))
    x = np.random.randint(0,image.shape[0]-shape[0]-1,n)
    y = np.random.randint(0,image.shape[1]-shape[1]-1,n)
    for i in range(n):
        Z[i] = image[x[i]:x[i]+shape[0],y[i]:y[i]+shape[1]].flatten()
    return Z
