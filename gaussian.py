import numpy as np
import matplotlib.pyplot as plt

A = 1
x0, y0 = 0, 0
sigma_x, sigma_y = .75, 2
shape=(16,16)
theta=np.pi/3

a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
X,Y = np.meshgrid(np.arange(-5,+5,10./shape[0]),np.arange(-5,+5,10./shape[1]))
Z = A*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))

plt.imshow(Z, interpolation='bicubic', cmap=plt.cm.gray_r)
plt.show()

#surf(X,Y,Z);shading interp;view(-36,36);axis equal;drawnow
#end
