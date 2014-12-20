'''Illustrate spline basis function and its derivatives.
'''
from __future__ import print_function
from calc import spline
import numpy as np
import matplotlib.pyplot as plt
new_ax = lambda n : plt.figure(n).add_subplot(1,1,1)

x = np.linspace(0,12,13)
s = spline(x,np.zeros(x.shape)) # Function is zero everywhere
c = s.get_c()                   # The coefficients
t = s.get_t()                   # The knot locations
k = 6
print('Plots of basis function and 3 derivatives for knot {0:d} at {1:.1f}'\
.format(k, t[k]))
c[k] = 1.0
s.set_c(c)
x = np.linspace(0,12,1000)
n_x = len(x)
d = np.empty((4,n_x))
for i in range(n_x):
    d[:,i] = s.derivatives(x[i])
for i,key in enumerate(
'function, first derivative, second derivative, third derivative'.split(',')):
    new_ax(key).plot(x,d[i,:])   
    new_ax(key).plot(t,np.zeros(t.shape),'rx')
plt.show()
