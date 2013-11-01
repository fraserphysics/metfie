import pickle
triples,C = pickle.load(open('triples_C'))
triples.sort()
doubles = {}
for trip in triples:
    key = trip[:2]
    if doubles.has_key(key):
        doubles[key].append(trip[2])
    else:
        doubles[key] = [trip[2]]
for key in doubles.keys():
    L = doubles[key]
    L.sort()
    doubles[key] = (L[0],L[-1])
print('len(triples)=%d len(doubles)=%d'%(len(triples),len(doubles)))
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x=[]
y=[]
z=[]
w = []
for key in doubles.keys():
    i,j = key
    k = doubles[key][0]
    x.append(C[0,i])
    y.append(C[1,j])
    z.append(C[2,k])
    k = doubles[key][1]
    w.append(C[2,k])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker=',', label='bottom')
ax.scatter(x, y, w, c='r', marker=',', label='top')
ax.legend()

ax.set_xlabel('f(i)')
ax.set_ylabel('f(i+1)')
ax.set_zlabel('f(i+2)')

plt.show()

