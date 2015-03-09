"""level.py:

1. Makes a plot of the level set of a quadratic form using an inverse
   covariance matrix

2. Uses Monte Carlo integration to estimate the covariance of points
   inside that level set.

3. Uses the estimated covariance to plot an approximation to the level set.

"""

import sys
import numpy as np
import numpy.random as nr
import numpy.linalg as LA

def ellipse(C,M=500):
    ''' Find M points on the level set x CI x = 1
    '''
    CI = LA.inv(C)
    a = CI[0,0]
    b = CI[0,1]
    c = CI[1,1]
    step = 2*np.pi/M
    theta = np.arange(0,2*np.pi+0.5*step,step)
    sin = np.sin(theta)
    cos = np.cos(theta)
    rr = 1/(a*cos*cos + 2*b*cos*sin + c*sin*sin)
    r = np.sqrt(rr)
    return (r*cos,r*sin)

def main(argv=None):

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('Qt4Agg')
    def test(z,CI,r=1.0):
        ''' Return True/False if the Mahalinobis distance from z to
        the origin is less/greater than r
        '''
        d2 = (z*np.dot(z,CI)).sum(axis=1)
        return d2 < r**2
    def estimate(test,pars,X,Y,N):
        ''' Estimate covariance of region allowed by test(z,*pars)
        draw points from box defined by X and Y till N points pass
        test
        '''
        N = int(N)
        Z = np.empty((N,2))
        i = 0
        while i<N:
            # Next line draws N trails
            z = nr.ranf(2*N).reshape((N,2))*[X[1]-X[0],Y[1]-Y[0]] + [X[0],Y[0]]
            t = test(z,*pars) # t is a vector of N Boolean values
            for j in xrange(N):
                if t[j]:
                    Z[i,:] = z[j,:]
                    i += 1
                    if i >= N:
                        break
        return np.dot(Z.T,Z)/N
    if argv is None:
        argv = sys.argv[1:]
    C = np.array([[2.0, -1.9],[-1.9, 2.0]])
    #C = np.dot(A,A.T)
    CI = LA.inv(C)
    x,y = ellipse(C)
    fudge = 0.01
    Range = lambda T,B: (B-fudge*(T-B), T+fudge*(T-B))
    Flat_Factor = 4.0 # Ratio of variance of flat distribution over 1
                      # sigma region to sigma
    CZ = estimate(
        test,(CI,1.0),
        Range(max(x),min(x)),
        Range(max(y),min(y)),
        1e5)*Flat_Factor
    #_vals,_vecs = LA.eigh(CZ)
    #CZ = np.dot(_vecs*_vals,_vecs.T)
    CZI = LA.inv(CZ)
    print('CI=%s\nCZI=%s'%(CI,CZI))
    u,v = ellipse(CZ)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y)
    ax.plot(u,v)
    plt.show()
if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# eval: (python-mode)
# End:
