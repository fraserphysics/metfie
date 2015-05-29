"""first.py: Python3 code for discrete approximation of integral
operator for first order Markov process with 1-d states.  Derived
from ../first.py

"""
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg

class LO(scipy.sparse.linalg.LinearOperator):
    '''Custom version of LinearOperator that implements A*x and x*A for
    the adjacency matrix definied by the parameter T:

    The image of g_0 is [0,g_0+T).
    '''
    def __init__(self,              # LO instance
                 T,                 # Slope times dt
                 g_step,            # Size of steps in position g
                 ):
        self.dtype = np.dtype(np.float64)
        self.T = T
        self.g_step = g_step
        self.n_g = int(np.ceil(1.0/g_step))
        self.shape = (self.n_g, self.n_g)

    def matvec(self, v, rv=None):
        '''Calculate rv= A*v and return.  Use array rv if passed, otherwise
        allocate rv.
        '''
        print('matvec')
        if rv == None:
            rv = np.zeros(self.n_g)
        else:
            rv[:] = 0.0
        g_step = self.g_step
        T_int = int(self.T*self.n_g + 1)
        for G in range(self.n_g):
            b = min(self.n_g, G+T_int)
            x = v[G]*g_step
            rv[0:b] += x
        return rv
    def rmatvec(self, v, rv=None):
        '''Calculate rv = v*A and return.  Use array rv if passed, otherwise
        allocate rv.
        '''
        print('rmmatvec')
        if rv == None:
            rv = np.zeros(self.n_g)
        else:
            rv[:] = 0.0
        g_step = self.g_step
        T_int = int(self.T*self.n_g + 1)
        for G in range(self.n_g): # add to rv[i]
            b = min(self.n_g, G+T_int)
            rv[G] += v[0:b].sum()*g_step
        return rv
    def symmetry(self, v):
        assert v.shape == (self.n_g,)
        return v[::-1]
    def __mul__(self,x):
        '''Calls matvec for self*v.  FixMe: This doesn't call rmatvec for
        v*self

        '''
        x = np.asarray(x)

        if x.ndim == 1:
            return self.matvec(x)
        elif x.ndim == 2 and x.shape[1] == 1:
            return self.matvec(self, x[0])
        else:
            raise ValueError('expected x.shape = (n,) or (n,1)')
def _test():
    T = 0.251
    g_step = 1.0/1000
    A = LO(T, g_step)
    v = np.ones(A.n_g)
    Av = A*v
    vA = A.rmatvec(v)
    import pylab
    pylab.plot(Av)
    pylab.plot(vA)
    error = np.abs(vA-A.symmetry(Av)).sum()
    print('error={0:f}'.format(error))
    pylab.show()

if __name__ == "__main__":
    _test()

#---------------
# Local Variables:
# mode: python
# End:
