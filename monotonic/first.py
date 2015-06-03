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
        allocate rv.  In cython with prange, this is faster than matvec
        because writes (rv[G] += ...) are to chunks of memory that are
        different for different threads.
        '''
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
    def power(self, n_iter=3000, small=1.0e-6, v=None, op=None, verbose=False):
        '''Calculate self.eigevalue and self.eigenvector for the
        largest eigenvalue of op.
        '''
        if op == None: op = self.rmatvec
        if v != None:
            v_old = v
        else:
            v_old = np.ones(self.n_g)
        def step(s_old, # Last estimate of eigenvalue
                 v_old, # Last estimate of eigenvector.  Assume norm(v) = 1
                 ):
            v_new = op(v_old)
            s = LA.norm(v_new)
            v_new /= s
            dv = LA.norm(v_new-v_old)
            ds = abs((s-s_old)/s)
            return s, v_new, ds, dv
        s, v_old, ds, dv = step(0, v_old)
        for i in range(n_iter):
            s, v_old, ds, dv = step(s, v_old)
            if ds < small and dv < small:
                break
        if verbose: print(
'''With n_g=%d, finished power() at iteration %d.  Eigenvalue=%f
    ds=%g, dv=%g'''%(self.n_g, i, s, ds, dv))
        self.iterations = i
        self.eigenvalue = s
        self.eigenvector = v_old
        return s,v_old
def _test():
    T = 0.251
    g_step = 1.0/1000
    A = LO(T, g_step)
    v = np.ones(A.n_g)
    Av = A*v
    vA = A.rmatvec(v)
    error = np.abs(vA-A.symmetry(Av)).sum()
    assert error < 1e-10

    import pylab
    e_val = {}
    for n in (1, 2, 3, 5, 10, 20, 30):
        for n_g in (100, 1000, 3000, 10000):
            g_step = 1.0/n_g
            T = 1.0/n
            A = LO(T, g_step)
            A.power(op=A.matvec, small=1.0e-8, verbose=True)
            v = A.eigenvector
            x = np.linspace(0,1,A.n_g)
            pylab.plot(x,v/v[0])
            e_val[n] = A.eigenvalue
        print('\neigenvalue[{0}] = {1}\n'.format(n, e_val[n]))
    pylab.show()

if __name__ == "__main__":
    _test()

#---------------
# Local Variables:
# mode: python
# End:
