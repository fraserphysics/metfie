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
        self.n_g = int(np.ceil(1.0/g_step))
        self.shape = (self.n_g, self.n_g)
        self.bins = np.linspace(0,1,self.n_g,endpoint=False)
        self.g_step = self.bins[1]
        assert len(self.bins) == self.n_g
        assert self.bins[0] == 0.0
        assert self.bins[-1] < 1.0 - self.g_step/2
    def index(self, x):
        if type(x) == float:
            return np.digitize([x], self.bins)[0]
        else:
            return np.digitize(x, self.bins)

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
    '''Takes 13 seconds on watcher'''
    import matplotlib.pyplot as plt  # must be after mpl.use
    T = 0.251
    g_step = 1.0/1000
    A = LO(T, g_step)
    v = np.ones(A.n_g)
    Av = A*v
    vA = A.rmatvec(v)
    error = np.abs(vA-A.symmetry(Av)).sum()
    assert error < 1e-10

    e_val = {}
    fig = plt.figure('eigenvectors')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r'$g$')
    ax.set_ylabel(r'eigenvector $v$')
    for n in (1, 2, 3, 5, 10, 20):
        T = 1.0/n
        for n_g in (100, 300, 1000, 3000):
            g_step = 1.0/n_g
            A = LO(T, g_step)
            A.power(op=A.matvec, small=1.0e-8, verbose=False)
            v = A.eigenvector
            x = np.linspace(0,1,A.n_g)
            ax.plot(x,v/v[0])
            e_val[n] = A.eigenvalue
        print('eigenvalue[T=1/{0}] = {1}'.format(n, e_val[n]))
    plt.show()
    return 0
def conditional(
        delta,
        f,
        h,
        n_T,
        n_g
        ):
    '''Calculate estimate of conditional density at t=0 given g(-delta) = f and
    g(delta) = h
    '''
    A = LO(delta/n_T, 1.0/n_g)
    F = np.zeros(A.n_g)
    F[A.index(f)] = 1.0
    H = np.zeros(A.n_g)
    H[A.index(h)] = 1.0
    for t in range(n_T):
        F = A.matvec(F)
        F /= F.max()
        H = A.rmatvec(H)
        H /= H.max()
    p = F*H
    p *= A.n_g/p.sum()
    return p,A,F,H
    
    
def work():
    import matplotlib.pyplot as plt  # must be after mpl.use
    fig = plt.figure('conditional density')
    ax_log = fig.add_subplot(2,1,1)
    ax = fig.add_subplot(2,1,2)
    delta = .1
    f = .95
    h = .05
    for n_g in (1000, 5000, 10000, 25000, 50000, 100000):
        for n_T in (200,):
            p,A,F,H = conditional(delta, f, h, n_T, n_g)
            ax_log.semilogy(
                A.bins, p, label=r'$n_T,n_g={0},{1}$'.format(n_T,n_g))
            ax.plot(
                A.bins, p, label=r'$n_T,n_g={0},{1}$'.format(n_T,n_g))
            # fig_n = plt.figure('n_T={0}, n_g={1}'.format(n_T,n_g))
            # axn = fig_n.add_subplot(1,1,1)
            # axn.semilogy(A.bins, F, label=r'$F$')
            # axn.semilogy(A.bins, H, label=r'$H$')
            # axn.semilogy(A.bins, H*F, label=r'$FH$')
            # axn.legend()
    ax.legend()
    ax_log.legend()
    plt.show()
    return 0

if __name__ == "__main__":
    import sys
    rv = 0
    if len(sys.argv) == 1 or sys.argv[1] == 'test':
        rv = _test()
    if sys.argv[1] == 'work':
        rv = work()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
