"""first.py: Python3 code for eigenfunctions of integral operators for
first order Markov process with 1-d states.  Derived from ../first.py
"""
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg

class LO(scipy.sparse.linalg.LinearOperator):
    '''Custom version of LinearOperator that implements A*x and x*A for
    the adjacency matrix definied by the parameter T:

    The image of g_0 is [0,g_0+T).
    '''
    def g2G(self, g):
        ''' Return integer index G for float coordinate g
        '''
        G = min(self.n_g-1, np.floor(g - self.origin_g)/self.g_step)
        assert G >= 0
        assert G < self.n_g, 'g={0:f} d={1:f} G={2:d} n_g={3:d}'.format(
            g, self.d, G, self.n_g)
        return int(G)
    def G2g(self, G):
        '''Return float position corresponding to integer index G

        '''
        if G == 0:
            return -self.d
        return self.origin_g + G*self.g_step
    def ab(self,        # LO instance
           G,           # Index of g
           ):
        '''Return range of allowed sucessor state indices for given G.
        '''
        g = self.G2g(G)
        return 0, self.g2G(g+self.T)
    
    def allowed(self):
        '''Calculate the allowed states. The are all allowed
        '''
        pass
    def pairs(self):
        '''Calculate allowed sequential pairs of states'''
        n_states = len(self.state_list)
        self.shape = (n_states, n_states)
        self.n_pairs = 0
        self.bounds_a = np.empty((n_states), np.object)
        self.bounds_b = np.empty((n_states), np.object)
        for i in range(n_states):
            # Put bounds in self.bounds_a and self.bounds_b
            self.n_pairs += self.s_bounds(i) # Most build time here
        return
    def __init__(self,              # LO instance
                 T,                 # Slope times dt
                 g_step,            # Size of steps in position g
                 ):
        self.dtype = np.dtype(np.float64)
        self.T = T
        self.g_step = g_step

    def calc_marginal(self):
        ''' Derive self.marginal from self.eigenvector.
        '''
        self.marginal = self.symmetry(self.eigenvector) * self.eigenvector
        self.marginal /= self.marginal.sum()*self.g_step*self.h_step
    def matvec(self, v, rv=None):
        '''Calculate rv= A*v and return.  Use array rv if passed, otherwise
        allocate rv.
        '''
        if rv == None:
            rv = np.zeros(self.n_states)
        else:
            rv[:] = 0.0
        dgdh = self.g_step*self.h_step
        for i in range(self.n_states):
            x = v[i]*dgdh
            for i_g in range(len(self.bounds_a[i])):
                # i_g is index for a line of constant g
                a = self.bounds_a[i][i_g]
                b = self.bounds_b[i][i_g]
                rv[a:b] += x # a:b corresponds to a range of h values
        return rv
    def rmatvec(self, v, rv=None):
        '''Calculate rv = v*A and return.  Use array rv if passed, otherwise
        allocate rv.
        '''
        if rv == None:
            rv = np.zeros(self.n_states)
        else:
            rv[:] = 0.0
        dgdh = self.g_step*self.h_step
        for i in range(self.n_states): # add to rv[i]
            for i_g in range(len(self.bounds_a[i])):
                # i_g is index for a line of constant g
                a = self.bounds_a[i][i_g]
                b = self.bounds_b[i][i_g]
                # a:b are states that correspond to a range of h values
                rv[i] += v[a:b].sum()*dgdh
        return rv
    def __mul__(self,x):
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
            v_old = np.ones(self.n_states)
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
'''With n_states=%d and n_pairs=%d, finished power() at iteration %d
    ds=%g, dv=%g'''%(self.n_states, self.n_pairs, i, ds, dv))
        self.iterations = i
        self.eigenvalue = s
        self.eigenvector = v_old
        return s,v_old
    def xyz(self, v=None):
        '''allocate array and save g[i], h[i], eigenvector[i] for
         i < n_states.  Used for building and evaluating spline
         approximations to eigenvector.
         '''
        if v == None:
            assert 'eigenvector' in vars(self)
            v = self.eigenvector
        else:
            v = v.reshape(-1)
            assert len(v) == self.n_states
        x = np.empty((3,self.n_states))
        for i in range(self.n_states):
            raise RuntimeError #FixMe: Use new state_list format'
            x[0,i],x[1,i] = self.state_list[i][:2] # x[0] = g, x[1] = h
            x[2,i] = v[i]
        return x
        
def fractions2indices(sources, A):
    ''' Translate fractions (h_frac,g_frac) to integers (H,G).
    '''
    int_sources = []
    h_max = A.h_lim(-A.d)
    for frac_h,frac_g,iterations in sources:
        H = A.h2H(h_max*frac_h)
        G = A.g2G(A.d*frac_g)
        int_sources.append((H,G,iterations))
    return int_sources

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

#---------------
# Local Variables:
# mode: python
# End:
