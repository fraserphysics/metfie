"""first.py: Code for eigenfunctions of integral operators for first
order Markov process with 2-d states.  A state is an ordered pair
(g,h).  g is the position and h is the derivative.  An allowed
sequential pair of states ((g_0,h_0),(g_1,h_1)) satisfies:

\sqrt{24(u-g_0)} \geq \frac{\Delta_g}{\Delta_y} + 6\Delta_y \geq h_0

\sqrt{24(u-g_1)} \geq h_1 \geq \frac{\Delta_g}{\Delta_y} - 6\Delta_y

The first pair of inequalities constrains g_1 via g_1 - g_0 = \Delta_g,
and the second pair of inequalities constrains h_1.  See equations
UL_g and UL_h in notes.tex for the more complicated bounds that hold
when (u-g_0) \leq 36 \Delta_y^2.
"""
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg
class LO(scipy.sparse.linalg.LinearOperator):
    ''' Custom version of LinearOperator that implements A*x for the
    adjacency matrix definied by the parameters:

    u   The upper bound for g (-u is the lower bound).  0.01 is a
        typical value.
    dy  The step between y_n and y_{n+1}.  0.001 is typical
    n_g The number of quantization steps for g
    n_h The number of quantization steps for h

    Note: Cython code for this class in first_c.pyx
    
    >>> u = .0001; dy = .002; n_g = 20; n_h = 40
    >>> A = LO( u, dy, n_g, n_h)
    >>> print('n_states=%d'%(A.n_states,))
    n_states=527
    >>> n_max = 6; step = 0.5; _min = -1.0; _max = 1.0
    >>> f2i = lambda f: A.f2I(f, n_max, step, _min, _max)
    >>> for x in np.arange(-1,1.1,.3):
    ...    print('%4.1f %d %5.2f'%((x,) + f2i(x)))
    -1.0 0 -1.00
    -0.7 1 -0.50
    -0.4 1 -0.50
    -0.1 2  0.00
     0.2 2  0.00
     0.5 3  0.50
     0.8 4  1.00
     1.1 4  1.00
    >>> for i in range(*A.fi_range(-1.1, 2, .5, -1.0)):
    ...    print('%d  %5.2f'%(i,i*.5-1.0))
    0  -1.00
    1  -0.50
    2   0.00
    3   0.50
    4   1.00
    5   1.50
    
    These tests of symmetry are vacuous for the way I do rmatvec().
    >>> val,v = A.power(small=1e-6)
    >>> u = A.symmetry(v)
    >>> w = A.symmetry(u)
    >>> print('norm(v)=%8.6f norm(v-S^2(v))=%8.6f'%(LA.norm(v), LA.norm(v-w)))
    norm(v)=1.000000 norm(v-S^2(v))=0.000000
    >>> wal, w = A.power(small=1e-6, op=A.rmatvec)
    >>> print('delta val=%8.6f delta vec=%8.6f'%(abs(val-wal), LA.norm(u-w)))
    delta val=0.000000 delta vec=0.000000

    '''
    def h_lim(self, g):
        '''Calculates the maximum possible slope h at position g.
        '''
        d_g = self.g_max - g
        if d_g < 0:
            assert (-d_g)/self.g_max < 1e-10
            return 0.0
        return np.sqrt(24*(d_g))
    def ab(self, # LO instance
           low,  # Lower limit of h range
           high, # Upper limit of h range
           G     # Index of g, ie column number
           ):
        '''Return range of allowed state indices for given G and
        low <= h <= high
        '''
        s_i, s_f = self.G2state[G]  # Index of first and last states for G
        h_i, h_f = self.G2h_list[G] # Corresponding h values
        if s_i == s_f:
            assert h_i == h_f
            if low <= h_i and h_f <= high:
                return s_i, s_i+1           # Single state in image
            else:
                return -1, -2               # No states allowed
        dh_ds = (h_f-h_i)/(s_f-s_i)
        if low <= h_i:
            a = s_i
        else:
            a = s_i + int((low-h_i)/dh_ds)
        if high >= h_f:
            b = s_f + 1
        else:
            b = s_i + int((h_f - high)/dh_ds) + 1
        return a,b
    def fi_range(self, # LO instance
                 low, high, step, floor):
        '''Return pair of integers Low, High that satisfy:
        Low <= (low-floor)/step < Low+1
        High-1 < (high-floor)/step <= High

        Used in self.s_bounds and self.allowed to provide integer ranges of
        G and H.
        '''
        if high < floor or high < low:
            return 0,0 # No allowed range
        Low = max(0,int(np.floor((low-floor)/step)))
        High = int(np.ceil((high-floor)/step))
        assert Low <= High
        if Low == High:
            High += 1 # Special case low=high and (low-floor)/step is int
        return Low,High

    def s_bounds(
            self,          # LO instance
            i,             # index of state in self.state_list
            ):
        '''Find legal successors of state i and save representations
        of them in the self.*bounds* arrays.
        '''
        g_0, h_0, G_0, H_0 = self.state_list[i]
        # Calculate float range of allowed g values
        if g_0 > self.g_max - 6*self.dy**2:
            U_g = self.g_max
        else:
            U_g = min(self.g_max, g_0 + self.dy*self.h_lim(g_0) - 6*self.dy**2)
        L_g = max(self.g_min,g_0 + h_0*self.dy - 6*self.dy**2)
        if L_g > U_g:
            return 0
        bounds_a = np.zeros(self.n_g, dtype=np.int32)
        bounds_b = np.zeros(self.n_g, dtype=np.int32)
        len_bounds = 0
        n_pairs = 0
        G_i, G_f = self.fi_range(L_g, U_g, self.g_step, self.g_min)
        for G_1 in range(G_i, G_f+1): # +1 because f(G_0) <= g_0 < f(G_0+1)
            g_1 = self.g_min + G_1*self.g_step
            if G_1 >= len(self.G2state):
                break
            U_h = self.h_lim(g_1)
            L_h = max( (g_1 - g_0)/self.dy - 6 * self.dy, -U_h)
            a,b = self.ab(L_h, U_h, G_1)
            if b >= a:
                bounds_a[len_bounds] = a
                bounds_b[len_bounds] = b
                len_bounds += 1
                n_pairs += b - a
        self.bounds_a[i] = np.array(bounds_a[:len_bounds])
        self.bounds_b[i] = np.array(bounds_b[:len_bounds])
        assert n_pairs >0,'''No successors:
 g_frac=%5.3f, h_frac=%5.3f L_g=%9.2e U_g=%9.2e, G_i=%2d G_f=%2d'''%(
g_0/self.g_max, h_0/self.h_lim(g_0), L_g, U_g, G_i, G_f)
        return n_pairs
    def allowed(self):
        '''Calculate the allowed states.
        '''
        ep_h = self.h_step*1e-10 # Fudge for h assertions in loop
        ep_g = self.g_step*1e-10 # Fudge for g
        self.state_list = []
        self.state_dict = {}
        self.G2h_list = []  # G2h_list[G] is the allowed interval in h
        self.G2state = []   # G2state[G] is the corresponding pair of states
        First = True
        for G in range(self.n_g):
            g = self.g_min + G*self.g_step
            assert self.g_max + ep_g >= g and g + ep_g >= self.g_min
            h_max = self.h_lim(g)
            h_min = -h_max
            H_i, H_f = self.fi_range(h_min, h_max, self.h_step, self.h_min)
            if H_i == 0:
                H_i = 1                # Zero measure special case
            if H_i >= H_f:
                continue               # No alowed states for this g
            s_i = len(self.state_list) # First allowed state for this g
            for H in range(H_i, H_f):
                h = self.h_min + H*self.h_step
                self.state_dict[(G,H)] = (g,h,len(self.state_list))
                self.state_list.append((g,h,G,H))
            s_f = len(self.state_list) - 1 # Last allowed state for this g
            h_i,h_f = (self.h_min + H*self.h_step for H in (H_i,H_f-1))
            assert (
                h_i < h_min+ep_h
                and h_min-ep_h <= h_i + self.h_step)
            assert (
                h_f < h_max+ep_h
                and h_max-ep_h < h_f + self.h_step)
            self.G2h_list.append(np.array((h_i, h_f), dtype=np.float64))
            self.G2state.append(np.array((s_i, s_f), dtype=np.int32))
        self.n_states = len(self.state_list)
        return
    def pairs(self):
        '''Calculate allowed sequential pairs of states'''
        n_states = self.n_states
        self.shape = (n_states, n_states)
        self.n_pairs = 0
        self.bounds_a = np.empty((n_states), np.object)
        self.bounds_b = np.empty((n_states), np.object)
        childless = 0
        for i in range(n_states):
            # Commenting out the following call reduces LO build time
            # (n_g=500, n_h=500) from 9.8 seconds to 0.3 seconds.
            self.n_pairs += self.s_bounds(i) # n_successors
        return
    def __init__(self,              # LO instance
                 u,                 # Upper bound
                 dy,                # Change in y
                 n_g,               # Number of steps in position g
                 n_h,               # Number of steps in slope h
                 skip_pairs=False   # Call self.pairs if False
                 ):
        self.eigenvector = None
        self.eigenvalue = None
        self.dtype = np.dtype(np.float64)
        self.dy = dy
        self.n_g = n_g
        self.g_max = u
        self.g_min = -u
        self.g_step = 2*u/(n_g-1)
        self.h_max = self.h_lim(self.g_min)
        self.h_step = 2*self.h_max/(n_h-1)
        self.h_min = -self.h_max - self.h_step
        self.n_h = n_h + 1 # Bottom value of h not used
        if self.h_step > 2*dy:
            print('WARNING: h_step=%f, dy=%f'%(self.h_step, dy))

        self.allowed()
        if not skip_pairs: self.pairs()
        return
    def symmetry(self, v, w=None):
        ''' Interchange h and -h.  Put result in w if passed, otherwise
        allocate new array for return value.
        '''
        if w == None:
            w = np.empty(self.n_states)
        for i in range(self.n_states):
            g,h1,G,H1 = self.state_list[i]
            h_lim = self.h_lim(g)
            H2 = int(np.floor((-h1-self.h_min)/self.h_step - .5))
            H = H2
            h2 = self.h_min + self.h_step*H2
            if h2+self.h_step <= -h_lim:
                H = H2 + 1
            if h2 >= h_lim:
                H = H2 - 1
            g,h,j = self.state_dict[(G,H)]
            w[j] = v[i]
        return w
    def gh(self,      # LO instance
           m_g=None,  # Number of sample points in g
           m_h=None   # Number of sample points in h
           ):
        '''Return two vectors of coordinates (g and h) that span the range
        of allowed values.  Useful for plotting state vectors.
        '''
        if m_g == None:
            m_g = self.n_g
        if m_h == None:
            m_h = self.n_h
        def vector(_min, step, n):
            v = np.arange(_min, (n+2)*step, step) # 2 extra to be safe
            return v[:n]
        return (vector(self.g_min, self.g_step*self.n_g/m_g, m_g),
                vector(self.h_min, self.h_step*self.n_h/m_h, m_h))
    def vec2z(self,      # LO instance
                v,       # vector  len(v.reshape(-1)) = self.n_states
                g=None,  # 1-d array of g values at sample points
                h=None,  # 1-d array of h values at sample points
                floor=0
                ):
        ''' Use spline to estimate z(g[i,j],h[i,j]) on the basis of v.
        Return 2-d arrays of z values, g values, and h values.
        '''
        v = v.reshape(-1)
        assert len(v) == self.n_states
        if g == None:
            # This case simply returns v + floor at legal (G,H) pairs
            # and floor elsewhere.
            assert h == None
            rv = np.zeros((self.n_g, self.n_h)) + floor
            for i in range(self.n_states):
                g,h,G,H = self.state_list[i]
                rv[G,H] = max(v[i],floor)
            return rv
        self.spline(v)
        G, H = np.meshgrid(g, h)
        assert G.shape == ( len(h), len(g) )
        rv = np.fmax(self.bs.ev(G.reshape(-1), H.reshape(-1)), floor)
        def _test(i_h,i_g):
            '''Return True if out of bounds, False otherwise.
            '''
            g_ = G[i_h,i_g]
            h_ = H[i_h,i_g]
            if g_ > self.g_max or g_ < self.g_min:
                return True
            h_lim = self.h_lim(g_)
            if h_ > h_lim or h_ < - h_lim:
                return True
            else: return False
        I_h, I_g = G.shape
        rv = rv.reshape(G.shape)
        for i_h in range(I_h):
            for i_g in range(I_g):
                if _test(i_h,i_g):
                    rv[i_h,i_g] = floor
        return rv.T #,G,H
    def matvec(self, v, rv=None):
        ''' Apply linear operator, self, to v and return.  Use array
        rv if passed, otherwise allocate rv.
        '''
        if rv == None:
            rv = np.zeros(self.n_states)
        else:
            rv[:] = 0.0
        dgdh = self.g_step*self.h_step
        for i in range(self.n_states):
            x = v[i]
            for j in range(len(self.bounds_a[i])):
                a = self.bounds_a[i][j]
                b = self.bounds_b[i][j]
                rv[a:b] += x*dgdh
        return rv
    def rmatvec(self, v, rv=None):
        ''' Apply transpose of linear operator, self, to v and return.
        Use matvec and symmetry to implement.  Use array rv if passed,
        otherwise allocate rv.
        '''
        if rv == None:
            rv = np.empty(self.n_states)
        self.symmetry(v, rv)              # S(v) -> rv
        temp = self.matvec(rv)            # L(rv) -> temp
        rv = self.symmetry(temp, rv)      # S(temp) -> rv
        return rv
    def __mul__(self,x):
        x = np.asarray(x)

        if x.ndim == 1:
            return self.matvec(x)
        elif x.ndim == 2 and x.shape[1] == 1:
            return self.matvec(self, x[0])
        else:
            raise ValueError('expected x.shape = (n,) or (n,1)')
    def power(self, n_iter=1000, small=1.0e-5, v=None, op=None, verbose=False):
        '''Calculate largest eigevalue and corresponding eigenvector of op.
        '''
        if op == None: op = self.matvec
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
        self.eigenvalue = s
        self.eigenvector = v_old
        return s,v_old
    def xyz(self, v=None):
        '''allocate array and save g[i], h[i], eigenvector[i] for
         i < n_states.  Used for building and evaluating spline
         approximations to eigenvector.
         '''
        if v == None:
            assert self.eigenvector != None
            v = self.eigenvector
        else:
            v = v.reshape(-1)
            assert len(v) == self.n_states
        x = np.empty((3,self.n_states))
        for i in range(self.n_states):
            x[0,i],x[1,i] = self.state_list[i][:2] # x[0] = g, x[1] = h
            x[2,i] = v[i]
        return x
    def spline(self, v=None):
        ''' Build spline approximation to eigenvector or v.
        '''
        if v == None:
            assert self.eigenvector != None
            v = self.eigenvector
        else:
            v = v.reshape(-1)
            assert len(v) == self.n_states
        
        '''Create arrays g, h, and z for RectBivariateSpline.  g and h
        are 1-d arrays of floats with integer indices G and H.  For
        legal pairs, (G,H), self.state_dict maps to i(G,H) an index
        of v.  So, z[G,H] <- v[i(G,H)] for legal pairs (G,H).  For
        illegal pairs, the code uses linear extrapolation.  The
        extrapolation makes the splines smoother in the legal region
        than they would be if the illegal region were set to zero.
        '''
        z = np.ones((self.n_g, self.n_h))*(-1)
        g,h = self.gh()
        last_z = 0
        H_0 = int(self.n_h/2)
        for G in range(self.n_g):
            for H in range(H_0,-1,-1):
                if (G,H) in self.state_dict:
                    i = self.state_dict[(G,H)][-1]
                    z[G,H] = v[i]
                    dzdh = last_z - z[G,H]
                    last_z = z[G,H]
                else:
                    z[G,H] = last_z - dzdh
            for H in range(H_0,self.n_h):
                if (G,H) in self.state_dict:
                    i = self.state_dict[(G,H)][-1]
                    z[G,H] = v[i]
                    dzdh = z[G,H] - last_z
                    last_z = z[G,H]
                else:
                    z[G,H] = last_z + dzdh
        from scipy.interpolate import RectBivariateSpline
        self.bs = RectBivariateSpline(g, h, z, kx=3, ky=3)
        return
    def diff(self,     #LO instance
             g,        #List(like) of g values
             h,        #List(like) of h values
             v,        #eigenfunction at (g,h)
             rv=False  #Return the vector difference 
                    ):
        ''' Calculate and return error of spline self.eigenvector at
        points g[i],h[i]
        '''

        # Fit a spline and evaluate
        self.spline()          # Build spline of self
        w = self.bs.ev(g,h)    # Evaluate spline at points of v
        w /= LA.norm(w)        # Make w a unit vector
        u = v-w
        d = LA.norm(u)
        if rv: return d, u
        else: return d
class LO_step(LO):
    '''Variant of LO that uses g_step and h_step rather than n_h and n_g
    as arguments to __init__
    '''
    def __init__(self,              # LO_step instance
                 u,                 # Upper bound
                 dy,                # Change in y
                 g_step,            # Size of steps in position g
                 h_step,            # Size of steps in slope h
                 skip_pairs=False   # Call self.pairs if False
                 ):
        if h_step > 2*dy:
            print('WARNING: h_step=%f, dy=%f'%(h_step, dy))
        self.eigenvector = None
        self.eigenvalue = None
        self.dtype = np.dtype(np.float64)
        self.dy = dy
        self.g_max = u
        self.g_min = -u
        self.g_step = g_step
        self.h_step = h_step
        self.h_max = self.h_lim(self.g_min)
        self.h_min = -self.h_max - self.h_step  # Bottom h value not used
        self.n_h = int(2 + 2*self.h_max/h_step)
        self.n_g = int(1 + 2*self.g_max/g_step)

        self.allowed()
        if not skip_pairs: self.pairs()
        return
def sym_diff(A, B):
        '''
        Calculate difference between eigenvectors of LO instances A and B.
        d = \sqrt{ \frac{1}{N_A + N_B} \left(
             \sum_{i=1}^{N_A} |A_i - interp(B,i)|^2 +
             \sum_{i=1}^{N_B} |B_i - interp(A,i)|^2 \right) }
        '''
        x = A.xyz()
        d1 = B.diff(x[0],x[1],x[2])
        x = B.xyz()
        d2 = A.diff(x[0],x[1],x[2])
        return (d1+d2)/2
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

#---------------
# Local Variables:
# mode: python
# End:
