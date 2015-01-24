"""first.py: Python3 code for eigenfunctions of integral operators for
first order Markov process with 2-d states.  A state is an ordered
pair (g,h).  g is the position and h is the derivative.  An allowed
sequential pair of states ((g_0,h_0),(g_1,h_1)) satisfies:

\sqrt{24(d-g_0)} \geq \Delta_g + 6 \geq h_0

\sqrt{24(d-g_1)} \geq h_1 \geq \Delta_g - 6

The first pair of inequalities constrains g_1 via g_1 - g_0 = \Delta_g,
and the second pair of inequalities constrains h_1.  See equations
UL_g and UL_h in notes.tex for the more complicated bounds that hold
when (d-g_0) \leq 36.

"""
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg
class LO(scipy.sparse.linalg.LinearOperator):
    ''' Custom version of LinearOperator that implements A*x for the
    adjacency matrix definied by the parameters:

    d   The scale u/(dy^2)
    n_g The number of quantization steps for g
    n_h The number of quantization steps for h

    Note: Cython code for this class in first_c.pyx
    
    >>> d = 25; n_g = 20; n_h = 40
    >>> A = LO( d, n_g, n_h)
    >>> print('n_states=%d'%(A.n_states,))
    n_states=530
    >>> n_max = 6; step = 0.5; _min = -1.0; _max = 1.0

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
        d_g = self.d - g
        if d_g < 0:
            assert (-d_g)/self.d < 1e-10
            return 0.0
        return np.sqrt(24*(d_g))
    def ab(self, # LO instance
           low,  # Lower limit of h range
           high, # Upper limit of h range
           G     # Index of g, ie column number
           ):
        '''Return range of allowed state indices for given G.  Any H whose
        image in h overlaps [low,high] should be included.

        '''
        s_i, s_f = self.G2state[G]  # Index of first and last+1 states for G
        H_lim = self.G2h_list[G]    # Corresponding limit on H
        assert s_f > s_i
        H_low = H_lim + int(np.floor(low/self.h_step))
        H_high = H_lim + int(np.floor(high/self.h_step)) + 1
        return max( s_i, s_i + H_low),min( s_f, s_i + H_high)
    def s_bounds(
            self,          # LO instance
            i,             # index of state in self.state_list
            ):
        '''Find legal successors of state i and save representations
        of them in the self.*bounds* arrays.
        '''
        g_0, h_0, G_0, H_0 = self.state_list[i]
        # Calculate float range of allowed g values
        if g_0 > self.d - 6:
            U_g = self.d
        else:
            U_g = min(self.d, g_0 + self.h_lim(g_0) - 6)
        L_g = max(-self.d, g_0 + h_0 - 6)
        assert L_g <= U_g
        bounds_a = np.zeros(self.n_g, dtype=np.int32)
        bounds_b = np.zeros(self.n_g, dtype=np.int32)
        len_bounds = 0
        n_pairs = 0
        # Calculate range of G in image
        G_i = max(0, int(np.floor( (L_g+self.d)/self.g_step )))
        G_f = min(self.n_g, 1+int(np.floor( (U_g+self.d)/self.g_step )))
        
        for G_1 in range(G_i, G_f):
            g_1 = -self.d + G_1*self.g_step # Float g of image
            U_h = self.h_lim(g_1)
            L_h = max( (g_1 - g_0) - 6 , -U_h)
            # Given g_1, h is in [L_h,U_h)
            a,b = self.ab(L_h, U_h, G_1)
            if b > a:
                bounds_a[len_bounds] = a
                bounds_b[len_bounds] = b
                len_bounds += 1
                n_pairs += b - a
        self.bounds_a[i] = np.array(bounds_a[:len_bounds])
        self.bounds_b[i] = np.array(bounds_b[:len_bounds])
        assert n_pairs > 0
        return n_pairs
    def allowed(self):
        '''Calculate the allowed states.
        '''
        self.state_list = []# state_list[i] = (g,h,G,H)
        self.state_dict = {}# state_dict[(G,H)] = index of state_list
        G2h_list = []       # G2h_list[G] = number of h_steps from zero to edge
        self.G2state = []   # G2state[G] is the corresponding pair of states
        dh = self.h_step    # Abbreviation
        for G in range(self.n_g):
            g = -self.d + G*self.g_step
            assert self.d > g and g >= -self.d
            h_max = self.h_lim(g)
            H_lim = int(np.floor(h_max/dh) + 1)
            if (H_lim-1)*dh >= h_max or -H_lim*dh < dh-h_max:
                H_lim -= 1
            s_i = len(self.state_list) # First allowed state for this g
            for H in range(-H_lim, H_lim):
                h = H*dh
                self.state_dict[(G,H)] = len(self.state_list)
                self.state_list.append((g,h,G,H))
            G2h_list.append(H_lim)
            self.G2state.append(np.array((s_i, len(self.state_list)),
                                         dtype=np.int32))
        self.n_states = len(self.state_list)
        self.G2h_list = np.array(G2h_list, dtype=np.int32)
        self.G2state = np.array(self.G2state, dtype=np.int32)
        return
    def pairs(self):
        '''Calculate allowed sequential pairs of states'''
        n_states = self.n_states
        self.shape = (n_states, n_states)
        self.n_pairs = 0
        self.bounds_a = np.empty((n_states), np.object)
        self.bounds_b = np.empty((n_states), np.object)
        for i in range(n_states):
            # Put bounds in self.bounds_a and self.bounds_b
            self.n_pairs += self.s_bounds(i) # Most build time here
        return
    def __init__(self,              # LO instance
                 d,                 # Scale = u/(dy^2)
                 n_g,               # Number of steps in position g
                 n_h,               # Number of steps in slope h
                 skip_pairs=False   # Call self.pairs if False
                 ):
        assert n_h%2 == 0
        self.dtype = np.dtype(np.float64)
        self.d = d
        self.n_g = n_g
        self.g_step = 2*d/n_g
        self.h_max = self.h_lim(-d)
        self.h_step = 2*self.h_max/n_h
        self.h_min = -self.h_max
        self.n_h = n_h
        self.allowed()
        if not skip_pairs: self.pairs()
        return
    def conj(self,i):
        ''' Conjugate: Get index for state with -h
        '''
        g,h,G,H1 = self.state_list[i]
        H = -(H1+1)
        assert (G,H) in self.state_dict
        return self.state_dict[(G,H)]
    def symmetry(self, v, w=None):
        ''' Interchange h and -h.  Put result in w if passed, otherwise
        allocate new array for return value.
        '''
        if w == None:
            w = np.empty(self.n_states)
        for i in range(self.n_states):
            w[i] = v[self.conj(i)]
        return w
    def calc_marginal(self):
        ''' Derive self.marginal from self.eigenvector.
        '''
        self.marginal = self.symmetry(self.eigenvector) * self.eigenvector
        self.marginal /= self.marginal.sum()*self.g_step*self.h_step
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
        return (np.linspace(-self.d, self.d, m_g, endpoint=False),
                np.linspace(self.h_min, self.h_max, m_h, endpoint=False))
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
                rv[G,H+self.n_h/2] = max(v[i],floor)
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
            if g_ > self.d or g_ < -self.d:
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
                # j is index for a line of constant g
                a = self.bounds_a[i][j]
                b = self.bounds_b[i][j]
                rv[a:b] += x*dgdh # a:b corresponds to a range of h values
        return rv
    def rmatvec(self, v, rv=None):
        '''Use array rv if passed, otherwise allocate rv.

        '''
        if rv == None:
            rv = np.zeros(self.n_states)
        else:
            rv[:] = 0.0
        dgdh = self.g_step*self.h_step
        for i in range(self.n_states): # assign to rv[i]
            for i_g in range(len(self.bounds_a[i])):
                # i_g is index for a line of constant g
                a = self.bounds_a[i][i_g]
                b = self.bounds_b[i][i_g]
                rv[i] += v[a:b].sum()*dgdh # a:b is for a range of h values
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
            x[0,i],x[1,i] = self.state_list[i][:2] # x[0] = g, x[1] = h
            x[2,i] = v[i]
        return x
    def spline(self, v=None):
        ''' Build spline approximation to eigenvector or v.
        '''
        if v == None:
            assert 'eigenvector' in vars(self)
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
        z = np.empty((self.n_g, self.n_h))
        g,h = self.gh()
        last_z = 0
        last_z_ = 0
        H_0 = int(self.n_h/2)
        for G in range(self.n_g):
            for H in range(H_0):
                # Block for 0 <= h = H*h_step, H_0 <= H_i < 2*H_0
                H_i = H_0 + H  # Index
                if (G,H) in self.state_dict:
                    i = self.state_dict[(G,H)]
                    z[G,H_i] = v[i]
                    dzdh = last_z - z[G,H_i]
                    last_z = z[G,H_i]
                else:
                    z[G,H_i] = last_z - dzdh
                # Block for 0 > h = -(H+1)*h_step, 0 <= H_i < H_0
                H_ = -(H+1)
                H_i = H_0 + H_
                if (G,H_) in self.state_dict:
                    i = self.state_dict[(G,H_)]
                    z[G,H_i] = v[i]
                    dzdh_ = last_z_ - z[G,H_i]
                    last_z_ = z[G,H_i]
                else:
                    z[G,H_i] = last_z_ - dzdh_
        from scipy.interpolate import RectBivariateSpline
        self.bs = RectBivariateSpline(g, h, z, kx=3, ky=3)
        return
    def set_eigenvector(
            self,        # LO instance
            A,           # Other LO instance or np array
            exact=False  # Use np array A without interpolation
    ):
        '''For self.eigenvector, use eigenvector of other LO instance or use
        array.

        '''
        if exact:
            assert A.shape == (self.n_states,)
            self.eigenvector = A
            return
        x = np.empty((2,self.n_states))
        for i in range(self.n_states):
            x[0,i],x[1,i] = self.state_list[i][:2] # x[0] = g, x[1] = h
        A.spline()
        self.eigenvector = A.bs.ev(x[0],x[1])
        self.eigenvector /= LA.norm(self.eigenvector)
        self.spline() # Fit to new self.eigenvector
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
        if not 'bs' in vars(self):
            self.spline()      # Build spline of self
        w = self.bs.ev(g,h)    # Evaluate spline at points of v
        w /= LA.norm(w)        # Make w a unit vector
        u = v-w
        d = LA.norm(u)
        if rv: return d, u
        else: return d
def read_LO_step(filename, dirname='archive'):
    '''From an archive build an LO_step instance and read its eigenvector
    from disk.
    '''
    import pickle, os.path
    archive = pickle.load(open(os.path.join(dirname,filename),'rb'))
    d, g_step, h_step = archive['args']
    A = LO_step(d, g_step, h_step, skip_pairs=True)
    A.set_eigenvector(np.fromfile(archive['vec_filename']),exact=True)
    return A
class LO_step(LO):
    '''Variant of LO that uses g_step and h_step rather than n_h and n_g
    as arguments to __init__
    '''
    def __init__(self,              # LO_step instance
                 d,                 # Scale = u/(dy^2)
                 g_step,            # Size of steps in position g
                 h_step,            # Size of steps in slope h
                 skip_pairs=False   # Call self.pairs if False
                 ):
        self.dtype = np.dtype(np.float64)
        self.d = d
        self.h_max = self.h_lim(-d)
        self.h_min = -self.h_max
        self.n_h = 2 * int(self.h_max/h_step) # Ensure that self.n_h is even
        self.h_step = 2*self.h_max/self.n_h
        self.n_g = int(2*self.d/g_step)
        self.g_step = 2*d/self.n_g
        #assert self.h_step == h_step
        #assert self.g_step == g_step

        self.allowed()
        if not skip_pairs: self.pairs()
        return
    def archive(self, filename, more={}, dirname='archive'):
        import tempfile
        import pickle
        import os.path

        vec_file = tempfile.NamedTemporaryFile(
            prefix='%s.e_vec_'%filename,dir=dirname,delete=False)
        self.eigenvector.tofile(vec_file)
        dict_ = {
            'args':(float(self.d), float(self.g_step), float(self.h_step)),
            'vec_filename':vec_file.name}
        dict_.update(more)
        pickle.dump(dict_, open( os.path.join(dirname,filename), "wb" ),2 )
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
