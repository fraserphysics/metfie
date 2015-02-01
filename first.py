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

class LO_step(scipy.sparse.linalg.LinearOperator):
    '''Custom version of LinearOperator that implements A*x for the
    adjacency matrix definied by the parameters:

    d   The scale, that is u/(dy^2)

    Note: Cython code for this class in first_c.pyx uses:
    self.n_states, self.n_pairs self.G2state, self.G2h_list, self.d,
    self.state_list, self.h_lim() self.g2G(), self.n_g, self.origin_g,
    self.g_step, self.h_step, self.bounds_a, self.bounds_b

    '''
    def h2H(self, h):
        ''' Return integer index H for float coordinate h
        '''
        H = int(np.floor(h - self.origin_h)/self.h_step)
        assert H >= 0
        assert H < self.n_h
        return H
    def g2G(self, g):
        ''' Return integer index G for float coordinate g
        '''
        G = min(self.n_g-1, int(np.floor(g - self.origin_g)/self.g_step))
        assert G >= 0
        assert G < self.n_g, 'g={0:f} d={1:f} G={2:d} n_g={3:d}'.format(
            g, self.d, G, self.n_g)
        return G
    def h_lim(self, g):
        '''Calculates the maximum possible slope, h, at position g.  Boundary
        equation is g(h) = d -h^2/24.

        '''
        d_g = self.d - g
        if d_g < 0:
            assert (-d_g)/self.d < 1e-10
            return 0.0
        return np.sqrt(24*(d_g))
    def ab(self,        # LO instance
           h_edge,      # Lower limit of h range
           g_min,       # Lowest g in image
           g_intercept, # g value on slope 1 line where it crosses h=0
           G,           # Index of row
           ):
        '''Return range of allowed state indices for given h interval and G.
        Including any integer H whose image in h overlaps [low,high]
        will include any H whose (H,G) cell intersects the domain.
        '''
        if G == 0:
            g_ = -self.d
        else:
            g_ = self.origin_g + G*self.g_step
        g = max(g_,g_min)
        high = self.h_lim(g)                           # Top of h range
        low = max(-high, max(h_edge, (g-g_intercept))) # Bottom of h range
        
        s_i, s_f = self.G2state[G]  # Index of first and last+1 states in domain for G
        H_lim = self.G2h_list[G]    # Number of steps from 0 to edge
        assert s_f - s_i == H_lim*2
        # Calculate integer steps from left edge
        Low = int(np.floor(low/self.h_step)) + H_lim
        High = int(np.ceil(high/self.h_step)) + H_lim

        # Return clipped offsets from s_i
        return max(s_i, s_i + Low), min(s_f, s_i + High)

    def boundary_image(self, h_1, g_1, g_1_top):
        '''Calculations for images when the upper left corner is outside of
        the quadratic boundary
        '''
        _in_ = lambda x : x[1]<self.d-x[0]**2/24 # Is (h,g) inside boundary parabola?
        def solve(a, b, corners):
            '''Return intersection of boundary and line between a and b
            '''
            a,b = [corners[x] for x in (a,b)]
            from scipy.optimize import brentq
            def f(s):
                '''Difference between g=c[1] and boundary at h=c[0]'''
                c = s*a + (1-s)*b
                return c[1] - (self.d - c[0]**2/24)
            s = brentq(f, 0,1)
            rv = s*a + (1-s)*b
            return rv
        
        # Search over cell corners and intersections of boundary with cell edges
        h_edge = self.h_lim(-self.d)   # Right extreme
        g_min = self.d                 # Top extreme
        g_intercept = -self.d - h_edge # Low extreme
        points = []
        corners = [np.array(x, dtype=np.float64) for x in [
                (h_1,g_1),
                (h_1+self.h_step, g_1+self.h_step),
                (h_1+self.h_step, g_1_top+self.h_step),
                (h_1,g_1_top)]]
        edges = [(i-1,i) for i in range(4)]
        for corner in corners:
            if _in_(corner):
                points.append(corner)
        for a,b in edges:
            if int(_in_(corners[a])) + int(_in_(corners[b])) == 1:
                points.append(solve(a,b,corners))
        assert len(points) >= 3
        for h,g in points:
            if h < h_edge:
                h_edge = h
            if g-h > g_intercept:
                g_intercept = g-h
            if g < g_min:
                g_min = g
        return h_edge, g_min, g_intercept
    def s_bounds(
            self,          # LO instance
            i,             # index of state in self.state_list
            ):
        '''Find legal successors of state i and save representations
        of them in the self.*bounds* arrays.

        Notes:

        * cell[i] is a rectangle with sides h_step and g_step
        * state[i] is the intersection of cell[i] with the domain
        * A((h,g)) = (h-12, g+h-6) is an affine function that maps
            rectangles to parallelograms and maps a point z_0 in the
            domain to z_1 the apex of it's pie slice image.
        * The parabolic boundary is invariant under A
        '''
        
        H,G = self.state_list[i] # Indices of lower left corner of cell[i]
        h_0 = H*self.h_step      # Float coordinate of left side of cell[i]
        g_0 = self.origin_g + G*self.g_step # Bottom of cell[i]
        g_0_top = g_0 + self.g_step
        # g_0_top is the top of cell.  Grid is laid out so that
        # state[i] goes all the way to g_0_top for some value of h.
        if G == 0:
            g_0 = -self.d # Cell[i] may extend below -d
        h_1 = h_0 - 12
        g_1, g_1_top = (g+h_0-6 for g in (g_0, g_0_top))
        # (h_1,g_1) and (h_1,g_1_top) are the images under A of the
        # left side of cell[i] (perhaps with g_0 truncated to -d)

        if g_1_top < self.d - h_1*h_1/24: # Inside quadratic boundary
            h_edge = h_1
            g_min = g_1
            g_intercept = g_1_top - h_1
        else:
            h_edge, g_min, g_intercept = self.boundary_image(h_1, g_1, g_1_top)
                        
        # Allocate arrays for result that are at least big enough
        bounds_a = np.zeros(self.n_g, dtype=np.int32)
        bounds_b = np.zeros(self.n_g, dtype=np.int32)
        len_bounds = 0
        n_pairs = 0

        G_i = self.g2G(max(-self.d,g_min)) # Lowest G index in image
        for G_ in range(G_i, self.n_g):
            a,b = self.ab(h_edge, g_min, g_intercept, G_)
            if a >= b:
                break # End of pie slice
            bounds_a[len_bounds] = a
            bounds_b[len_bounds] = b
            len_bounds += 1
            n_pairs += b - a
        # Trim unused parts of arrays
        self.bounds_a[i] = np.array(bounds_a[:len_bounds])
        self.bounds_b[i] = np.array(bounds_b[:len_bounds])
        assert n_pairs > 0
        return n_pairs
    
    def allowed(self):
        '''Calculate the allowed states and other properties.
        '''
        # The folowing 4 objects are attached to self in this method
        state_list = []      # state_list[i] = (H,G)
        self.state_dict = {} # state_dict[(H,G)] = index of state_list
        G2h_list = []        # G2h_list[G] = number of h_steps from zero to edge
        G2state = []         # G2state[G] = (min_state, max_state+1)

        for G in range(self.n_g):
            if G == 0:
                g = -self.d
            else:
                g = G*self.g_step + self.origin_g
            # H_lim = number of different values of h2H(h): 0 < h < h_lim(g)
            H_lim = int(np.ceil(self.h_lim(g)/self.h_step))
            G2h_list.append(H_lim)
            s_i = len(state_list) #index of first allowed state for this g
            for H in range(-H_lim, H_lim):
                self.state_dict[(H,G)] = len(state_list)
                state_list.append((H,G))
            G2state.append(np.array((s_i,len(state_list)),dtype=np.int32))
        self.n_states = len(state_list)
        self.state_list = np.array(state_list, dtype=np.int32)
        self.G2h_list = np.array(G2h_list, dtype=np.int32)
        self.G2state = np.array(G2state, dtype=np.int32)
        return
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
    def __init__(self,              # LO_step instance
                 d,                 # Scale = u/(dy^2)
                 g_step,            # Size of steps in position g
                 h_step,            # Size of steps in slope h
                 skip_pairs=False   # Call self.pairs if False
                 ):
        self.dtype = np.dtype(np.float64)
        self.d = d
        self.g_step = g_step
        self.h_step = h_step
        h_max = self.h_lim(-d)

        # Make h=0 align with quantization boundary
        n = int(np.floor(h_max/h_step))
        if n*h_step < h_max:
            n += 1
        self.origin_h = -n*h_step
        self.n_h = 2*n
        
        # Make g=d align with quantization boundary
        n = int(np.floor(d*2.0/g_step))
        if n*g_step < 2*d:
            n += 1
        self.origin_g = d-n*g_step
        self.n_g = n

        self.allowed()
        if not skip_pairs:
            self.pairs()
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
    def conj(self,i):
        ''' Conjugate: Get index for state with -h
        '''
        H1, G = self.state_list[i]
        H = -(H1+1)
        assert (H,G) in self.state_dict
        return self.state_dict[(H,G)]
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
        h_max = self.h_lim(-self.d)
        return (np.linspace(-self.d, self.d, m_g, endpoint=False),
                np.linspace(-h_max, h_max, m_h, endpoint=False))
    def vec2z(self,      # LO instance
                v,       # vector with component for each state
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
            rv = np.zeros((self.n_h, self.n_g)) + floor
            for i in range(self.n_states):
                H,G = self.state_list[i]
                rv[H+self.n_h/2,G] = max(v[i],floor)
            return rv
        self.spline(v)
        H,G = np.meshgrid(g, h)
        assert G.shape == ( len(h), len(g) )
        rv = np.fmax(self.bs.ev(H.reshape(-1)), G.reshape(-1), floor)
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
        return rv.T
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
            x = v[i]*dgdh
            for i_g in range(len(self.bounds_a[i])):
                # i_g is index for a line of constant g
                a = self.bounds_a[i][i_g]
                b = self.bounds_b[i][i_g]
                rv[a:b] += x # a:b are states that correspond to a range of h values
        return rv
    def rmatvec(self, v, rv=None):
        '''Use array rv if passed, otherwise allocate rv.

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
            raise RuntimeError #FixMe: Use new state_list format'
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
        legal pairs, (H,G), self.state_dict maps to i(H,G) an index
        of v.  So, z[H,G] <- v[i(H,G)] for legal pairs (H,G).  For
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
                if (H,G) in self.state_dict:
                    i = self.state_dict[(H,G)]
                    z[H_i,G] = v[i]
                    dzdh = last_z - z[H_i,G]
                    last_z = z[H_i,G]
                else:
                    z[H_i,G] = last_z - dzdh
                # Block for 0 > h = -(H+1)*h_step, 0 <= H_i < H_0
                H_ = -(H+1)
                H_i = H_0 + H_
                if (G,H_) in self.state_dict:
                    i = self.state_dict[(H_,G)]
                    z[H_i,G] = v[i]
                    dzdh_ = last_z_ - z[H_i,G]
                    last_z_ = z[H_i,G]
                else:
                    z[H_i,G] = last_z_ - dzdh_
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
            raise RuntimeError#,'FixMe: Use new state_list format'
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
