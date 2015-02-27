"""first.py: Python3 code for eigenfunctions of integral operators for
first order Markov process with 2-d states.  A state is an ordered
pair (h,g).  g is the position and h is the derivative.  An allowed
sequential pair of states ((h_0,g_0),(h_1,g_1)) satisfies:

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
    '''Custom version of LinearOperator that implements A*x and x*A for
    the adjacency matrix definied by the parameters:

    d        The scale, that is u/(dy^2)
    h_step   Quantization in h
    g_step   Quantization in g
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
        G = min(self.n_g-1, np.floor(g - self.origin_g)/self.g_step)
        assert G >= 0
        assert G < self.n_g, 'g={0:f} d={1:f} G={2:d} n_g={3:d}'.format(
            g, self.d, G, self.n_g)
        return int(G)
    def H2h(self, H):
        '''Return float position corresponding to integer index H

        '''
        return self.origin_h + H*self.h_step
    def G2g(self, G):
        '''Return float position corresponding to integer index G

        '''
        if G == 0:
            return -self.d
        return self.origin_g + G*self.g_step
    def h_lim(self, g):
        '''Calculates the maximum possible slope, h, at position g.  Boundary
        equation is g(h) = d -h^2/24.

        '''
        d_g = self.d - g
        if d_g < 0:
            assert (-d_g)/self.d < 1e-10
            return 0.0
        return np.sqrt(24*(d_g))
    def ab(self,        # LO_step instance
           h_edge,      # Lower limit of h range
           g_min,       # Lowest g in image
           g_intercept, # g value on slope 1 line where it crosses h=0
           G,           # Index of row
           ):
        '''Return range of allowed state indices for given h interval and G.
        Including any integer H whose image in h overlaps [low,high]
        will include any H whose (H,G) cell intersects the domain.
        '''
        g = max(g_min, self.G2g(G))# Float g of this row
        s_i, s_f = self.G2state[G:G+2] # Range of states in domain for this row
        H_lim = (s_f-s_i)/2        # Number of steps from h=0 to quad boundary
        high = self.h_lim(g)       # Top of h range
        low = max(-high, h_edge, (g-g_intercept)) # Bottom of h range
        
        # Low is integer steps from center (h=0) to beginning of image
        # states.  It would be (-H_lim,H_lim) if low = (high,-high)
        # respectively.
        Low = int(np.floor(low/self.h_step))
        Low += H_lim # Now Low is integer steps from left quadratic boundary
        assert Low >=0
        High = int(np.ceil(high/self.h_step)) + H_lim
        assert High <= s_f - s_i
        return s_i + Low, s_i + High
    
    def boundary_image(
            self,   # LO_step instance     
            h_1,    # z_1=(h_1,g_1) is apex of pie slice for image of
            g_1,    # lower left corner z_0
            g_1_top # Apex for image of upper left z_0
    ):
        '''Calculations for images when the upper left corner is outside of
        the quadratic boundary
        '''
        _in_ = lambda x : x[1]<self.d-x[0]**2/24 # Inside boundary parabola?
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
        h_edge = self.h_lim(-self.d)   # Seed search for left boundary
        g_min = self.d                 # Seed search for bottom
        g_intercept = -self.d - h_edge # Seed search for slope 1 boundary
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
        of them in the arrays self.bounds_a and self.bounds_b.

        Notes:

        * cell[i] is a rectangle with sides h_step and g_step
        * state[i] is the intersection of cell[i] with the domain
        * A((h,g)) = (h-12, g+h-6) is an affine function that maps
            rectangles to parallelograms and maps a point z_0 in the
            domain to z_1 the apex of it's pie slice image.
        * The parabolic boundary is invariant under A
        '''
        
        H,G = self.state_list[i] # Indices of lower left corner of cell[i]
        h_0 = self.H2h(H)        # Float coordinate of left side of cell[i]
        g_0 = self.G2g(G)        # Bottom of cell[i]
        g_0_top = self.G2g(G+1)
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
        '''Calculate the allowed states.  Set the following attributes of
        self: state_list, state_dict, G2state and n_states.  States
        with each value of G are contiguous in state_list and start at
        G2state[G].
        '''
        # The folowing 4 objects are attached to self in this method
        state_list = []      # state_list[i] = (H,G)
        self.state_dict = {} # state_dict[(H,G)] = index of state_list
        self.G2state = np.zeros(self.n_g+1, dtype=np.int32)

        for G in range(self.n_g):
            g = self.G2g(G)
            # H_lim = number of different values of h2H(h): 0 < h < h_lim(g)
            H_lim = int(np.ceil(self.h_lim(g)/self.h_step))
            s_i = len(state_list) #index of first allowed state for this g
            for H_ in range(-H_lim, H_lim):
                H = H_ + self.n_h/2
                self.state_dict[(H,G)] = len(state_list)
                state_list.append((H,G))
            self.G2state[G+1] = len(state_list)
        self.n_states = len(state_list)
        self.state_list = np.array(state_list, dtype=np.int32)
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
                 h_step,            # Size of steps in slope h
                 g_step,            # Size of steps in position g
                 archive_dict=None
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
        # Map from z_0 to z_1 apex of pie slice
        self.affine = lambda h,g: (h-12, h+g-6)

        if archive_dict == None: # Create without reading archive
            self.allowed()
            self.pairs()
            return
        # Read from archive
        for key,value in archive_dict.items():
            setattr(self, key, value)
        self.state_dict = {}
        for i, state in enumerate(self.state_list):
            self.state_dict[tuple(state)] = i
    def conj(self,i):
        ''' Conjugate: Get index for state with -h
        '''
        H, G = self.state_list[i]
        H_c = self.n_h - H -1
        assert (H_c,G) in self.state_dict
        return self.state_dict[(H_c,G)]
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
    def hg(self,      # LO instance
           m_h=None,  # Number of sample points in h
           m_g=None,  # Number of sample points in g
           ):
        '''Return two vectors of coordinates (g and h) that span the range
        of allowed values.  Useful for plotting state vectors.
        '''
        if m_h == None:
            m_h = self.n_h
        if m_g == None:
            m_g = self.n_g
        h_max = self.h_lim(-self.d)
        return (
            np.linspace(-h_max, h_max, m_h, endpoint=False),
            np.linspace(-self.d, self.d, m_g, endpoint=False),
        )
    def vec2z(self,      # LO instance
                v,       # vector with component for each state
                h=None,  # 1-d array of h values at sample points
                g=None,  # 1-d array of g values at sample points
                floor=0
                ):
        ''' Use spline to estimate z(h[i,j],g[i,j]) on the basis of v.
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
                rv[H,G] = max(v[i],floor)
            return rv
        self.spline(v)
        H,G = np.meshgrid(g, h)
        assert G.shape == ( len(h), len(g) )
        rv = self.bs.ev(H.reshape(-1), G.reshape(-1)) # Evaluate bv spline
        rv = np.fmax(rv, floor)
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
        z = np.zeros((self.n_h, self.n_g))
        for G in range(self.n_g):
            for H in range(self.n_h):
                if (H,G) in self.state_dict:
                    i = self.state_dict[(H,G)]
                    z[H,G] = v[i]
        for G in range(self.n_g):
            h = self.h_lim(self.G2g(G))
            H_plus = self.h2H(h)
            if (H_plus,G) not in self.state_dict:
                H_plus -= 1
            H_minus = self.h2H(-h)
            if (H_minus,G) not in self.state_dict:
                H_minus += 1
            if H_plus <= H_minus + 1:
                print('H_plus={0}, H_minus={1}'.format(H_plus, H_minus))
                continue
            # extrapolate on the +h side
            z_last = v[self.state_dict[(H_plus,G)]]
            dz_dh = z_last - v[self.state_dict[(H_plus-1,G)]]
            for H in range(H_plus+1, self.n_h):
                z_last += dz_dh
                z[H,G] = z_last
            # extrapolate on the -h side
            z_last = v[self.state_dict[(H_minus,G)]]
            dz_dh = v[self.state_dict[(H_minus+1,G)]] - z_last
            for H in range(H_minus-1, -1, -1):
                z_last -= dz_dh
                z[H,G] = z_last
        h,g = self.hg() # linspaces for ranges of g and h values
        from scipy.interpolate import RectBivariateSpline
        self.bs = RectBivariateSpline(h, g, z, kx=3, ky=3)
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
        if not 'bs' in vars(self):
            self.spline()      # Build spline of self
        w = self.bs.ev(g,h)    # Evaluate spline at points of v
        w /= LA.norm(w)        # Make w a unit vector
        u = v-w
        d = LA.norm(u)
        if rv: return d, u
        else: return d
        
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

class Archive:
    '''self.op_dict has keys that are tuples (d, h_step, g_step). Values
    are file names.  Those files are pickled dicts with the following
    key/value pairs:

    'self': dict with the following keys (d, h_step, g_step, n_states,
        n_pairs, iterations).  Values are corresponding floats or
        ints.  Note: iterations is the from eigenvector calculation.

    'self_arrays': The name of a file created by numpy.savez that
        contains the arrays G2state, state_list and eigenvector

    'image_dict': A dict with keys that are triples (G,H,iterations)
        and values names of files created by numpy.tofile.

    '''
    
    def __init__(self, LO, dir_name='archive'):
        '''Read relevant files in dir_name and build self.op_dict which has keys
        (d, h_step, g_step).
        '''
        import glob
        import pickle
        import os.path
        self.LO = LO
        self.dir_name = dir_name
        self.op_dict = {}
        for dir_ in glob.glob(dir_name+'/LO*'):
            path = os.path.join(dir_,'dict.pickle')
            s = pickle.load(open(path,'rb'))['self']
            key = tuple([s[x] for x in 'd h_step g_step'.split()])
            if key in self.op_dict:
                print('WARNING {0} and {1} have same key: {2}'.format(
                    self.op_dict[key], name, key))
            self.op_dict[key] = dir_
        return
    
    def update(
            self,       # Archive instance
            key_0,      # (d, d_h, d_g)
            dict_,      # Dictionary for operator key_0
            new_images):
        
        from tempfile import NamedTemporaryFile as NTF
        import pickle
        import os.path

        dir_ = self.op_dict[key_0] # EG archive/LO_100.0_4.00_4.00_bgqmlrrj/
        image_dict = dict_['image_dict']
        for key, value in new_images.items():
            vec_file = NTF(prefix='image_{0}'.format(key),
                           dir=dir_, delete=False)
            value.tofile(vec_file)
            image_dict[key] = os.path.basename(vec_file.name)
        file_ = open(os.path.join(dir_, 'dict.pickle'), 'wb')
        pickle.dump(dict_, file_, 2)
        return
    def write(self, A, image_vector_dict):
        
        from tempfile import NamedTemporaryFile as NTF
        from tempfile import mkdtemp
        import pickle
        import os.path

        prefix = 'LO_{0:.1f}_{1:.2f}_{2:.2f}_'.format(A.d, A.h_step, A.g_step)
        dir_ = mkdtemp(prefix=prefix, dir=self.dir_name)
        np.savez(
            os.path.join(dir_, 'arrays'),
            G2state=A.G2state,
            state_list=A.state_list,
            eigenvector=A.eigenvector,
        )

        image_dict = {}
        for key, value in image_vector_dict.items():
            vec_file = NTF(prefix='image_{0}'.format(key),
                           dir=dir_, delete=False)
            value.tofile(vec_file)
            image_dict[key] = os.path.basename(vec_file.name)
        dict_ = {'image_dict':image_dict}
        
        # Use s for attributes of LO that are not np.arrays so that d,
        # h_step, g_step can be read quickly for dictionary key
        s = {}
        for key in '''d h_step g_step
        n_states n_pairs shape n_g n_h origin_h origin_g'''.split():
            s[key] = getattr(A,key)
        dict_['self'] = s
        file_ = open(os.path.join(dir_,'dict.pickle'),'wb')
        pickle.dump(dict_, file_, 2)
        return
    
    def get(
        self,             # Archive instance
        d, h_step, g_step,
        sources=[],       # List of triples (frac_h,frac_g,iterations)
        ):
        '''Read or calculate.  Note that bounds_a and bounds_b are not stored
        or read because they get too big to store on disk.

        Return LO, image_vector_dict

        '''
        import os.path
        def read_operator(args, dir_name):
            import pickle
            dict_ = pickle.load(open(os.path.join(dir_name,'dict.pickle'),'rb'))
            arg_dict = dict_['self'].copy() # For call to LO
            for key,value in np.load(os.path.join(dir_name,'arrays.npz')).items():
                arg_dict[key] = value
            return self.LO(*args, archive_dict=arg_dict), dict_
        def make_images(sources, A):
            rv = {}
            for key in sources:
                H,G,iterations = key
                point_index = A.state_dict[(H,G)]
                v = np.zeros(A.n_states)
                v[point_index] = 1.0
                for i in range(iterations):
                    v = A.matvec(v)
                    v /= v.max()
                rv[key] = v
            return rv

        # To start, read or create linear operator A and eigenvector
        key = (d, h_step, g_step)
        if key in self.op_dict:
            A, dict_ = read_operator(key, self.op_dict[key])
        else:
            A = self.LO(*key)
            A.power()
        int_sources = fractions2indices(sources,A)

        if key in self.op_dict: # Read images
            dir_ = self.op_dict[key]
            image_vector_dict = {}
            for image_key,value in dict_['image_dict'].items():
                image_vector_dict[image_key] = np.fromfile(
                    os.path.join(dir_,value))
            # Calculate set of additional images required by int_sources
            need = set(int_sources) - set(image_vector_dict.keys())
            if need: # Calculate missing images
                A.pairs()
                new_images =  make_images(need, A)
                self.update(key, dict_, new_images)
                image_vector_dict.update(new_images)
        # Calculate whole archive file because it doesn't exist
        else:
            image_vector_dict = make_images(int_sources, A)
            self.write(A, image_vector_dict)
        return A, image_vector_dict
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
