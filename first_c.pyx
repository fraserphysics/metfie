"""first_c.pyx: Cython version of LO from first.py.  This code is faster
than first.py, but it does not introduce any new methods.

Requires the following from pure python: self.n_states, self.n_pairs
    self.G2state, self.d, self.state_list, self.h_lim() self.g2G(),
    self.n_g, self.origin_g, self.g_step, self.h_step, self.bounds_a,
    self.bounds_b

"""
# See http://docs.cython.org/src/userguide/memoryviews.html
#To build do the following sequence:
# python3 setup.py build_ext --inplace
# python setup.py build_ext --inplace
import numpy as np
cimport cython, numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

PTYPE = np.int64               # For pointers
ctypedef np.int64_t PTYPE_t

import first
from cython.parallel import prange

cdef extern from "math.h":
    double floor(double x)
    double ceil(double x)
    double sqrt(double x)

class LO_step(first.LO_step):
    ''' 
    '''
    @cython.boundscheck(False)
    def matvec(self, v, rv=None):
        '''Implements matrix vector multiply.  Allocates new vector and returns
        self * v
        '''
        if not 'n_bounds' in self.__dict__: # See if rebound() already done
            self.rebound()
        if rv == None:
            rv = np.zeros(self.n_states)
        else:
            rv[:] = 0.0
        # Make views of numpy arrays
        cdef ITYPE_t [:] n_bounds = self.n_bounds
        cdef PTYPE_t [:] a_pointers = self.a_pointers
        cdef PTYPE_t [:] b_pointers = self.b_pointers
        cdef ITYPE_t *bounds_a_i, *bounds_b_i
        cdef DTYPE_t [:] v_ = v
        cdef DTYPE_t [:] rv_ = rv
        cdef int i, j, k, a, b, n
        cdef int n_states = self.n_states
        cdef float dgdh = self.g_step*self.h_step
        cdef float t

        for i in range(n_states): # prange is slower
            t = v_[i] * dgdh
            if t == 0.0:
                continue
            n = n_bounds[i]
            bounds_a_i = <ITYPE_t *>(a_pointers[i])
            bounds_b_i = <ITYPE_t *>(b_pointers[i])
            for k in range(n):
                a = bounds_a_i[k]
                b = bounds_b_i[k]
                for j in range(a,b):
                    rv_[j] += t
        return rv
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def rmatvec(self, v, rv=None):
        '''Implements transpose matrix vector multiply.  Allocates new vector
        and returns v * self
        '''
        if not 'n_bounds' in self.__dict__: # See if rebound() already done
            self.rebound()
        if rv == None:
            rv = np.zeros(self.n_states)
        else:
            rv[:] = 0.0
        # Make views of numpy arrays
        cdef ITYPE_t [:] n_bounds = self.n_bounds
        cdef PTYPE_t [:] a_pointers = self.a_pointers
        cdef PTYPE_t [:] b_pointers = self.b_pointers
        cdef ITYPE_t *bounds_a_i, *bounds_b_i
        cdef DTYPE_t [:] v__ = v
        cdef DTYPE_t * v_
        cdef DTYPE_t [:] rv_ = rv
        cdef DTYPE_t temp
        cdef int i, j, k, a, b, n, t_n, n_threads=8
        cdef int n_states = self.n_states
        cdef float dgdh = self.g_step*self.h_step
        cdef float t

        '''See http://docs.cython.org/src/userguide/parallelism.html.  Here
        prange() over i is fast because rv_[i] gets written.  That is,
        the writes to rv_[i] are local and the reads from v_[j]
        aren't.  Prange() is bad for matvec because the reads are
        local and the writes aren't.
        '''
        v_ = &(v__[0])
        for i in prange(n_states, nogil=True):
            temp = 0.0
            n = n_bounds[i]
            bounds_a_i = <ITYPE_t *>(a_pointers[i])
            bounds_b_i = <ITYPE_t *>(b_pointers[i])
            for k in range(n):
                a = bounds_a_i[k]
                b = bounds_b_i[k]
                for j in range(a,b):
                    temp += v_[j]
            rv_[i] +=  temp * dgdh
        return rv
    def rebound(self):
        '''Make numpy arrays for speed in rmatvec
        '''
        cdef int i, m, n = self.n_states
        cdef PTYPE_t [:] a_pointers = np.empty(n, dtype=np.int64)
        cdef PTYPE_t [:] b_pointers = np.empty(n, dtype=np.int64)
        cdef ITYPE_t [:] n_bounds = np.empty(n, dtype=np.int32)
        cdef ITYPE_t [:] i_view

        for i in range(n):
            m = len(self.bounds_a[i])
            n_bounds[i] = m
            if m == 0:
                a_pointers[i] = <PTYPE_t>0
                b_pointers[i] = <PTYPE_t>0
            else:
                i_view = self.bounds_a[i]
                a_pointers[i] = <PTYPE_t>(&(i_view[0]))
                i_view = self.bounds_b[i]
                b_pointers[i] = <PTYPE_t>(&(i_view[0]))
            
        self.n_bounds = n_bounds
        self.a_pointers = a_pointers
        self.b_pointers = b_pointers
        return

    def power(self, n_iter=1000, small=1.0e-5, v=None, op=None, verbose=False):
        '''Calculate largest eigevalue and corresponding eigenvector of op.
        '''
        import numpy.linalg as LA
        assert op == None
        mem = np.ones((2,self.n_states)) # Double buffer, no allocate in loop
        if v != None:
            mem[1,:] = v
        def step(s_old, # Last estimate of eigenvalue
                 v_old, # Last estimate of eigenvector.  Assume norm(v) = 1
                 v_new  # Storage for new eigenvector estimate
                 ):
            self.rmatvec(v_old, v_new)
            s = LA.norm(v_new)
            assert s > 0.0
            v_new /= s
            v_old -= v_new
            dv = LA.norm(v_old)
            ds = abs((s-s_old)/s)
            return s, ds, dv
        s, ds, dv = step(0, mem[1], mem[0])
        for i in range(n_iter):
            v_old = mem[i%2]
            v_new = mem[(i+1)%2]
            s, ds, dv = step(s, v_old, v_new)
            if ds < small and dv < small:
                break
        if verbose: print(
'''With n_states=%d and n_pairs=%d, finished power() at iteration %d
    ds=%g, dv=%g'''%(self.n_states, self.n_pairs, i, ds, dv))
        self.iterations = i
        self.eigenvalue = s
        self.eigenvector = v_new
        return s,v_new
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def s_bounds(
            self,      # LO instance
            int i_,    # index of state in self.state_list
            ):
        '''Given g_0 and (L_g, U_g), limits on g_1 derived from g_0 and h_0,
        find sequences of state indices for allowed successors and append
        them to bounds.  This is where LO_step build spends most time.
        '''
        cdef ITYPE_t [:] G2state = self.G2state
        cdef double d = self.d, h_step =  self.h_step, g_step = self.g_step
        cdef double origin_g = self.origin_g
        cdef int G_, n_g = self.n_g

        H,G = self.state_list[i_] # Indices of lower left corner of cell[i]
        h_0 = self.H2h(H)        # Float coordinate of left side of cell[i]
        g_0 = self.G2g(G)        # Bottom of cell[i]
        g_0_top = self.G2g(G+1)
        h_1 = h_0 - 12
        g_1, g_1_top = (g+h_0-6 for g in (g_0, g_0_top))
        # (h_1,g_1) and (h_1,g_1_top) are the images under A of the
        # left side of cell[i] (perhaps with g_0 truncated to -d)

        if g_1_top < d - h_1*h_1/24: # Inside quadratic boundary
            h_edge = h_1
            g_min = g_1
            g_intercept = g_1_top - h_1
        else:
            h_edge, g_min, g_intercept = self.boundary_image(h_1, g_1, g_1_top)

        # Prepare for loop over g values in image.  Have no python
        # objects in the loop.
        cdef int n_pairs = 0
        cdef ITYPE_t [:] bounds_a = np.zeros(self.n_g, dtype=np.int32)
        cdef ITYPE_t [:] bounds_b = np.zeros(self.n_g, dtype=np.int32)
        cdef int i = i_, j, s_i, s_f, Ds, a, b, len_bounds = 0
        cdef double h_i, h_f, Dh
        cdef ITYPE_t *I_pointer
        cdef DTYPE_t *D_pointer
        cdef int G_i = self.g2G(max(-d,g_min))#self.g2G(max(-self.d,g_min))
        # Want this loop to compile as pure c
        for G_ in range(G_i, n_g):
            # Begin code segment that is self.ab() in first.py
            if G_ == 0:
                g_ = -d
            else:
                g_ = origin_g + G_*g_step
            if g_min < g_: #g = max(g_,g)
                g = g_
            else:
                g = g_min
            high = sqrt((24*(d - g))) #self.h_lim(g)
            low = g - g_intercept
            if low < h_edge:
                low = h_edge
            if low < -high:
                low = -high # max(-high, max(h_edge, (g-g_intercept)))
            s_i = G2state[G_]
            s_f = G2state[G_+1]
            H_lim = (s_f-s_i)/2
            a = s_i + H_lim + <ITYPE_t>(floor(low/h_step))
            b = s_i + H_lim + <ITYPE_t>(ceil(high/h_step))
            if s_i > a:
                a = s_i
            if s_f < b:
                b = s_f
            # End code segment that is self.ab() in first.py
            # Uncomment to check
            #assert (a,b) == self.ab(h_edge, g_min, g_intercept, G_)

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
    
#---------------
# Local Variables:
# mode: python
# End:
