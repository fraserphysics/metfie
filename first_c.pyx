"""first_c.pyx: Cython version of LO from first.py.  This code is faster
than first.py, but it does not introduce any new methods.
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

from first import LO_step
from cython.parallel import prange


class LO_step(LO_step):
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
                    temp += v_[j] * dgdh
            rv_[i] +=  temp
        return rv
    def rebound(self):
        '''Make numpy arrays for speed in matvec
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
        mem = np.empty((2,self.n_states)) # Double buffer, no allocate in loop
        if v != None:
            mem[1,:] = v
        else:
            mem[1,:] = 1.0
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
        self.eigenvalue = s
        self.eigenvector = v_new
        return s,v_new
    
    def pointerize_G2(self):
        '''Get arrays of pointers to np.arrays in self.G2state and
        self.G2h_list.  In the main loop in s_bounds, those pointers are
        accessed as c objects for speed.
        '''
        cdef int i, n = len(self.G2state)
        cdef PTYPE_t [:] G2state = np.empty(n, dtype=np.int64)
        cdef PTYPE_t [:] G2h_list = np.empty(n, dtype=np.int64)
        cdef ITYPE_t [:] i_view
        cdef DTYPE_t [:] d_view
        for i in range(n):
            i_view = self.G2state[i]
            G2state[i] = <PTYPE_t>(&(i_view[0]))
            d_view = self.G2h_list[i]
            G2h_list[i] = <PTYPE_t>(&(d_view[0]))
        self.G2state_ = G2state
        self.G2h_list_ = G2h_list

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def s_bounds(
            self,         # LO instance
            int state_n,  # index of state in self.state_list
            ):
        '''Given g_0 and (L_g, U_g), limits on g_1 derived from g_0 and h_0,
        find sequences of state indices for allowed successors and append
        them to bounds.
        '''
        # "return 1" here reduces LO build time (n_g=500, n_h=500)
        # from 9.8 seconds to 0.3 seconds.
        if not 'G2state_' in self.__dict__:
            self.pointerize_G2()
        cdef PTYPE_t [:] G2state = self.G2state_
        cdef PTYPE_t [:] G2h_list = self.G2h_list_
        cdef double g_max = self.g_max
        cdef double dy = self.dy
        cdef double g_0, h_0, g_1, L_h, U_h, U_g, L_g
        cdef int G_0, G_1

        # "return 1" here reduces LO build time (n_g=500, n_h=500)
        # from 9.8 seconds to 0.4 seconds.
        
        g_0, h_0, G_0, H_0 = self.state_list[state_n]
        # Calculate float range of allowed g values
        if g_0 > g_max - 6*dy**2:
            U_g = g_max
        else:
            U_g = min(g_max,
                g_0 + dy*self.h_lim(g_0) - 6*dy**2)
        L_g = max(self.g_min,g_0 + h_0*dy - 6*dy**2)
        if L_g > U_g:
            self.bounds_a[state_n] = np.zeros(0, dtype=np.int32)
            self.bounds_b[state_n] = self.bounds_a[state_n]
            return 0
        
        # Prepare for loop over g values in image.  Have no python
        # objects in the loop.
        G_i_, G_f_ = self.fi_range(L_g, U_g, self.g_step, self.g_min)
        cdef int n_pairs = 0
        cdef ITYPE_t [:] bounds_a = np.zeros(self.n_g, dtype=np.int32)
        cdef ITYPE_t [:] bounds_b = np.zeros(self.n_g, dtype=np.int32)
        cdef int i, j, s_i, s_f, Ds, a, b, len_bounds = 0
        cdef double h_i, h_f, Dh
        cdef int G_i = G_i_
        cdef int G_f = min(G_f_+1,self.n_g) # +1 cause f(G_0) <= g_0 < f(G_0+1)
        cdef double g_step = self.g_step, g_min = self.g_min
        cdef ITYPE_t *I_pointer
        cdef DTYPE_t *D_pointer
        # This loop compiles as pure c
        for G_1 in range(G_i, G_f):
            g_1 = g_min + G_1 * g_step
            U_h = (24*(g_max - g_1))**.5
            L_h = max( (g_1 - g_0)/dy - 6 * dy, -U_h)
            # Begin code segment that is self.ab() in first.py
            I_pointer = <ITYPE_t *>(G2state[G_1])
            D_pointer = <DTYPE_t *>(G2h_list[G_1])
            s_i = I_pointer[0]
            s_f = I_pointer[1]
            h_i = D_pointer[0]
            h_f = D_pointer[1]
            if s_i == s_f and not (L_h <= h_i and h_f <= U_h):
                continue               # No successors for G_1
            Ds = s_f-s_i
            Dh = h_f-h_i
            if L_h <= h_i:
                a = s_i
            else:
                i = int((L_h-h_i)/(Dh/Ds))
                a = s_i + i
            if U_h >= h_f:
                b = s_f + 1
            else:
                i = int((h_f - U_h)/(Dh/Ds))
                b = s_i + i + 1
            # End code segment that is self.ab() in first.py
            if b <= a:
                continue              # No successors for G_1
            bounds_a[len_bounds] = a
            bounds_b[len_bounds] = b
            len_bounds += 1
            n_pairs += b - a
        self.bounds_a[state_n] = np.array(bounds_a[:len_bounds])
        self.bounds_b[state_n] = np.array(bounds_b[:len_bounds])
        return n_pairs
    
#---------------
# Local Variables:
# mode: python
# End:
