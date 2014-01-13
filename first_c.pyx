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
ITYPE = np.int32
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t
from first import LO_step
from cython.parallel import prange


class LO(LO_step):
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
        cdef ITYPE_t [:,:] a_bounds = self.a_bounds
        cdef ITYPE_t [:,:] b_bounds = self.b_bounds
        cdef DTYPE_t [:] v_ = v
        cdef DTYPE_t [:] rv_ = rv
        cdef int i, j, k, a, b, n
        cdef int n_states = self.n_states
        cdef float dgdh = self.g_step*self.h_step
        cdef float t

        for i in range(n_states):
            t = v_[i] * dgdh
            n = n_bounds[i]
            for k in range(n):
                a = a_bounds[i,k]
                b = b_bounds[i,k]
                for j in range(a,b):
                    rv_[j] += t
        return rv
    @cython.boundscheck(False)
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
        cdef ITYPE_t [:,:] a_bounds = self.a_bounds
        cdef ITYPE_t [:,:] b_bounds = self.b_bounds
        cdef DTYPE_t [:] v__ = v
        cdef DTYPE_t * v_
        cdef DTYPE_t [:] rv_ = rv
        cdef DTYPE_t temp
        cdef int i, j, k, a, b, n, t_n, n_threads=8
        cdef int n_states = self.n_states
        cdef float dgdh = self.g_step*self.h_step
        cdef float t

        '''See http://docs.cython.org/src/userguide/parallelism.html.  Work on matvec()'''
        v_ = &(v__[0])
        for i in prange(n_states, nogil=True):
            temp = 0.0
            n = n_bounds[i]
            for k in range(n):
                a = a_bounds[i,k]
                b = b_bounds[i,k]
                for j in range(a,b):
                    temp += v_[j] * dgdh
            rv_[i] +=  temp
        return rv
    def rebound(self):
        '''Make numpy arrays from bounds for speed in matvec
        '''
        cdef int n_states = self.n_states
        cdef int i, j, n
        bounds = self.bounds
        max_len = max([len(bounds[i]) for i in range(n_states)])
        self.n_bounds = np.zeros(n_states, dtype=np.int32)
        self.a_bounds = np.zeros((n_states, max_len), dtype=np.int32)
        self.b_bounds = np.zeros((n_states, max_len), dtype=np.int32)
        # Make views of numpy arrays
        cdef ITYPE_t [:] n_bounds = self.n_bounds
        cdef ITYPE_t [:,:] a_bounds = self.a_bounds
        cdef ITYPE_t [:,:] b_bounds = self.b_bounds
        for i in range(n_states):
            n = len(bounds[i])
            n_bounds[i] = n
            for j in range(n):
                a_bounds[i,j],b_bounds[i,j] = bounds[i][j]
        self.bounds = None # Memory management
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
    
    def ab(self,  # LO instance
           low_,  # Lower limit of h range
           high_, # Upper limit of h range
           G_     # Combining G with each H in retuned range must be allowed
           ):
        ''' Return range of allowed state indices
        '''

        cdef float f, g, h, h_
        cdef float h_step = self.h_step
        cdef float h_min = self.h_min
        cdef float low = low_
        cdef float high = high_
        
        cdef int H
        cdef int G = G_
        cdef int a = -1
        cdef int b = -2

        h = low - h_step
        while h < high + h_step:
            H = int(round((h-h_min)/h_step))
            f = h_min + H * h_step
            if (G,H) in self.state_dict and f >= low:
                g,h_,a = self.state_dict[(G,H)]
                break
            h += h_step*.4 # FixMe: why not .9 ?
        if h > high:
            return -1, -2 # No h values allowed
        h = high + h_step
        while h >  low - h_step:
            H = int(round((h-h_min)/h_step))
            f = h_min + H * h_step
            if (G,H) in self.state_dict and f <= high:
                g,h_,b = self.state_dict[(G,H)]
                break
            h -= h_step*.4
        return a,b
    def s_bounds(self, L_g_, U_g_, g_0_, bounds, backward=False):
        '''Given g_0 and (L_g, U_g), limits on g_1 derived from g_0 and h_0,
        find sequences of state indices for allowed successors and append
        them to bounds.
        '''
        cdef int i_min, i_max, G_1, n_pairs

        cdef float g_1
        cdef float g_step = self.g_step
        cdef float g_min = self.g_min
        cdef float g_max = self.g_max
        cdef float L_g = L_g_
        cdef float U_g = U_g_
        cdef float g_0 = g_0_
        cdef float dy = self.dy
        
        if L_g > U_g:
            return 0
        
        if backward:
            ab = lambda L, U, G: self.ab(-U, -L, G)
        else:
            ab = lambda L, U, G: self.ab(L, U, G)
        n_pairs = 0

        i_min = int(round((L_g-g_min)/g_step)) - 1
        i_max = int(round((U_g-g_min)/g_step)) + 1
        
        for G_1 in range(i_min,i_max):
            g_1 = g_min + G_1 * g_step
            if g_1 < L_g or g_1 > U_g:
                continue
            L_h = (g_1 - g_0)/dy - 6 * dy
            U_h = np.sqrt(24*(g_max - g_1)) # self.h_lim(g_1) Slow
            if g_0 > g_max - 6*dy:
                if L_h < -U_h:
                    L_h = -U_h
            a,b = ab(L_h, U_h, G_1)         # Slow
            if b >= a:
                bounds.append( (a,b+1) )    # Slow
                n_pairs += b + 1 - a
        return n_pairs
    
#---------------
# Local Variables:
# mode: python
# End:
