'''C.pyx: cython code for speed up.  Derived from C.pyx in hmmds3.

'''
#To build do the following sequence:
# python3 setup.py build_ext --inplace
# python setup.py build_ext --inplace
import numpy as np
#import warnings
#warnings.simplefilter('ignore', SS.SparseEfficiencyWarning)
#warnings.simplefilter('ignore',SS.SparseEfficiencyWarning)
# Imitate http://docs.cython.org/src/tutorial/numpy.html
# http://docs.cython.org/src/userguide/memoryviews.html
cimport cython, numpy as np
DTYPE = np.float64
ITYPE = np.int32
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

class LO(object):
    ''' Custom version of LinearOperator.  All of calc runs 10.5 times
    faster with this in Cython.
    '''
    def __init__(self, i_list, N_states, Nf):
        self.i_list = i_list
        self.dtype = np.dtype(np.float64)
        self.Nf = Nf
        self.N_states = N_states
        self.shape = (N_states, N_states)
    #@cython.boundscheck(False)
    def matvec(self, v):
        rv = np.zeros(self.N_states)
        # Make views of numpy arrays
        cdef DTYPE_t [:] v_ = v
        cdef DTYPE_t [:] rv_ = rv
        cdef ITYPE_t [:,:] kk
        cdef int i, Nf, j_bottom, j_top, i_n, jk_0, j, ij, k_bottom, k_top, k
        Nf = self.Nf

        for i in range(Nf):
            i_data = self.i_list[i]
            j_bottom = i_data.j_bottom
            j_top = i_data.j_top
            i_n = i_data.n
            kk = i_data.kj
            for j in range(0, j_top - j_bottom):
                ij = i_n + j
                k_bottom, k_top = kk[j]
                j_data = self.i_list[j+j_bottom]
                jk_0 = j_data.n - j_data.j_bottom + k_bottom
                for k in range(0, k_top - k_bottom):
                    rv_[ij] += v_[jk_0+k]
        return rv
    def rmatvec(self, v):
        rv = np.zeros(self.N_states)
        # Make views of numpy arrays
        cdef DTYPE_t [:] v_ = v
        cdef DTYPE_t [:] rv_ = rv
        cdef ITYPE_t [:,:] kk
        cdef int i, Nf, j_bottom, j_top, i_n, jk_0, j, ij, k_bottom, k_top, k
        Nf = self.Nf

        for i in range(Nf):
            i_data = self.i_list[i]
            j_bottom = i_data.j_bottom
            j_top = i_data.j_top
            i_n = i_data.n
            kk = i_data.kj
            for j in range(0, j_top - j_bottom):
                ij = i_n + j
                k_bottom, k_top = kk[j]
                j_data = self.i_list[j+j_bottom]
                jk_0 = j_data.n - j_data.j_bottom + k_bottom
                for k in range(0, k_top - k_bottom):
                    rv_[jk_0+k] += v_[ij]
        return rv
    def __mul__(self,x):
        x = np.asarray(x)

        if x.ndim == 1:
            return self.matvec(x)
        elif x.ndim == 2 and x.shape[1] == 1:
            return self.matvec(self, x[0])
        else:
            raise ValueError('expected x.shape = (n,) or (n,1)')

class LOT(LO):
    ''' LO transpose
    '''
    def __init__(self, ij, N_states, Nf):
        LO.__init__(self, ij, N_states, Nf)
        t = self.matvec
        self.matvec = self.rmatvec
        self.rmatvec = t

class QLO(LO):
    ''' Custom version of LinearOperator.  FixMe: This implementation
    out of date.  Change to imitate LO.
    '''
    def __init__(self, ij, N_states, Nf):
        self.ij = ij
        self.dtype = np.dtype(np.float64)
        self.Nf = Nf
        self.N_states = N_states
        self.shape = (Nf, N_states)
    def matvec(self, v):
        assert v.shape == (self.N_states,)
        rv = np.zeros(self.Nf)
        
        # Make views of numpy arrays
        cdef DTYPE_t [:] v_ = v
        cdef DTYPE_t [:] rv_ = rv
        cdef int i, j, n, dum_0, dum_1
        
        for (i,j), (dum_0, dum_1, n) in self.ij.items():
            rv_[j] += v_[n]
        return rv
    def rmatvec(self, v):
        assert v.shape == (self.Nf,)
        rv = np.zeros(self.N_states)
        
        # Make views of numpy arrays
        cdef DTYPE_t [:] v_ = v
        cdef DTYPE_t [:] rv_ = rv
        cdef int i, j, n, dum_0, dum_1
        
        for (i,j), (dum_0, dum_1, n) in self.ij.items():
            rv_[n] += v_[j]
        return rv
        
#--------------------------------
# Local Variables:
# mode: python
# End:
