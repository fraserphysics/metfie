import numpy as np
cimport numpy as np
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t
DTYPE = np.int64
ctypedef np.int64_t DTYPE_t

class staggered:
    def __init__(self, A):
        cdef int i, n = len(A)
        cdef DTYPE_t [:] pointers = np.empty(n, dtype=np.int64)
        cdef ITYPE_t [:] B = np.empty(n, dtype=np.int32)
        cdef ITYPE_t [:] A_i

        self.B = B
        for i in range(n):
            A_i = A[i]
            pointers[i] = <DTYPE_t>(&(A_i[0]))
            self.B[i] = len(A[i])
        self.pointers = pointers
        return
    def Print(self):
        cdef DTYPE_t [:] pointers = self.pointers
        cdef ITYPE_t [:] B = self.B
        cdef ITYPE_t *A
        cdef int i, j, n = len(self.B)
        # Except for "print" the loop below can be parallel prange()
        for i in range(n):
            A = <ITYPE_t *> pointers[i]
            for j in range(B[i]):
                v = A[j]
                print('A[%d,%d]=%d'%(i,j,v))
    
