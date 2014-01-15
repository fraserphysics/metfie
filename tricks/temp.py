from temp_c import staggered
import numpy as np

A = np.empty(3, np.object)
B = np.zeros(3, np.int32)

A[:] = [np.array([-1,5,9,23], dtype=np.int32), np.arange(5, dtype=np.int32),
        np.arange(2, dtype=np.int32)]
B[:] = [len(A[i]) for i in range(3)]
print('A=%s\nB=%s'%(A,B))
S = staggered(A)
S.Print()
