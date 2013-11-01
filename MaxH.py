""" MaxH.py:

Author:  Andy Fraser

Purpose: I wrote this code to do entropy calcuations

Use/imported by calc.py

"""
import numpy as np

def H_P_calc(mu,P):
    """Calculate probability based entropy rate for transition matrix
    P and stationary probability mu.
    """
    N = len(mu)
    H = 0.0
    for i in range(N):
        for j in range(N):
            if P[i,j] > 0.0:
                H -= mu[i]*P[i,j]*np.log(P[i,j])
    return H
def H_top_calc(A,tol=1e-14,N_init=5):
    """Calculate topological entropy for adjacency matrix A.  A must
    be square, ie, NxN.
    """
    N = A.shape[0]
    n = np.matrix(np.ones(N).reshape(1,N))/N
    for i in range(N_init): # Avoid termination due to initial transients
        n *= A
        n /= n.sum()
    growth = 0
    while True:
        new = n*A
        sum_new = new.sum()
        if abs(sum_new-growth)/sum_new < tol:
            return np.log(growth)
        growth = sum_new
        n = new/sum_new
def P_calc(A, AT,      # LinearOperator and its transpose
           tol=1e-6,
           maxiter=10000):
    """Calculate the probabilities that maximize the entropy rate for
    adjacency matrix A.
    """
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import eigs as sparse_eigs
    w,b = sparse_eigs(A,  k=1, tol=tol, maxiter=maxiter)
    print('eigenvalue=%f'%(w[0].real,))
    w,e = sparse_eigs(AT, k=1, tol=tol, maxiter=maxiter)
    e = e[:,-1].real # Left eigenvector
    b = b[:,-1].real # Right eigenvector
    norm = np.dot(e, A*b)
    Ab = A*b # A vector?
    mu = e*Ab/norm # Also a vector ?
    L = e/mu
    R = b/norm
    P = LinearOperator(A.shape,
                       lambda v: L*(A*(R*v)),
                       rmatvec = lambda v: (A.rmatvec(v*L))*R
                       )
    return mu,P
def Test(M):
    """Calculate and the transition probability matrix that achieves
    the maximum entropy rate for the adjacency matrix M.
    """
    import scipy.sparse as ssm
    A = ssm.csr_matrix(M, dtype=np.float)
    HT = H_top_calc(A)
    mu,P = P_calc(A)
    HP = H_P_calc(mu,P)
    print("""
H_top  = %f
H_prob = %f
    mu = %s
     T =
%s"""%(HT,HP,mu,P.todense()))
if __name__ == "__main__":
    A = np.array([
        [1,1,0,1],
        [0,1,1,1],
        [1,0,1,1],
        [0,1,1,0]])
    B = np.array([
        [1,1],
        [1,0]])
    Test(A)
    Test(B)

#---------------
# Local Variables:
# eval: (python-mode)
# End:
