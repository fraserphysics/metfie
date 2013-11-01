"""pq_calc.py: Variant of calc.py that uses 2-d state consisting of
position and derivative and first order Markov model rather than
second order model with 1-d state.

To do:

1. Translate to pq

>>> Nx = 10     # len(self.x)
>>> C = 2.56e10 # EOS parameter
>>> xi = .4     # Initial x
>>> xf = 4.0    # Final x
>>> mass = 100.0
>>> Nx = 200
>>> dev = 0.01
>>> Nf = 50      # Number of quantization levels of f at each x
>>> gun = GUN(C,xi,xf,mass,Nx,dev)
>>> gun.stationary(Nf)
eigenvalue=18.546752


Note: I use "+step/3" in calls to np.arrange to get N+1 items.
"""
import numpy as np
import numpy.linalg.linalg as LA, scipy.interpolate
class GO(object):
    """Generic object to hold variables.  magic.C is prettier than
    magic['C'].  Test suggests assignment to a new attibute of an
    instance is OK.
    """
    def __init__(self,d):
        for key,value in list(d.items()):
            self.__setattr__(key,value)

class LO(object):
    ''' Custom version of LinearOperator
    '''
    def __init__(self, i_list, N_states, Nf):
        import scipy.sparse as SS
        self.dtype = np.dtype(np.float64)
        self.shape = (N_states, N_states)
        A = SS.lil_matrix((N_states, N_states))
        for i in range(Nf):
            for j in range(*i_list[i].j_range()):
                ij = i_list[i].n_j(j)
                for k in range(*i_list[i].k_range(j)):
                    jk = i_list[j].n_j(k)
                    A[ij,jk] = 1.0
        self.A = A.tocsc()
    def matvec(self, v):
        return self.A*v
    def rmatvec(self, v):
        return v*self.A
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
    def __init__(self, i_list, N_states, Nf):
        LO.__init__(self, i_list, N_states, Nf)
        t = self.matvec
        self.matvec = self.rmatvec
        self.rmatvec = t
          
class QLO(LO):
    ''' Custom version of LinearOperator.  ToDo: Implement as sparse
    csc matrix.
    
    '''
    def __init__(self, i_list, N_states, Nf):
        import scipy.sparse as SS
        self.dtype = np.dtype(np.float64)
        self.shape = (Nf, N_states)
        A = SS.lil_matrix((Nf, N_states))
        for i in range(Nf):
            for j in range(*i_list[i].j_range()):
                ij = i_list[i].n_j(j)
                A[j,ij] = 1.0
        self.A = A.tocsc()
class GUN(object):
    def __init__(self,  # GUN instance
                 C,     # Constant in nominal equation of state
                 xi,    # Initial position of projectile / cm
                 xf,    # Final/muzzle position of projectile /cm
                 m,     # Mass of projectile / g
                 N,     # Number of intervals between xi and xf
                 dev,   # Constraint on EOS variation: |f/\tilde f -1| < dev/2
                 ):
        self.dev = dev
        self.C = C
        self.xi = xi
        self.xf = xf
        self.m = m
        self._set_N(N)
        self.LU = lambda a: ( # General upper/lower bounds
            a*(1-self.dev/2),
            a*(1+self.dev/2))
        self.LUi = lambda sf,i: self.LU(sf[i]) # Bounds at i
        return
    def _set_N(self,  # GUN instance
               N,     # Number of intervals between xi and xf
               ):
        '''Interval lengths are uniform on a log scale ie constant ratio.  x_c
        are points in the centers of the intervals and x are the end
        points.

        '''
        step = np.log(self.xf/self.xi)/N
        log_x = np.arange(np.log(self.xi),np.log(self.xf)+step/3,step)
        assert len(log_x) == N+1
        # N+1 points and N intervals equal spacing on log scale
        log_c = log_x[:-1] + step/2
        self.x = np.exp(log_x)
        self.x_c = np.exp(log_c) # Centers of intervals on log scale
        self.dx = self.x[1:] - self.x[:-1]
        self.s = self.C/self.x_c**3  # Nominal eos at sample points
        return
    def recursive(self, # GUN instance
                  i,    # Index of position x
                  f     # Value of EOS
                  ):
        '''Calculate allowed envelope of slopes and one step
        successors implied by requirement that function can continue
        indefinitely forward and backward without violating the
        convexity constraint.  Named "recursive" because one can
        recursively define these bounds by requiring that legal
        predecessors and successors must have legal predecessors and
        succesors.

        Return GO with attributes:
        x_a          Where line through (x_c[i],f) kisses U x_a <= x
        x_b          Where line through (x_c[i],f) kisses U x_b >= x
        df_a         Slope of tangent at x_a
        df_b         Slope of tangent at x_b
        f_next_max   Maximum of allowed values at x_c[i+1]
        f_next_min   Minimum of allowed values at x_c[i+1]
        '''
        x_i = self.x_c[i]
        f_1 = f*x_i**3    # Shift to x=1
        U = (1+self.dev/2)*self.C
        dU = lambda x: -3*U/x**4
        F = lambda x: f_1*x**4 - 4*U*x + 3*U # Zero at tangent points
        # Calculate tangent points
        if f_1 < U:
            from scipy.optimize import brentq
            x = 1.0
            x_ = x
            while F(x)*F(x_) >= 0:
                x_ /= 2
                if x_ < 1e-4:
                    raise RuntimeError
            x_a = brentq(F, x, x_)*x_i
            x_ = x
            while F(x)*F(x_) >= 0:
                x_ *=2
                if x_ > 100:
                    raise RuntimeError
            x_b = brentq(F, x, x_)*x_i
        else:
            x_a = x_b = x_i
        # Calculate tangent slopes
        df_a = dU(x_a)
        df_b = dU(x_b)
        # Calculate bottom and top for f(x_c[i+1]) from the two slopes
        x_next = self.x_c[i+1]
        x_next2 = self.x_c[i+2]
        b,t = (f + (x_next - x_i)*d for d in (df_a, df_b))
        b2,t2 = (f + (x_next2 - x_i)*d for d in (df_a, df_b))
        b = max(b, (1-self.dev/2)*self.C/x_next**3) # Apply lower dev bound
        T = U/x_next**3  # Upper dev bound on f(x_next)
        if x_b < x_next: # OK to follow df_b line to U then follow U to x_next
            t = T
        assert t <= T
        T2 = U/x_next2**3  # Upper dev bound on f(x_next)
        if x_b < x_next2: # OK to follow df_b line to U then follow U to x_next
            t2 = T2
        dict_ = {
            'x_a':x_a,
            'x_b':x_b,
            'df_a':df_a,
            'df_b':df_b,
            'f_next_max':t,
            'f_next_min':b,
            'f_next2_max':t2,
            'f_next2_min':b2}
        return GO(dict_)
    def peron(self, # GUN instance
              Nf,   # Number of intervals in range of f(x)
              ):
        ''' Calculate and assign to self pi, P and cond, and return Q.
        '''
        import MaxH
        from scipy.sparse.linalg import LinearOperator
        g_L,g_U = (np.log(r) for r in self.LU(1.0))
        step = (g_U-g_L)/Nf
        self.W = step
        S_g = np.arange(g_L,g_U+step/3,step)
        S_r = np.exp(S_g) # Sampled ratio (f/\tilde f) points
        L = S_r[:-1]      # Factors for lower edges of intervals
        U = S_r[1:]       # Factors for upper edges of intervals
        C = (U+L)/2       # Factors for centers of intervals
        assert len(C) == Nf
        # Next, map from factors to EOS values.  Will use U and L for
        # adjacency and C for moments.
        self.f = self.s[0]*C
        L = np.outer(self.s[:3],L)
        U = np.outer(self.s[:3],U)
        class I_DATA(object):
            def __init__(self):
                self.j_bottom = Nf
                self.j_top = -1
                self.kj = []
                return
            def enter(self, j, k_bottom, k_top):
                self.kj.append([k_bottom, k_top])
                self.j_top = j
                if j < self.j_bottom:
                    self.j_bottom = j
                return
            def reformat(self, n):
                assert self.j_top >= self.j_bottom
                assert len(self.kj) > 0
                self.n = n
                self.j_top += 1
                self.kj = np.array(self.kj, np.int32)
                return n + self.j_top - self.j_bottom
            def j_range(self):
                return (self.j_bottom, self.j_top)
            def k_range(self,j):
                '''Return pair that defines the range of allowed k
                values given i=self and j
                '''
                kk = self.kj[j-self.j_bottom]
                return (kk[0], kk[1])
            def n_j(self, j):
                '''Return the unique index in range [0, N_states]
                corresponding to i=self and j.
                '''
                return self.n + j-self.j_bottom                
            def has_j(self, j):
                if self.j_bottom <= j and self.j_top > j:
                    return True
                else:
                    return False
        N_states = 0
        i_list = []
        jk = set()   # (j,k) in jk if allowed.  Only for check match
        for i in range(0,Nf):
            # i: index at t=0, j: index at t=1, k: index at t=2
            i_go_L = self.recursive(0, L[0, i])
            j_bottom = max(0,np.searchsorted(L[1,:], [i_go_L.f_next_min])-1)
            i_go_U = self.recursive(0, U[0, i])
            j_top = min(Nf, np.searchsorted(U[1,:], [i_go_U.f_next_max])+1)
            i_data = I_DATA()
            for j in range(j_bottom, j_top):
                j_go_U = self.recursive(1, U[1, j])
                j_go_L = self.recursive(1, L[1, j])
                top = min(U[2,-1], j_go_U.f_next_max, i_go_U.f_next2_max)
                bottom = max(
                    L[2,0],          # epsilon constraint
                    (L[1,j]*(self.dx[0]+self.dx[1])-U[0,i]*self.dx[1])/
                    self.dx[0],      # convexity constraint
                    j_go_L.f_next_min,
                    i_go_L.f_next2_min
                )
                if bottom > top:
                    continue
                k_top = min(Nf, np.searchsorted(U[2,:], [top])[0]+1)
                k_bottom = max(0, np.searchsorted(L[2,:], [bottom])[0]-1)
                i_data.enter(j, k_bottom, k_top)
                for k in range(k_bottom, k_top):
                    jk.add((j,k))
            N_states = i_data.reformat(N_states)
            i_list.append(i_data)
        # Verify that legal values of (i,j) are the same as legal (j,k)
        for i in range(Nf):
            for j in range(Nf):
                A = ((i,j) in jk)
                B = i_list[i].has_j(j)
                assert A == B,(
                    'i=%d, j=%d, (i,j) in jk = %s, (i,j) in ij = %s'
                    %(i, j, A, B))
        
        #import C
        #A = C.LO(i_list, N_states, Nf)
        #AT = C.LOT(i_list, N_states, Nf)
        A = LO(i_list, N_states, Nf)
        AT = LOT(i_list, N_states, Nf)
        Q = QLO(i_list, N_states, Nf)
        self.pi, self.P_tran = MaxH.P_calc(A, AT) # stationary and transition
        self.p_stat = self.pi/self.W**2
        # Diagnostic: Conditional given maximum of stationary dist
        cond = np.zeros(N_states)
        i = int(Nf/2)
        #cond[ij[(i,i)][2]] = 1
        cond[i_list[i].n_j(j)] = 1
        cond = Q*(self.P_tran.rmatvec(cond))
        self.cond = cond
        # End of diagnostic
        return Q
    def stationary(self, # GUN instance
                   Nf,   # Number of intervals in range of f(x)
                   ):
        """Calculate transition matrix A and stationary distribution
        for discrete approximation of P(g).  Characterize in terms of
        mean and covariance.

        Note: g = \log( \frac{f}{\tilde f} )
              y = \log(x)

        """
        from scipy.sparse.linalg import LinearOperator
        import scipy.sparse as ssm

        Q = self.peron(Nf)
        # Create sparse adjacency matrix
        n, N_states = Q.shape
        assert n == Nf,'n=%d, Nf=%d, N_states=%d'%(n,Nf,N_states)
        self.Pf = Q*self.pi
        return
    
def explore():
    '''For looking at sensitivity of time and results to Nx and Nf, the
    number of quantization levels in log(x) and log(f) respectively.
    Roughly Nf > Nx/10 is OK for the stationary distribution.  I have
    not found Nf that yields convergence of the ACF.  If Nx is too
    small, neither the monotonic nor the convex constraint are active.
    Roughly Nx > 1500 is OK.  Picking Nx=2000, Nf=300, takes less than
    a minute of cpu time.

    '''
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np
    C = 2.56e10 # EOS parameter
    xi = .4     # Initial x
    xf = 4.0    # Final x
    mass = 100.0
    dev = 0.05
    Nacf = 400 # Number of auto-covariance points
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_xlabel('f')
    ax1.set_ylabel('Pf')
    ax2 = fig.add_subplot(1,2,2)
    #ax2.set_xlabel(r'$x$')
    #ax2.set_ylabel('ACF')
    for Nx in np.arange(300,301,300):
        Nacf = Nx
        for f in (1, 2,):
            Nf = int(Nx*f)
            #for Nf in np.arange(int(Nx*.50), int(Nx*2.51), int(Nx*0.5)):
            try:
                print('Nx=%d Nf=%d'%(Nx,Nf))
                gun = GUN(C,xi,xf,mass,Nx,dev)
                gun.stationary(Nf,)
                # Plot probability density function
                ax1.plot(gun.f,gun.Pf/gun.W,label=r'Nx=%d Nf=%d'%(Nx,Nf))
                # Plot Conditional distribution
                X = np.arange(0,Nf,1.0)/float(Nf)
                ax2.plot(X, gun.cond*Nf, label=r'Nx=%d Nf=%d'%(Nx,Nf))
            except:
                print('Nx=%d Nf=%d failed'%(Nx,Nf))
                raise
    ax1.legend(loc='lower left')
    ax2.legend()
    plt.show()
    
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    explore()
    #_test()

#---------------
# Local Variables:
# mode: python
# End:
