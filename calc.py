"""calc.py: Classes for EOS and gun simulation

>>> Nx = 10     # len(self.x)
>>> C = 2.56e10 # EOS parameter
>>> xi = .4     # Initial x
>>> xf = 4.0    # Final x
>>> mass = 100.0
>>> dev = 0.2   # +/- 10%
>>> gun = GUN(C,xi,xf,mass,Nx,dev)
>>> gun.set_a_b(-1.0,-0.4)
>>> print('E(2.0)=%.3e'%gun.E(2.0))
E(2.0)=7.680e+10
>>> print('x_dot(2.0)=%.3e'%gun.x_dot(2.0))
x_dot(2.0)=3.919e+04
>>> print('dT_dx(2.0)=%.3e'%gun.dT_dx(0.0,2.0))
dT_dx(2.0)=2.552e-05
>>> print('T(4.0)=%.3e'%gun.T(gun.x)[-1])
T(4.0)=9.712e-05
>>> gun.ET()
>>> c,v = gun.edge(gun.e_0(),gun.e_1(),np.zeros(Nx),5)
>>> format='c.shape=%s, v.shape=%s, c[0,0]=%.1f, v[0,0]=%.2e'
>>> print(format%(c.shape, v.shape, c[1,0], v[0,0]))
c.shape=(2, 5), v.shape=(5, 10), c[0,0]=0.0, v[0,0]=5.65e+07

>>> Nx = 200
>>> dev = 0.01
>>> Nf = 10      # Number of quantization levels of f at each x
>>> Nacf = 20    # Number of auto-covariance points
>>> gun = GUN(C,xi,xf,mass,Nx,dev)
>>> gun.stationary(Nf)

Note: I use "+step/3" in calls to np.arrange to get N+1 items.
"""
import numpy as np
from scipy.integrate import quad, odeint
import numpy.linalg.linalg as LA, scipy.interpolate
class GO(object):
    """Generic object to hold variables.  magic.C is prettier than
    magic['C'].  Test suggests assignment to a new attibute of an
    instance is OK.
    """
    def __init__(self,d):
        for key,value in d.items():
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
              
def acf2d(ACF,       # A 1-d auto-covariance function
          scale=None # A 1-d array for scaling the result
          ):
    ''' Derive a 2-d auto-covariance function "S" from a 1-d
    auto-covariance function "ACF" using S[i,j] =
    ACF[abs(i-j)]*scale[i]*scale[j]
    '''
    n = len(ACF)
    S = np.empty((n,n))
    for i in range(n):
        S[i,i:] = ACF[:n-i]
        S[i,:i] = ACF[1:i+1][-1::-1]
    if scale is None:
        return S
    S *= scale
    return S.T*scale
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
        self.eos = lambda t: self.C/t**3
        self.s =self.eos(self.x_c)  # Nominal eos at sample points
        def dT_dx(t, x):
            ''' a service function called by odeint to calculate
            muzzle time T.  Since odeint calls dT_dx with two
            arguments, dT_dx has the unused t argument.
            '''
            # Since self.E(self.xi) = 0, I put a floor of self.x_c[0] on x
            x = max(x,self.x_c[0])
            x = min(x,self.xf)
            return np.sqrt((self.m/(2*self.E(x))))
        self.dT_dx = dT_dx
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
        ''' FixMe: This will calculate and assign to self or return:
        pi, P, Q, and cond.  Rather than creating A and Q as sparse
        matrices, I will create them as LinearOperators.
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
        
        import C
        A = C.LO(i_list, N_states, Nf)
        AT = C.LOT(i_list, N_states, Nf)
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
    def envelope_(self, Nf):
        g_L,g_U = (np.log(r) for r in self.LU(1.0))
        step = (g_U-g_L)/Nf
        self.W = step
        S_g = np.arange(g_L,g_U+step/3,step)
        S_r = np.exp(S_g) # Sampled ratio (f/\tilde f) points
        L = S_r[:-1]      # Factors for lower edges of intervals
        U = S_r[1:]       # Factors for upper edges of intervals
        # Next, map from factors to EOS values.  Will use U and L for
        # adjacency and C for moments.
        L = np.outer(self.s[:3],L)
        U = np.outer(self.s[:3],U)
        envelope = np.empty((Nf, Nf, 2))
        envelope[:,:,:] = np.nan     # To hold range of allowed values
        for i in range(0,Nf):
            # i: index at t=0, j: index at t=1, k: index at t=2
            for j in range(Nf):
                top = min(U[2,-1],   # epsilon constraint
                          U[1,j])    # Monotonic
                bottom = max(
                    L[2,0],          # epsilon constraint
                    (L[1,j]*(self.dx[0]+self.dx[1])-U[0,i]*self.dx[1])/
                    self.dx[0]       # convexity constraint
                    )
                if bottom > top:     # empty intersection
                    break
                envelope[i,j,:] = (bottom,top)
        self.envelope = envelope
    def stationary(self, # GUN instance
                   Nf,   # Number of intervals in range of f(x)
                   short = False # Skip ACF
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
        if short:
            return
        self.mean = np.dot(self.Pf,self.f)        # Expected value of f(0)
        n_s = len(self.s)
        ACF = np.zeros(n_s)
        diff = self.f-self.mean
        vn = Q.rmatvec(diff)*self.pi
        diff /= self.s[0]
        for n in range(n_s): # acf(n) \equiv EV (f(0)-mean)(f(n)-mean)
            ACF[n] = np.dot(Q*vn,diff*self.s[n])
            vn = self.P_tran.rmatvec(vn)
        tilde = acf2d(ACF,self.dx*self.s/self.s[0])
        _vals,_vecs = LA.eigh(tilde)
        i = np.argsort(_vals)[-1::-1]
        self.cov_vals = _vals[i].flatten()
        self.cov_vecs = _vecs.T[i,:]
        self.ACF = ACF
        self.Cov = acf2d(ACF, self.s/self.s[0])
        # self.inner(self.inner(a,Cov),b) gives covariance of a and b
        ### 6 lines to verify that I understand the spectral decomposition
        vecT2 = self.cov_vecs[2]
        a = np.dot(vecT2, np.dot(tilde, vecT2))
        b = self.cov_vals[2]
        assert np.abs((a-b)/b) < 1e-10,'a=%e b=%e'%(a, b)
        a = self.inner(vecT2, self.inner(self.Cov, vecT2))
        assert np.abs((a-b)/b) < 1e-10,'a=%e b=%e'%(a, b)
        ### End of verification
        return
    def mutual_information(self,NI):
        ''' Not used anymore
        '''
        A = self.P_tran
        P = np.matrix(np.diag(self.P_stat)+1e-30)
        I = -1.0*np.ones(NI) # Make empty after degugging
        H = lambda P: -(P.A*np.log(P.A)).sum()
        H0 = H(P)
        for k in range(NI):
            I[k] = 2*H0 - H(P)
            P *= A
        return I
    def inner(self, # GUN instance
              e,    # Array(s) representing one or more sampled functions
              f,    # 1-d array representing sampled function
              ):
        '''Calculate inner product(s) of discretely sampled functions.
        Since self.x_c[i] is the logarithmic center of an interval of
        length self.dx[i], functions should be sampled at self.x_c.
        '''
        return np.dot(e*self.dx,f)
    def ellipse(self, u, v, Marginal=False, M=500):
        '''In a 2-d subspace defined by vectors u and v and given
        covariance in terms self.cov_vecs and self.cov_vals, calculate
        and return sets of coefficients (x,y) that satisfy <x_i
        u,\Sigma^{-1}y_iv> = 1.  Default: Use conditional distribution
        in the subspace.  Alternative with Marginal=True: Use marginal
        distribution in the subspace.

        '''
        if Marginal:
            # First use spectral decomposition
            t = np.dot(np.array((u,v)), self.cov_vecs.T) # 2xL
            Q = np.dot(self.cov_vals.flatten()*t, t.T) # covar of u and v
            QI = LA.inv(Q)
            '''The above is (E * W)^T * Lambda * E * W where W has columns
            u and v and the rows of E are the eigenvectors.  Next I verify
            that the following is equivalent:

            t = np.array((u,v))
            Q = self.inner(t, self.inner(self.Cov,t.T))
            '''
            t = np.array((u,v))
            Q = self.inner(t, self.inner(self.Cov,t.T))
            error = np.eye(2) - np.dot(Q,QI)
            assert np.max(np.abs(error)) < 1e-10
            a = QI[0,0]
            b = QI[0,1]
            c = QI[1,1]
        else: # Use conditional variance
            def iqf(f,g):
                """ Inverse quadratic form <f,\Sigma^{-1}g>
                """
                # get f and g in diagonal basis
                t = np.dot(np.array((f,g)), self.cov_vecs)
                return np.dot(t[0],t[1]/self.cov_vals.flatten())
            a = iqf(u,u)
            b = iqf(u,v)
            c = iqf(v,v)
        step = 2*np.pi/M
        theta = np.arange(0,2*np.pi+0.5*step,step)
        sin = np.sin(theta)
        cos = np.cos(theta)
        rr = 1/(a*cos*cos + 2*b*cos*sin + c*sin*sin)
        r = np.sqrt(rr)
        return (r*cos,r*sin)
    def set_eos(self,    # GUN instance
                F=None): # Vector of eos perturbations at self.x_c
        if F == None:    # Restore nominal eos
            self.eos = lambda t: self.C/t**3
            return
        N = len(F)
        x = np.empty(N+2)
        y = np.empty(N+2)
        x[1:-1] = self.x_c
        y[1:-1] = F + self.s
        # Next 4 lines extrapolate beyond [xi,xf] for np.integration routines
        x[0] = 2*self.xi - x[1]
        x[-1] = 2*self.xf - x[-2]
        y[0] = y[1] + (y[1]-y[2])*(x[0]-x[1])/(x[1]-x[2])
        y[-1] = y[-2] + (y[-2]-y[-3])*(x[-1]-x[-2])/(x[-2]-x[-3])
        self.eos = scipy.interpolate.interp1d(
                x,y,kind='linear',bounds_error=True)
        return
    def set_a_b(self,   # GUN instance
                a, b,   # Pair of coefficients for perturbation
                ):
        ''' Perturb self.eos by linear combination of e_0 and e_1.
        The function self.eos can take a scalar or an array argument.
        '''
        self.eos = lambda t: self.C/t**3 + a*self.e_0(t) + b*self.e_1(t)
    def E(self, # GUN instance
          x     # Scalar position at which to calculate energy
          ):
        ''' Integrate eos between xi and x using numpy.integrate.quad
        to get energy of projectile at position.
        '''
        rv, err = quad(self.eos,self.xi,x)
        return rv
    def x_dot(self, # GUN instance
              x     # Scalar position /cm at which to calculate velocity
              ):
        ''' Calculate the projectile velocity at position x
        '''
        return np.sqrt(2*self.E(x)/self.m)
    def T(self,  # GUN instance
          x      # Array of positions /cm at which to calculate time
          ):
        ''' Calculate projectile time as a function of position
        '''
        rv = odeint(self.dT_dx, # RHS of ODE
                         0,     # Initial value of time
                         x,     # Solve for time at each position in x
                         )
        return rv.flatten()
    def q_0(self,    # GUN instance
            x=None,  # Position/s /cm at which to calculate q_0
            ):
        ''' Returns q_0 = 1.0 \forall x.  Only trick is to figure out
        how many ones to return.
        '''
        if x== None:
            x = self.x_c
        try:
            return np.ones(x.shape)
        except:
            return 1.0
    def q_1(self,   # GUN instance
            x=None, # Position/s /cm at which to calculate q_1
            ):
        ''' Returns q_1(x) for scalar x or array x
        '''
        if x== None:
            x = self.x_c
        if x[0] <= self.xi:
            raise RuntimeError('q_1(x) is not defined for x<=%f. x[0]=%f'%(
                self.xi, x[0]))
        return np.sqrt(self.m/self.C**3)*self.xi**3*(
                (x**2-2*self.xi**2)/np.sqrt(x**2-self.xi**2)
                -
                (self.xf**2-2*self.xi**2)/np.sqrt(self.xf**2-self.xi**2)
                )
    def e_1(self,    # GUN instance
            x=None,  # Position/s /cm at which to calculate e_1
            ):
        ''' Returns e_1(x), a perturbation function used in gun.tex,
        for scalar x or array x
        '''
        if x== None:
            x = self.x_c
        return -1/x
    def e_0(self,    # GUN instance
            x=None,  # Position/s /cm at which to calculate e_0
            ):
        ''' Returns e_0(x), a perturbation function used in gun.tex,
        for scalar x or array x.  By coincidence e_0 = q_0, but the
        notions are different.
        '''
        if x== None:
            x = self.x_c
        try:
            return np.ones(x.shape)
        except:
            return 1.0
    def ET(self,    # GUN instance
           ):
        '''Construct set of perturbations that are as high as possible
        and as low as possible and switch as fast as possible between
        those extremes.  This set explores the limits of the set of
        possible E and T pairs.
        '''
        N = len(self.x_c)
        z1 = np.empty((2*N+1,N))
        z = z1[:2*N] # Calculate 2N functions then copy first to last position
        bot = z[:N,:]    # bot points to the first N functions (start low)
        top = z[N:2*N,:] # top points to the second N functions (start high)
        # bot starts low and ends high.  top starts high and ends low.
        # Set the first two elements of top functions
        top[:,0] = self.LUi(self.s, 0)[1]
        top[:,1] = self.LUi(self.s, 1)[1]
        top[0,1] = self.LUi(self.s, 1)[0]  # Start first downward transition
        dx = self.dx                       # Just abbreviation
        for i in range(2,N):
            L,U = self.LUi(self.s,i) # Constraints from self.dev
            lower = np.maximum(             # Lower traces at i
                ((dx[i-2]+dx[i-1])*z[:,i-1] -
                 dx[i-1]*z[:,i-2])/dx[i-2], # Convexity
                L                           # Constraint of self.dev
            )
            upper = np.minimum(z[:,i-1],U)  # Upper traces at i
            top[:i,i] = lower[N:i+N]
            top[i:,i] = upper[N+i:]
        # Going backwards, set the first two elements of bot functions
        bot[:,N-1] = self.LUi(self.s, N-1)[1]
        bot[:,N-2] = self.LUi(self.s, N-2)[1]
        bot[0,N-2] = self.LUi(self.s, N-2)[0]
        for i in range(N-3,-1,-1):
            L,U = self.LUi(self.s,i) # Constraints from self.dev
            lower = np.maximum(             # Lower trace at i
                ((dx[i+2]+dx[i+1])*z[:,i+1] -
                 dx[i+1]*z[:,i+2])/dx[i+2], # Convexity
                L                           # Constraint of self.dev
            )
            upper = np.maximum(z[:,i+1],U) # Upper trace at i
            bot[:i,i] = upper[:i]
            bot[i:,i] = lower[i:N]
        z -= self.s
        z1[-1] =z[0] # Duplicate first value at last position to close loop
        self.ETz = z1
        return
    def edge(self,   # GUN instance
             h0,     # first direction
             h1,     # second direction
             center, # Shift from 0 to center
             M       # number of angles to use
             ):
        '''Find M boundary points of allowed polytope in subspace
        spanned by h0 and h1.  Used for plotting.
        '''
        step = 2*np.pi/(M-1)
        theta = np.arange(0,2*np.pi+step/3,step)
        sin = np.sin(theta)
        cos = np.cos(theta)
        v = np.outer(cos,h0) + np.outer(sin,h1) # Array of directions
        assert v.shape==(M,len(self.x_c)),'v.shape=%s, M=%d, len(self.x)=%d'%(
            v.shape, M, len(self.x_c)) # v[i,:] is a perturbation of the EOS
        r = np.ones(M)
        rb = r*1.0        # Radius Big
        for i in range(100):
            OK = self.test(v,rb,center)
            rb += rb*OK   # double the length of those inside
            if not OK.sum():
                break     # all rb rays are outside
            assert i < 90 # Error if many iterations to get all outside
        rs = r*1.0        # Radius Small
        for i in range(100):
            OK = self.test(v,rs,center)
            rs -= rs*(1-OK)/2  # Halve the length of those outside
            if OK.prod():
                break     # all rs rays are inside
            assert i < 90 # Error if many iterations to get all inside
        for i in range(100):
            rc = (rs+rb)/2
            OK = self.test(v,rc,center)
            rs = rs + OK*(rc-rs)
            rb = rc + OK*(rb-rc)
            re = (rb-rs)/rs # Radius error
            if re.max() < 0.01:
                break
            assert i < 90
        r = (rs+rb)/2
        border_vectors = (v.T*r).T
        border_coefficients = np.array([r*cos,r*sin])
        return (border_coefficients,border_vectors)
    def test_z(self, # GUN instance
               z     # Array of eos functions, z[:,i] is the ith function
    ):
        """Return R, a vector of bools: R[i] is True if z[:,i] satisfies all
        constraints.

        """
        monotonic = (z[:,1:] <= z[:,:-1])
        # dx(i-1)*z(i+1) + dx(i)*z(i-1) > (dx(i) + dx(i-1))*z(i) see
        # eq:constraint1 of gun.tex
        dx_i = self.dx[1:-1]
        dx_i_m1 = self.dx[:-2]
        t1 = dx_i_m1*z[:,2:] # dx(i-1)*z(i+1)
        t2 = dx_i*z[:,:-2] # dx(i)*z(i-1)
        t3 = (dx_i_m1+dx_i)*z[:,1:-1]   # (dx(i-1)+dx(i))*z(i)
        convex = t1 + t2 > t3 # eq:constraint a
        L,U = self.LU(self.s)
        upper = (z < U)
        lower = (z > L)
        rv = 1
        for test in (monotonic,convex,upper,lower):
            rv *= test.prod(axis=1)
        return rv
    def test(self,   # GUN instance
             v,      # Array of directions.  v[i,:] is a direction
             r,      # Array of distances
             center  # An offset
             ):
        ''' Return Boolean vector with True/False if component
        satisfies/violates constraints.  The ith component is self.s +
        center + r[i]*v[i,:].  Used only in edge()
        '''
        M,T = v.shape
        assert len(r) == M       # Number of directions or functions
        assert len(center) == T  # Length of each funciton
        assert self.s.shape == center.shape
        z = self.s + (v.T*r).T + center
        return self.test_z(z)

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
        for f in (1, 2, 5, 10):
            Nf = int(Nx*f)
            #for Nf in np.arange(int(Nx*.50), int(Nx*2.51), int(Nx*0.5)):
            try:
                print('Nx=%d Nf=%d'%(Nx,Nf))
                gun = GUN(C,xi,xf,mass,Nx,dev)
                gun.stationary(Nf, short=True)
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
