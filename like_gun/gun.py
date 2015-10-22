"""gun.py: Classes for gun experiment.  Goals are: to develop, analyze
and understand a procedure for estimating an isentrope on the basis of
data and simulations.

"""
import numpy as np
import matplotlib.pyplot as plt
    
class go:
    ''' Generic object.  For storing magic numbers.
    '''
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
magic = go(
    x_i=0.4,             # Initial position of projectile / cm'
    x_f=4.0,             # Final/muzzle position of projectile /cm'
    m=100.0,             # Mass of projectile / g
    var=1.0e0,           # Variance attributed to v measurements
    n_t=500,             # Number of ts for t2v spline
    t_min=-5.0e-6,       # Range of times for t2v spline
    t_max=110.0e-6,      # Range of times for t2v spline
    D_frac=2.0e-2,       # Fractional finte difference for esimating dv/df
    cm2km=1.0e5,         # For conversion from cm/sec to km/sec
    n_t_sim=1000,        # len(vt), simulated velocity time pairs
           )

class Gun:
    '''Represents an imagined experiment and actual simulations.

    Section 2 of the document ../juq.tex describes the imagined
    experiment.

    Units: cgs

    Methods:

    f(x)          Force in dynes on projectile
    
    E(x)          Energy

    x_dot(x)      Velocity

    shoot()       Integrate ode(eos) to get t,x,v
    
    set_t2v()     Fit t->v spline

    set_D         Calculates derivative of velocity wrt to eos in
                  terms of spline coefficients

    set_B_ep(vt)  Calculates ep = errors of predictions of experimental
                  velocities and B = dv/dc_v

    Pq(BD,ep_v)   Return P,q for a quadratic approximation of the log
                  likelihood.  For a small change d in the spline
                  coefficients, delta L = d*P*d + q*d

    log_like(vt)  Calculate the exponent of the likelihood function
    
    '''
    def __init__(
            self,            # Gun instance
            eos,             # Pressure(specific volume)
            x_i=magic.x_i,   # Initial position
            x_f=magic.x_f,   # Final position
            m=magic.m,       # Projectile mass
            var=magic.var,   # Variance v measurement errors
            ):
        self.eos = eos
        self.x_i = x_i
        self.x_f = x_f
        self.m = m
        self.var = var
        return
    def f(self, x):
        ''' Force in dynes.

        area = 1e-4 m^2
        1 Newton = 10^5 dynes

        eos(x) is in Pa = Newtons/m^2
        f = eos(x)*1e-4 Newtons = eos(x)*10 dynes
        '''
        return 10*self.eos(x)
    def E(self, # Gun instance
          x     # Scalar position at which to calculate energy
          ):
        ''' Integrate eos between x_i and x using numpy.integrate.quad
        to get energy of projectile at position x.
        '''
        from scipy.integrate import quad
        rv, err = quad(self.f, self.x_i, min(x,self.x_f))
        assert rv == rv # test for nan
        return rv
    def x_dot(self, # Gun instance
              x     # Position[s] /cm at which to calculate velocity
              ):
        '''Calculate the projectile velocity at a single position x, or
        if x is an array, calculate the velocity for each element of x
              '''
        if isinstance(x, np.ndarray):
            return np.array([self.x_dot(x_) for x_ in x])
        elif isinstance(x, np.float):
            if x <= self.x_i:
                return 0.0
            assert(self.E(x)>=0.0),'E(%s)=%s'%(x,self.E(x))
            return np.sqrt(2*self.E(x)/self.m) # Invert E = (mv^2)/2
        else:
            raise RuntimeError('x has type %s'%(type(x),))
    def shoot(self):
        '''Run a simulation and return the results: t, [x,v]
        '''
        from scipy.integrate import odeint
        def F(x,t):
            '''vector field for integration F((position,velocity),t) =
            \frac{d}{dt} (position,velocity)
            '''
            if t<0:
                return np.zeros(2)
            if x[0] > self.x_f: # beyond end of gun barrel, muzzle
                acceleration = 0.0
            else:
                acceleration = self.f(x[0])/self.m # F = MA
            return np.array([x[1], acceleration])
        t = np.linspace(magic.t_min, magic.t_max, magic.n_t)
        xv = odeint(
            F,            # 
            [self.x_i,0],
            t,
            atol = 1e-11, # Default 1.49012e-8
            rtol = 1e-11, # Default 1.49012e-8
            )
        assert xv.shape == (len(t),2)
        # xv is array of calculated positions and velocities at times in t
        return t, xv
    def set_t2v(self):
        '''Run a simulation and build a spline that maps times to
        velocities.  Return the spline and save it as self.t2v.
        '''
        from eos import Spline
        t, xv = self.shoot()
        # xv is array of calculated positions and velocities at times in t
        self.t2v = Spline(t,xv[:,1])
        return self.t2v
    def set_D(
            self,                   # Gun instance
            fraction=magic.D_frac   # Finite difference fraction
            ):
        '''Calculate dv/df in terms of spline coefficients and save as self.D.
              
        Note self.D.shape = (len(c_v)-4, len(c_f)-4) because I drop the
        last 4 which are always 0.
        '''
        # Spline.new_c(c) does not modify Spline.  It returns a copy
        # of Spline with the coefficients c and an updated provenance.
        eos_nom = self.eos
        end = eos_nom.end            # Length of padding on cubic splines
        c_f_nom = eos_nom.get_c()    # Nominal coefficients for eos
        n_f = len(c_f_nom) - end     # Dimension of optimization var
        t2v_nom = self.set_t2v()
        c_v_nom = t2v_nom.get_c()
        n_v = len(c_v_nom) - end
        self.D = np.empty((n_v,n_f)) # To be dv/df matrix
        for i in range(n_f):
            c_f = c_f_nom.copy()
            # Next set size of finite difference for derivative approximation
            d_f = float(c_f[i]*fraction)
            c_f[i] += d_f
            # Next run a simulation to get a v(t) spline a for a modified eos
            self.eos = eos_nom.new_c(c_f)
            c_v = self.set_t2v().get_c()
            D_i = ((c_v - c_v_nom)[:-end])/d_f
            self.D[:,i] = D_i
        self.eos = eos_nom
        self.t2v = t2v_nom
        return self.D
    def set_B_ep(
            self, # Gun instance
            vt,   # (velocities, times)
            ):
        '''Map experimental velocities v and times t to the error ep
        and B = dv/dc_v more specifically:
        
        self.ep[i]=v[i] - t2v(t[i]) Difference between simulation and data
        
        self.B[i,j] = b_j(t[i]) Where b_j is the jth basis function for
                                the t2v spline
        '''
        v,t = vt
        n_vt = len(v)
        assert len(t) == n_vt
        c_ = np.zeros(self.set_t2v().get_c().shape) # Also creates self.t2v
        self.ep = v - self.t2v(t)   # Calculate errors
        v_dim = len(c_) - magic.end # end values are always 0
        self.B = np.zeros((n_vt,v_dim))
        for j in range(v_dim):
            c_[j] = 1.0
            delta_t2v = self.t2v.new_c(c_)
            self.B[:,j] = delta_t2v(t)
            c_[j] = 0.0
        return self.B, self.ep
    def Pq(
            self,         # Gun instance
            BD,           # dv/df matrix
            ep_v,         # v_exp - v_sim
            ):
        '''From the section "A Posteriori Probability" of notes.tex,

        P = Sigma_f^{-1} + (BD)^T Sigma^{-1}_v BD
        q^T = ep_f^T Sigma_f^{-1} - ep_v^T Sigma_v^{-1} BD
        R = ep_f^T Sigma_f^{-1} ep_f + ep_v^T Sigma_v^{-1} ep_v

        This method calculates the contribution from the likelihood of
        the gun data, ie:
        
        P =  (BD)^T Sigma^{-1}_v BD
        q^T =  - ep_v^T Sigma_v^{-1} BD

        If eos.precondition use these \tilde values instead:

        \tilde P = U^{-1} P U^{-1} (variable t_P here)
        \tilde q^T = q^T U^{-1} (variable t_q here)
        '''
        P = np.dot(BD.T, BD)
        q = -np.dot(ep_v, BD)
        if self.eos.precondition:
            P = np.dot(self.eos.U_inv, np.dot(P, self.eos.U_inv))
            q = np.dot(self.eos.U_inv, q)
        return P, q
    def log_like(
            self, # Gun instance
            vt,   # Arrays of measured times and velocities
            ):
        ''' Assume t are accurate and that for model velocities m

            L(m) = log(p(v|m)) =\sum_t - \frac{(v-m)^2}{2\sigma^2}

        '''
        v,t = vt
        t2v = self.set_t2v()
        m = t2v(t)
        d = v-m
        return -np.dot(d/self.var,d)/2

def experiment():
    '''Make "experimental" data from gun with eos defined in eos.
    '''
    from eos import Experiment
    gun = Gun(Experiment())
    # Make simulated measurements
    t2v = gun.set_t2v()
    t = np.linspace(0, magic.t_max,magic.n_t_sim)
    v = t2v(t)
    return (v, t)
def work():
    ''' This code for debugging stuff will change often
    '''
    from eos import Experiment
    gun = Gun(Experiment())
    x = np.linspace(gun.x_i, gun.x_f, 500)
    f = Experiment()(x)
    e = np.array([gun.E(x_) for x_ in x])
    v = gun.x_dot(x)
    t,xv = gun.shoot()
    
    fig = plt.figure('t,x')
    ax = fig.add_subplot(1,1,1)
    ax.plot(t, xv[:,0])
    
    v_exp, t_exp = experiment()
    fig = plt.figure('exp')
    ax = fig.add_subplot(1,1,1)
    ax.plot(t_exp, v_exp)
    plt.show()
    return 0
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) >1 and sys.argv[1] == 'test':
        rv = test() + test_opt()
        sys.exit(rv)
    if len(sys.argv) >1 and sys.argv[1] == 'work':
        sys.exit(work())
    main()

#---------------
# Local Variables:
# mode: python
# End:
