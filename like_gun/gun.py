"""gun.py: Class for gun experiment.  Goals are: to develop, analyze
and understand a procedure for estimating an isentrope on the basis of
data and simulations.

"""
import numpy as np

from eos import Go
import fit
magic = Go(
    x_i=0.4,             # Initial position of projectile / cm'
    x_f=4.0,             # Final/muzzle position of projectile /cm'
    m=100.0,             # Mass of projectile / g
    area=1e-4,           # Projectile cross section in m^2
    newton2dyne=1e5,     # dynes per newton
    var=1.0e-0,          # Variance attributed to v measurements
    n_t=500,             # Number of ts for t2v spline
    t_min=-5.0e-6,       # Range of times for t2v spline
    t_max=110.0e-6,      # Range of times for t2v spline
    cm2km=1.0e5,         # For conversion from cm/sec to km/sec
    n_t_sim=1000,        # len(vt), simulated velocity time pairs
           )
magic.add(D_frac=fit.magic.D_frac) # Fractional finite difference for dv/df
class Gun:
    '''Represents an imagined experiment and actual simulations.

    Section 2 of the document ../juq.tex describes the imagined
    experiment.

    Units: cgs

    Methods:

    f(x)          Force in dynes on projectile
    
    E(x)          Energy

    x_dot(x)      Velocity

    shoot(...)    Integrate ode(eos) to get t,x,v
    
    fit_t2v(...)  Fit t->v spline

    compare(vt,c) Return d_ep/d_c, ep, Sigma_inv

    fit_C()       Calculates derivative of velocity wrt to eos in
                  terms of spline coefficients

    fit_B_ep(vt)  Calculates ep = errors of predictions of experimental
                  velocities and B = dv/dc_v

    log_like(...) Calculate the exponent of the likelihood function
    
    '''
    from eos import Spline_eos
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
        '''
        return magic.newton2dyne*magic.area*self.eos(x)
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
    def shoot(
            self,              # Gun instance
            t_min,
            t_max,
            n_t,
            ):
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
        t = np.linspace(t_min, t_max, n_t)
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
    def fit_t2v(
            self,
            t_min=magic.t_min,
            t_max=magic.t_max,
            n_t=magic.n_t,
            ):
        '''Run a simulation and build a spline that maps times to
        velocities.  Return the spline.
        '''
        from eos import Spline
        t, xv = self.shoot(t_min, t_max, n_t)
        # xv is array of calculated positions and velocities at times in t
        return Spline(t,xv[:,1])
    def compare(
            self,
            vt,
            c=None
            ):
        ''' Calculate:
        ep = prediction - data
        D = d prediction / d c_eos
        '''
        if type(c) != type(None):
            self.eos = self.eos.new_c(c)
        C = self.fit_C()
        B,ep = self.fit_B_ep(vt)
        D = np.dot(B,C)
        Sigma_inv = np.diag(np.ones(len(ep))/self.var)
        return (D, ep, Sigma_inv)
    def fit_C(
            self,                   # Gun instance
            fraction=magic.D_frac,  # Finite difference fraction
            t_min=magic.t_min,
            t_max=magic.t_max,
            n_t=magic.n_t
            ):
        '''Calculate dv/df in terms of spline coefficients and return
        '''
        # Spline.new_c(c) does not modify Spline.  It returns a copy
        # of Spline with the coefficients c.
        eos_nom = self.eos
        c_f_nom = eos_nom.get_c()    # Nominal coefficients for eos
        t2v_nom = self.fit_t2v(t_min, t_max, n_t)
        c_v_nom = t2v_nom.get_c()
        C = np.empty((len(c_v_nom),len(c_f_nom))) # To be dv/df matrix
        for i in range(len(c_f_nom)):
            c_f = c_f_nom.copy()
            # Next set size of finite difference for derivative approximation
            d_f = float(c_f[i]*fraction)
            c_f[i] += d_f
            # Next run a simulation to get a v(t) spline a for a modified eos
            self.eos = eos_nom.new_c(c_f)
            C[:,i] = (self.fit_t2v(t_min, t_max, n_t).get_c() -c_v_nom)/d_f
        self.eos = eos_nom
        return C
    def fit_B_ep(
            self, # Gun instance
            vt,   # (velocities, times)
            ):
        '''Map experimental velocities v and times t to the error ep
        and B = dv/dc_v more specifically:
        
        ep[i]=v[i] - t2v(t[i]) Difference between simulation and data
        
        B[i,j] = b_j(t[i]) Where b_j is the jth basis function for
                                the t2v spline
        '''
        v,t = vt
        assert len(t) == len(v)
        t2v = self.fit_t2v()
        c = np.zeros(t2v.get_c().shape)
        ep = v - t2v(t)   # Calculate errors
        B = np.zeros((len(v), len(c)))
        for j in range(len(c)):
            c[j] = 1.0
            delta_t2v = t2v.new_c(c)
            B[:,j] = delta_t2v(t)
            c[j] = 0.0
        return B, ep
    def log_like(
            self, # Gun instance
            D,
            ep,
            Sigma_inv,
            ):
        ''' Assume t are accurate and that for model velocities m

            L(m) = log(p(v|m)) =\sum_t - \frac{(v-m)^2}{2\sigma^2}

        '''
        return -np.dot(ep, np.dot(Sigma_inv, ep))/2
    def debug_plot(self, vt, key, show=False):
        import matplotlib.pyplot as plt

        if show:
            self.ax1.legend(loc='lower right')
            self.ax2.legend()
            plt.show()
            return
        t_mic = vt[1]*1e6
        plotv = lambda v,s: self.ax1.plot(
            t_mic, v/1e5, label=r'$v_{{\rm {0}}}$'.format(s))
        if not hasattr(self, 'fig'):
            self.fig = plt.figure('debug gun',figsize=(8,12))
            
            self.ax1 = self.fig.add_subplot(2,1,1)
            self.ax1.set_xlabel(r'$t/(\mu \rm sec)$')
            self.ax1.set_ylabel(r'$v/(\rm{km/s})$')
            plotv(vt[0],'exp')
            
            self.ax2 = self.fig.add_subplot(2,1,2)
            self.ax2.set_xlabel(r'$x/\rm cm$')
            self.ax2.set_ylabel(r'$f/{\rm dyn}$')
        plotv(self.fit_t2v()(vt[1]),key)
        x = np.linspace(magic.x_i, magic.x_f, 500)
        self.ax2.plot(x, self.f(x), label=r'$f_{{\rm {0}}}$'.format(key))
        t2v = self.fit_t2v()
        return

def data():
    '''Make "experimental" data from gun with eos from eos.Experiment
    '''
    from eos import Experiment
    gun = Gun(Experiment())
    # Make simulated measurements
    t2v = gun.fit_t2v()
    t = np.linspace(0, magic.t_max,magic.n_t_sim)
    v = t2v(t)
    return (v, t)
# Test functions
import numpy.testing as nt
close = lambda a,b: a*(1-1e-7) < b < a*(1+1e-7)
def test_log_like():
    from eos import Nominal, Spline_eos
    vt = (data())
    gun = Gun(Spline_eos(Nominal()))
    ll = gun.log_like(*gun.compare(vt))
    if close(-ll, 970963830.012):
        return 0
    else:
        return 1
def test_shoot_t2v():
    from eos import Experiment
    gun = Gun(Experiment())
    t,xv = gun.shoot(magic.t_min, magic.t_max, magic.n_t)
    t2v = gun.fit_t2v()
    v_ = t2v(t[-1])
    this_computation = list(xv[-1,:]) + [t2v(t[-1])]
    old_results = (4.52167252, 4.12180082e+04, 4.12180082e+04)
    for old, this, name in zip(old_results, this_computation,
                               'x v spline'.split()):
        assert close(old,this),'{0}: old={1:.8e}, new={2:.8e}'.format(
            name, old, this)
    return 0
def test_x_dot():
    from eos import Experiment
    v = Gun(Experiment()).x_dot(4.0)
    assert close(v, 4.121800824e+04)
    return 0
def test_C():
    from eos import Experiment, Spline_eos

    n_t = 15
    N = 10
    eos = Spline_eos(Experiment(), N=N, v_min=.38, v_max=4.2)
    C = Gun(eos).fit_C(n_t=n_t)
    assert C.shape == (n_t, N)
    assert close(C[1,0], 2.05695528e-07)
    assert close(C[-1,-1], 6.45823399e-07)
    return 0
def test_B_ep():
    ''' B[i,j] = dv(t[i])/dc_velocity[j]
    ep[i] = v_simulation(t[i]) - v_experiment(t[i])
    '''
    from eos import Nominal, Spline_eos
    vt = (data())
    eos = Spline_eos(Nominal())
    B,ep = Gun(eos).fit_B_ep(vt)
    assert B.shape == (1000,500)
    assert ep.shape == (1000,)
    assert np.argmax(ep) == 345
    assert close(ep[345], 1461.8632379214614)
    assert np.argmax(B[300,:]) == 165
    assert close(B[300,165], 0.66576300914271824)
    return 0
def test_Pq():
    '''
    '''
    from eos import Nominal, Spline_eos
    vt = (data())
    eos = Spline_eos(Nominal(),precondition=False)
    c = eos.get_c()
    gun = Gun(eos)
    P,q = eos.Pq_like(*gun.compare(vt, c))
    assert P.shape == (50,50)
    assert q.shape == (50,)
    assert P[23,23] == P.max()
    value = 3.286529435e-10
    assert close(P.max(), value), 'P.max={0:.9e} != {1:.9e}'.format(
        P.max(), value)
    i = 22
    assert np.argmin(q) == i, 'argmin={0} != 11'.format(
        np.argmin(q), i)
    value = 5.680863201e-01
    assert close(-q[i], value), '-q[{2}]={0:.9e} != {1:.9e}'.format(
        -q[i], value, i)
    return 0    
def test():
    for name,value in globals().items():
        if not name.startswith('test_'):
            continue
        if value() == 0:
            print('{0} passed'.format(name))
        else:
            print('\nFAILED            {0}\n'.format(name))
    return 0
    
def work():
    ''' This code for debugging stuff will change often
    '''
    import matplotlib.pyplot as plt
    from eos import Nominal, Spline_eos
    vt = (data())
    gun = Gun(Spline_eos(Nominal(),precondition=False))
    gun.debug_plot(vt, 'test')
    gun.debug_plot(None, None, show=True)
    return 0
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) >1 and sys.argv[1] == 'test':
        rv = test()
        sys.exit(rv)
    if len(sys.argv) >1 and sys.argv[1] == 'work':
        sys.exit(work())
    main()

#---------------
# Local Variables:
# mode: python
# End:
