"""calc.py: Classes for EOS and gun simulation.  Derived from calc.py
in parent directory.  Goals are:

1. Develop, analyze and understand a procedure for estimating an
isentrope on the basis of data and simulations.

2. Demonstrate provenance tracking using Component, Provenance and Float
   from cmf_models

Run using python2 and scipy 0.14

For the splines, I use
scipy.interpolate.InterpolatedUnivariateSpline. See:
https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py

For the spline optimization I use cvxopt.solvers.qp.  See:
http://cvxopt.org/userguide/coneprog.html

"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
from cmf_models import Component, Provenance, Float
from markup import oneliner
import matplotlib.pyplot as plt

class Spline(IU_Spline, Component):
    '''From the source:

    data = dfitpack.fpcurf0(x,y,k,w=w,xb=bbox[0],xe=bbox[1],s=s)
    n,t,c,k,ier = data[7],data[8],data[9],data[5],data[-1]
    self._eval_args = t[:n],c[:n],k
    
    t is the array of knots, c is the array of coefficients and k is the
    order.    
    '''
    def __init__(self, x, y, comment=''):
        IU_Spline.__init__(self, x, y)
        Component.__init__(self, self.get_c(), comment)
    def get_t(self):
        'Return the knot locations'
        return self._eval_args[0]
    def get_c(self):
        'Return the coefficients for the basis functions'
        return self._eval_args[1]
    def new_c(self,c):
        '''Return a new Spline instance that is copy of self except
        that the coefficients for the basis functions are c and
        provenance is updated.'''
        import copy
        from inspect import stack
        rv = copy.deepcopy(self)
        rv._eval_args = self._eval_args[0], c, self._eval_args[2]
        # stack()[1] is context that called new_c
        rv.provenance = Provenance(
            stack()[1], 'New coefficients', branches=[self.provenance],
            max_hist=50)
        return rv
    def display(self):
        '''This method serves the Component class and the make_html
        function defined in the cmf_models module.  It returns an html
        description of self and writes a plot to 'eos.jpg'.
        '''
        # Write file named eos.jpg
        fig = plt.figure('eos',figsize=(7,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel(r'$x/{\rm cm}$')
        ax.set_ylabel(r'$f/{\rm dyn}$')
        x = np.linspace(.4,4,100)
        y = self(x)
        ax.plot(x,y)
        fig.savefig('eos.jpg', format='jpg')

        # Make an html formated return value
        html = oneliner.p('''
        Table of coefficients for spline representation of force
        as a function of position along the barrel''')
        html += oneliner.p(self.get_c().__str__())
        html += oneliner.p('''
        Plot of force as a function of position along the barrel''')
        html += oneliner.img(
            width=700, height=500, alt='plot of eos', src='eos.jpg')
        return html
    
class go:
    ''' Generic object.  For storing magic numbers.
    '''
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
magic = go(
    n_t=500,              # Number of ts for t2v spline
    t_min=-5.0e-6,        # Range of times for t2v spline
    t_max=110.0e-6,       # Range of times for t2v spline
    n_x_pert=1000,        # Number of x points for perturbed EOS
    D_frac=2.0e-2,        # Fractional finte difference for esimating dv/df
    cm2km=1.0e5,          # For conversion from cm/sec to km/sec
    n_t_sim=1000,         # len(vt), simulated velocity time pairs
    fit_dim=50,           # Number of x points for EOS fit
    like_iter=50,         # Bound for iterations maximizing likelihood
    converge=1.0e-3,      # Fractional tolerance for max like
    stretch=1.25,         # Expansion of x beyond (xi,xf)
    k_factor=1.0e6,       # (f(xi)/f(xf))^2
    end=4,                # Last 4 coefficients of all splines are 0
           )

class GUN:
    '''Represents an imagined experiment and actual simulations.

    Section 2 of the document ../juq.tex describes the imagined
    experiment.

    Units: cgs
    '''
    def __init__(
            self,             # GUN instance
            C=Float(
                2.56e10,    # dynes (cm)^3
                'Constant in nominal equation of state: F = C/x^3 dynes',
                ),
            xi=Float(0.4, 'Initial position of projectile / cm'),
            xf=Float(4.0, 'Final/muzzle position of projectile /cm'),
            m=Float(
                100.0,
                '{0} of projectile / g'.format(
                    oneliner.a('Mass', href='http://en.wikipedia.org/wiki/Mass')
                )),
            N=400,            # Number of intervals between xi and xf
            sigma_sq_v=1.0e5, # Variance attributed to v measurements
            ):
        self.C = C + 0.0 # Vacuous demonstration of Float + float
        self.xi = xi
        self.xf = xf
        self.m = m
        self._set_N(N)
        self.sigma_sq_v = sigma_sq_v
        self.components = set(('C','xi','xf','m'))
        # Set of component instance names for exercising provenance tracking
        return
    def opt_init(self):
        ''' Assign the following to self:
        original_eos: Serves as mean of prior
        Sigma_f_inv:  Inverse covariance of prior
        U_inv:        Preconditioner for optimization
        '''
        import copy
        self.original_eos = copy.deepcopy(self.eos)
        c_f = self.original_eos.get_c()[:-magic.end]
        dev_f = 0.005*c_f # Sam Shaw's 1/2%
        self.Sigma_f_inv = np.diag(1.0/(dev_f*dev_f))
        self.U_inv = np.diag(dev_f)
    def _set_N(
            self,                 # GUN instance
            N,                    # Number of intervals between xi and xf
            stretch=magic.stretch # Extension of EOS beyond barrel
    ):
        '''Interval lengths are uniform on a log scale ie
               self.x[i+1]/self.x[i] = constant.
        '''
        self.x = np.logspace(
            np.log(self.xi.value/stretch),
            np.log(self.xf.value*stretch), N, base=np.e)
        self.set_eos(lambda x:self.C.value/x**3) # Initial eos
        return
    def set_eos(self, func):
        self.eos = func
    def set_eos_spline(
            self,   # GUN instance
            x,      # Positions of samples
            f       # Force at samples
            ):
        s = '''Initialize the eos with a spline fit to the arrays x
and f with elements that are positions along the gun
barrel and the forces at those positons respectively. '''
# The number of knots is len(x)+4.  The first 4 knots are at x[0] and
# t[4]=x[2].  Similarly the last 4 knots are at x[-1] and t[-5]=x[-3].
        self.eos = Spline(x, f, s)
        self.components.add('eos')
        return self.eos
    def E(self, # GUN instance
          x     # Scalar position at which to calculate energy
          ):
        ''' Integrate eos between xi and x using numpy.integrate.quad
        to get energy of projectile at position x.
        '''
        from scipy.integrate import quad
        rv, err = quad(self.eos,self.xi.value, min(x,self.xf.value))
        assert rv == rv # test for nan
        return rv
    def x_dot(self, # GUN instance
              x     # Position[s] /cm at which to calculate velocity
              ):
        '''Calculate the projectile velocity at a single position x, or
        if x is an array, calculate the velocity for each element of x
              '''
        if isinstance(x, np.ndarray):
            return np.array([self.x_dot(x_) for x_ in x])
        elif isinstance(x, np.float):
            if x <= self.xi.value:
                return 0.0
            assert(self.E(x)>=0.0),'E(%s)=%s'%(x,self.E(x))
            return np.sqrt(2*self.E(x)/self.m.value) # Invert E = (mv^2)/2
        else:
            raise RuntimeError('x has type %s'%(type(x),))
    def set_t2v(self):
        '''Run a simulation to build a spline for mapping times to
        velocities and save as self.t2v
        '''
        from scipy.integrate import odeint
        m = self.m.value   # mass
        f = self.eos # force
        xf = self.xf.value # end of gun barrel, muzzle
        def F(x,t):
            '''return x dot for x = (position,velocity)
            '''
            if t<0:
                return np.zeros(2)
            if x[0] > xf:
                acceleration = 0.0
            else:
                acceleration = f(x[0])/m # F = MA
            return np.array([x[1], acceleration])
        t = np.linspace(magic.t_min, magic.t_max, magic.n_t)
        self.t_max = t[-1]
        xv = odeint(F,[self.xi.value,0],t,
                    atol = 1e-11, # Default 1.49012e-8
                    rtol = 1e-11, # Default 1.49012e-8
                    )
        assert xv.shape == (len(t),2)
        # xv is array of calculated positions and velocities at times in t
        self.t2v = Spline(t,xv[:,1])
        return self.t2v
    def set_D(
            self,                   # GUN instance
            fraction=magic.D_frac   # Finite difference fraction
            ):
        '''Calculate dv/df in terms of spline coefficients and save as self.D.
              
        Note self.D.shape = (len(c_v)-4, len(c_f)-4) because I drop the
        last 4 which are always 0.
        '''
        # Spline.new_c(c) does not modify Spline.  It returns a copy
        # of Spline with the coefficients c and an updated provenance.
        eos_nom = self.eos
        c_f_nom = eos_nom.get_c()      # Nominal coefficients for eos
        n_f = len(c_f_nom) - magic.end # Dimension of optimization var
        t2v_nom = self.set_t2v()
        c_v_nom = t2v_nom.get_c()
        n_v = len(c_v_nom) - magic.end
        self.D = np.empty((n_v,n_f)) # To be dv/df matrix
        for i in range(n_f):
            c_f = c_f_nom.copy()
            # Next set size of finite difference for derivative approximation
            d_f = float(c_f[i]*fraction)
            c_f[i] += d_f
            # Next run a simulation to get a v(t) spline a for a modified eos
            self.eos = eos_nom.new_c(c_f)
            c_v = self.set_t2v().get_c()
            D_i = ((c_v - c_v_nom)[:-magic.end])/d_f
            self.D[:,i] = D_i
        self.eos = eos_nom
        self.t2v = t2v_nom
        return self.D
    def set_B_ep(
            self, # GUN instance
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
    def func(self, d, BD):
        '''The linearized objective function, S(d), in notes.tex.
        '''
        r = self.ep - np.dot(BD,d) # Residual
        return float(np.dot(r.T,r))
    def Gh(self):
        ''' Calculate and return constraint matrix G and vector h.  The
        constraint enforced by cvxopt.solvers.qp is
        
        Gx leq_component_wise h

        Want the following constraints:

        f'' positive for all x
        f' negative for x_max
        f positive for x_max
        
        '''
        original_eos = self.eos
        new_c = self.eos.get_c().copy()
        dim = len(new_c) - magic.end
        t_all = self.eos.get_t()
        t_unique = t_all[magic.end-1:1-magic.end]
        c = np.zeros(dim + magic.end)
        n_constraints = len(t_unique) + 2
        G = np.zeros((len(t_unique)+2, dim))
        for i in range(dim):
            c[i] = 1.0
            f = self.eos.new_c(c)
            G[:-2,i] = -f.derivative(2)(t_unique)
            G[-2,i] = f.derivative(1)(t_unique[-1])
            G[-1,i] = -f(t_unique[-1])
            c[i] = 0.0
        self.eos = original_eos
        h = -np.dot(G,new_c[:-magic.end])
        return G,h
    def Pq(self, BD, ep_f, ep_v):
        '''From the section "A Posteriori Probability" of notes.tex,

        P = Sigma_f^{-1} + (BD)^T Sigma^{-1}_v BD
        q^T = ep_f^T Sigma_f^{-1} - ep_v^T Sigma_v^{-1} BD
        R = ep_f^T Sigma_f^{-1} ep_f + ep_v^T Sigma_v^{-1} ep_v

        This method returns R and preconditioned P and q, namely:

        \tilde P = U^{-1} P U^{-1} (variable t_P here)
        \tilde q^T = q^T U^{-1} (variable t_q here)
        '''

        UIS = np.dot(self.U_inv, self.Sigma_f_inv) # U^{-1} Sigma_f^{-1}
        BDUI = np.dot(BD, self.U_inv)              # BD  U^{-1}
        t_P = np.dot(UIS,self.U_inv) + np.dot(BDUI.T/self.sigma_sq_v,BDUI)
        t_q = np.dot(ep_f, UIS.T)-np.dot(ep_v/self.sigma_sq_v, BDUI)
        return t_P,t_q
    def opt(
            self,        # GUN instance
            vt,          # Simulated experimental data
            ):
        ''' Does a constrained optimization step.

            minimize (1/2) x^TPx + q^T x
            ubject to G x \preceq h (\preceq is \leq for each component)
        '''
        # Takes ~ 20 seconds on watcher
        from cvxopt import matrix, solvers
        solvers.options['maxiters']=1000 # 100 default
        solvers.options['feastol']=1e-12
        solvers.options['show_progress']=True#False
        # default, 1e-7 allows non-monotonic, 1e-16 -> "singular KKT matrix"

        D = self.set_D() # Expensive
        B,ep_v = self.set_B_ep(vt)
        ep_f = (self.eos.get_c() - self.original_eos.get_c())[:-magic.end]
        BD = np.dot(B, D)
        P,q = self.Pq(BD, ep_f, ep_v) # P and q are preconditioned
        G,h = self.Gh()
        G = np.dot(G, self.U_inv)
        #sol=solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h) )
        sol=solvers.qp(matrix(P), matrix(q))  # Unconstrained optimization
        if sol['status'] != 'optimal':
            for key,value in sol.items():
                print(key, value)
            raise RuntimeError
        z = np.array(sol['x']).reshape(-1)
        d_hat = np.dot(self.U_inv, z)
        new_c = self.eos.get_c().copy()
        new_c[:-magic.end] += d_hat
        self.eos = self.eos.new_c(new_c)
        B,ep_v = self.set_B_ep(vt)
        ep_f = (new_c - self.original_eos.get_c())[:-magic.end]
        quad = lambda M, v: np.dot(v, np.dot(M,v)) # Quadratic
        cost = quad(1.0/self.sigma_sq_v, ep_v) + quad(self.Sigma_f_inv, ep_f)
        return cost, d_hat
    def free_opt(
            self, # GUN instance
            vt,   # Simulated experimental data
            rcond=1e-9,
            ):
        ''' Do an unconstrained optimization step.  Uses an SVD solver.
        ''' 
        from numpy.linalg import lstsq
        new_c = self.eos.get_c().copy()
        B,ep = self.set_B_ep(vt)
        BD = np.dot(B, self.set_D())
        d_hat = lstsq(BD, ep, rcond=rcond)[0]
        new_c[:-magic.end] += d_hat
        self.eos = self.eos.new_c(new_c)
        return -self.log_like(vt), d_hat
    def log_like(
            self, # GUN instance
            vt,   # Arrays of measured times and velocities
            ):
        ''' Assume t are accurate and that for model velocities m

            L(m) = log(p(v|m)) =\sum_t - \frac{(v-m)^2}{2\sigma^2}

        '''
        v,t = vt
        t2v = self.set_t2v()
        m = t2v(t)
        d = v-m
        return -np.dot(d/self.sigma_sq_v,d)/2
def plot_f_v_e(
        data, # Tuple with elements (mod,(x,y,name))
        t,   # "Experimental" times
        e,    # Tuple of time series of velocity errors
        fig   # plt.figure instance
        ):
    '''Plot nominal, perturbed and fit eoses and the consequent
    velocities.  Also plot the errors used for fitting.
    '''
    cm = r'$x/(\rm{cm})$'
    mu_sec = r'$t/(\mu\rm{sec})$'
    v_key = r'$v/(\rm{km/s})$'
    e_key = r'$e/(\rm{m/s})$'
    ax_d = {
        'f':{'ax':fig.add_subplot(3,1,1), 'l_x':cm,'loc':'upper right'},
        'v':{'ax':fig.add_subplot(3,1,2), 'l_x':cm,
            'l_y':v_key, 'loc':'lower right'},
        'e':{'ax':fig.add_subplot(3,1,3), 'l_x':mu_sec,
             'l_y':e_key, 'loc':'lower right'}
    }
    for mod,xyn in data.items():
        for x,y,name in xyn:
            ax_d[name]['ax'].plot(x,y,label=r'$\rm %s$'%mod)
    for i in range(len(e)):
        ax_d['e']['ax'].plot(t*1e6,e[i]/100,label='%d'%i)
    for name,d in ax_d.items():
        d['ax'].legend(loc=ax_d[name]['loc'])
        d['ax'].set_xlabel(d['l_x'])
        d['ax'].set_ylabel(r'$%s$'%name)
        if 'l_y' in d:
            d['ax'].set_ylabel(d['l_y'])
    return
def plot_dv_df(gun, x, fig):
    '''Plot analysis of 2 finite difference approximations to dv/df
    '''
    # fraction = 2.0e-2 about max for convex f(x)
    # fraction = 1.0e-3 about min for bugless T(x) integration
    frac_a = 2.0e-2
    frac_b = 2.0e-3
    DA = gun.set_D(fraction=frac_a)
    DB = gun.set_D(fraction=frac_b)
    x_D = (gun.t2v.get_t()[:-magic.end])*1.0e6

    eos_nom = gun.eos
    c = eos_nom.get_c()
    f = gun.eos(x)
    t = np.linspace(0,105.0e-6,500)
    gun.set_t2v()
    v = gun.t2v(t)
    def delta(frac):
        df = []
        dv = []
        for i in range(len(c)):
            c_ = c.copy()
            c_[i] = c[i]*(1+frac)
            gun.eos = eos_nom.new_c(c_)
            gun.set_t2v()
            df.append(gun.eos(x)-f)
            dv.append(gun.t2v(t) - v)
        return np.array(df),np.array(dv)
    dfa, dva = delta(frac_a)
    dfb, dvb = delta(frac_b)
    # Positions for add_subplot(4,3,n_)
    #  1  2  3
    #  4  5  6
    #  7  8  9
    mic_sec = r'$t/(\mu \rm{sec})$'
    for n_, x_, y_, l_x, l_y in (
            (1, x, np.array([f]), '$x$', '$f$'),
            (2, x, dfa, '$x$', '$\Delta f$'),
            (3, x, dfb, '$x$', '$\Delta f$'),
            (4, t, np.array([v]),mic_sec, r'$v/(\rm{km/s})$'),
            (5, t, dva,mic_sec, '$\Delta v$'),
            (6, t, dvb, mic_sec,'$\Delta v$'),
            (7, x_D, DA.T-DB.T, mic_sec, r'$\rm Difference$'),
            (8, x_D, DA.T, mic_sec, '$\Delta v/\Delta f$'),
            (9, x_D, DB.T, mic_sec, '$\Delta v/\Delta f$'),
            ):
        ax = fig.add_subplot(3,3,n_)
        ax.set_xlabel(l_x)
        ax.set_ylabel(l_y)
        n_y, n_x = y_.shape
        if n_ in (4,5,6):
            y_ = y_/magic.cm2km
            if n_ in (5,6):
                y_ *= 1e3
                ax.set_ylabel(r'$\Delta v/(\rm{m/s})$')
            x_ = x_*1.0e6
        for i in range(n_y):
            if n_ in (1,2,3):
                ax.loglog(x_, y_[i])
                ax.set_ylim(ymin=1e4, ymax=1e12)
            else:
                ax.plot(x_, y_[i])
    fig.subplots_adjust(wspace=0.3) # Make more space for label

def experiment(plot_data=None):
    '''Make "experimental" data from gun with eos f(x) =
    C/x^3 + 2 sin(freq*(x-x_0)) * e^{(x-x_0)^2/(2*w^2)} * C/(freq*(x-x_0)^3)
    
    Also put samples of that eos and the consequent velocity in the dict
    "plot_data" if it is provided.
    '''
    x_0 = 0.6   # Location of maximum of Gaussian envelope
    freq=.2     # Frequency of oscillation
    w = .2      # Width of Gaussian envelope
    gun = GUN(N=magic.n_x_pert)
    pert = lambda x : 2*np.sin(freq*(x-x_0))*np.exp(-(x-x_0)**2/(2*w**2))
    f = lambda x : gun.C.value/x**3 + pert(x) * gun.C.value/(freq*x_0**3)
    gun.set_eos(f)
    # Make simulated measurements
    t2v = gun.set_t2v()
    t = np.linspace(0,gun.t_max,magic.n_t_sim)
    v = t2v(t)
    if plot_data:
        # Select samples in x for perturbation
        x = np.logspace(
            np.log(gun.xi.value/magic.stretch),
            np.log(gun.xf.value*magic.stretch),
            magic.n_x_pert, base=np.e)
        # plot f(x) and v(x) for perturbed gun
        f_p = gun.eos(gun.x)
        v_p = gun.x_dot(gun.x)
        plot_data['experimental'] = (
            (gun.x, f_p, 'f'),(gun.x, v_p/magic.cm2km, 'v'))
    return (v, t)
def best_fit(vt, constrained=True):
    '''Create new gun, adjust spline description of eos to make simulated
    velocities match "experimental" data vt and return adjusted gun.
    '''
    gun = GUN(N=magic.fit_dim)
    # Start with spline approximation to nominal eos
    gun.set_eos_spline(gun.x, gun.eos(gun.x))
    eps = []
    costs = []
    gun.opt_init() # Creates prior and preconditioner
    for i in range(magic.like_iter):
        old_eos = gun.eos
        if constrained:
            cost, d_hat = gun.opt(vt)      # -log a posteriori probability
        else:
            cost, d_hat = gun.free_opt(vt) # -log likelihood
        eps.append(gun.ep)
        costs.append(cost)
        if i == 0:
            tol = abs(cost)*magic.converge
            print('initial cost={0}'.format(cost))
            last = cost
            continue
        Delta = last - cost # Should be positive
        print('cost[%2d]=%e, delta=%e'%(i,cost,Delta))
        if Delta <= tol:
            if Delta < 0:
                gun.eos = old_eos
                eps.pop()
            return gun,eps
        last=cost
    raise RuntimeError('best_fit failed to converge')
def main():
    ''' Diagnostic plots
    '''
    # Make nominal gun and get data for plotting
    nom = GUN()
    x = nom.x    # Positions for plots
    plot_data = {'nominal':((x, nom.eos(x), 'f'),(x, nom.x_dot(x)/1e5, 'v'))}
    
    # Get "experimental" data and data for plotting "true" eos
    v,t = experiment(plot_data) # side effect assigns plot_data['experimental']

    # Calculate best fit to experiment and get data for plotting
    fit,e = best_fit((v,t),constrained=True)
    plot_data['fit']=((x, fit.eos(x), 'f'),(x, fit.x_dot(x)/magic.cm2km, 'v'))

    fig_fve = plt.figure('fve',figsize=(8,10))
    plot_f_v_e(plot_data, t, e, fig_fve)

    # Plot study of derivatives
    fig_d = plt.figure('derivatives', figsize=(14,16))
    study = GUN(N=10)
    study.set_eos_spline(study.x, study.eos(study.x))
    plot_dv_df(study, x, fig_d)

    if True:
        fig_d.savefig('fig_d.pdf', format='pdf')
        fig_fve.savefig('fig_fve.pdf', format='pdf')
    else:
        plt.show()
    return

# Test functions
import numpy.testing as nt
def test_spline():
    x_s = np.linspace(0,2*np.pi,20)
    y = np.sin(x_s)
    f = Spline(x_s,y)
    assert f.provenance.line == u'f = Spline(x_s,y)'
    nt.assert_array_equal(f.get_c()[-4:], np.zeros(4))        
    nt.assert_allclose(y, f(x_s), atol=1e-15)

def test_gun_splines(gun):
    '''test spline approximations of f(x) and v(t).  Side effects: Change eos
    and t2v to splines.
    '''
    # Get some values for C/x^3 eos
    x = gun.x
    y = gun.eos(x)
    v_a = gun.x_dot(x)
    t2v_a = gun.set_t2v()
    
    ts = t2v_a.get_t()
    v = t2v_a(ts)
    for i in range(len(ts)): # Test v(t) = 0 for t<0
        assert ts[i] > 0 or abs(v[i]) < 1e-13
    
    # Exercise/test GUN.set_eos_spline()
    gun.set_eos_spline(x,y)

    # Test closeness of spline to C/x^3 for f(x), v(x) and v(t)
    nt.assert_allclose(y, gun.eos(x))
    nt.assert_allclose(v_a, gun.x_dot(x))
    nt.assert_allclose(v, gun.set_t2v()(ts), atol=1e-11)
    return 0
def test_set_D(gun, plot_file=None):
    ''' Side effects:
    Sets gun.D
    
    '''
    n_f = len(gun.eos.get_c()) - magic.end
    n_v = len(gun.t2v.get_c()) - magic.end
    D = gun.set_D(fraction=1.0e-2) # 11.6 user seconds
    assert D.shape == (n_v, n_f)
    if plot_file == None:
        return 0
    fig = plt.figure('D', figsize=(7,5))
    ax = fig.add_subplot(1,1,1)
    for j in range(n_f):
        ax.plot(D[:,j])
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\left( \frac{\partial c_v[k]}{\partial c_f[i]} \right)$')
    fig.savefig(plot_file, format='pdf')
    return 0
def test_set_B_ep(gun, v_exp, t_exp, plot_file=None):
    n_v = len(gun.t2v.get_c()) - magic.end
    gun.set_B_ep((v_exp,t_exp))
    assert len(gun.ep) == len(t_exp)
    if plot_file == None:
        return 0
    t2v = gun.set_t2v()
    ts = t2v.get_t()
    v = t2v(ts)
    fmt = 'B.shape={0} != ({1},{2}) = (len(t_exp), n_f)'.format
    assert gun.B.shape == (len(t_exp), n_v),fmt(gun.B.shape, len(t_exp), n_v)
    fig = plt.figure('v,t')
    ax = fig.add_subplot(1,1,1)
    ax.plot(ts*1e6, v/1e5, label='simulation')
    ax.plot(t_exp*1e6, v_exp/1e5, label='experiment')
    ax.plot(t_exp*1e6, gun.ep/1e5, label=r'error $\epsilon$')
    ax.set_xlabel(r'$t/(\mu \rm{sec})$')
    ax.set_ylabel(r'$v/(\rm{km/s})$')
    ax.legend(loc='upper left')
    fig.savefig(plot_file,format='pdf')
    return 0
def plot_d_hat(d_hat, title='d_hat'):
    fig = plt.figure(title, figsize=(7,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(d_hat)
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\hat d[i]$')
    fig.savefig('d_hat_test.pdf',format='pdf')
    return 0
def test_func_etc(gun, eos_0, eos_1, d_hat, t_exp, v_exp, plot_files=None):

    ep_0 = gun.ep
    gun.set_eos(eos_0)
    D = gun.set_D() # Expensive
    vt = (v_exp, t_exp)
    B,ep_v = gun.set_B_ep(vt)
    BD = np.dot(B, D)
    d = np.zeros(len(d_hat))
    S_0 = gun.func(d, BD)        # Original cost function
    G_0,h = gun.Gh()
    ll_0 = gun.log_like((v_exp,t_exp))  # Original log likelihood
    f_0 = gun.eos(gun.x)                # Original eos values
    
    S_1 = gun.func(d_hat, BD)           # Updated cost function
    gun.set_eos(eos_1)
    G_1,h = gun.Gh() # FixMe: Should devise test of G_0 and G_1
    
    # Get epsilon for updated EOS
    gun.set_B_ep((v_exp,t_exp))
    ll_1 = gun.log_like((v_exp,t_exp))  # Updated log likelihood
    f_1 = gun.eos(gun.x)                # Updated eos values

    if plot_files == None:
        return 0
    
    print('''lstsq reduced func from {0:.3e} to {1:.3e}
 and the increase in log likelihood is {2:.3e} to {3:.3e}'''.format(
        S_0, S_1, ll_0, ll_1))
    if 'errors' in plot_files:
        # Plot orignal errors
        fig = plt.figure('errors')
        ax = fig.add_subplot(1,1,1)
        ax.plot(t_exp*1e6, ep_0, label='Original velocity error ep')
        # Plot reduced errors
        ax.plot(t_exp*1e6, gun.ep, label='New velocity error')
        ax.set_xlabel(r'$t/(\mu\rm{sec})$')
        ax.set_ylabel(r'$v/(\rm{x/s})$')
        ax.legend(loc='lower right')
        fig.savefig(plot_files['errors'],format='pdf')
    return 0
def test_opt():
    ''' Exercise GUN.opt()
    '''
    # Set up gun with spline eos
    gun = GUN()
    gun.set_eos_spline(gun.x,gun.eos(gun.x))
    gun.opt_init()
    old_eos = gun.eos
    # Make experimental data
    vt = experiment()
    # Calculate initial error
    gun.set_B_ep(vt)
    error_0 = gun.ep

    cost, d_hat = gun.opt(vt)

    # Calculate new error
    gun.set_B_ep(vt)
    error_1 = gun.ep
    #cost, d_hat = gun.free_opt((v_exp,t_exp), rcond=1e-6)
    fig = plt.figure('opt_result', figsize=(7,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(d_hat)
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\hat d[i]$')
    fig = plt.figure('opt_errors', figsize=(7,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(vt[1]*1e6,error_0,label='error_0')
    ax.plot(vt[1]*1e6,error_1,label='error_1')
    ax.set_xlabel(r'$t/(\mu\rm{sec})$')
    ax.legend()
    fig = plt.figure('eos', figsize=(7,6))
    ax = fig.add_subplot(1,1,1)
    x = gun.x
    ax.plot(x,old_eos(x),label='f_0')
    ax.plot(x,gun.eos(x),label='f_1')
    ax.legend()
    ax.set_xlabel(r'$x/{\rm cm}$')
    ax.set_ylabel(r'$f/{\rm dyn}$')
    #plt.show()
    fig.savefig('opt_result.pdf',format='pdf')  
    return 0 
   
def test():

    # test_spline()
    # Test __init__ _set_N, set_eos and E methods of GUN
    gun = GUN()
    s = '{0:.1f}'.format(gun.E(4))
    assert s  == '79200000000.0','gun.E(4)={0}'.format(s)

    test_gun_splines(gun) # Makes gun.eos and gun.t2v splines
    G,h = gun.Gh()
    eos_0 = gun.eos
    v_exp, t_exp = experiment()
    test_set_D(gun, 'D_test.pdf')
    test_set_B_ep(gun, v_exp, t_exp, 'vt_test.pdf')
    
    # Solve BD*d=epsilon for d without constraints
    cost, d_hat = gun.free_opt((v_exp, t_exp), rcond=1e-5)
    plot_d_hat(d_hat)
    eos_1 = gun.eos
    
    # Exercise func, d_func and constraint and make plots for d_hat, etc
    plot_files = {
        'errors':'errors_test.pdf',
        }
    test_func_etc(gun, eos_0, eos_1, d_hat, t_exp, v_exp, plot_files)
    
    #plt.show()
    return 0
def work():
    ''' This code for debugging stuff will change often
    '''
    v_exp, t_exp = experiment()
    gun = GUN()
    x = gun.x
    y = gun.eos(x)
    eos = gun.set_eos_spline(x,y)
    print('initial cost={0:.3e}'.format(-gun.log_like((v_exp,t_exp))))
    # Use numpy.linalg.lstsq
    for rcond in (1e-3, 1e-9):
        gun.eos = eos
        cost, d_hat = gun.free_opt((v_exp, t_exp), rcond=rcond)
        print('cost={0:.3e}, rcond={1:.2e}'.format(cost, rcond))
        plot_d_hat(d_hat, title='d_hat, rcond={0:.2e}'.format(rcond))
    # Use cvxopt.qp
    gun.eos = eos
    gun.opt_init()
    cost, d_hat = gun.opt((v_exp, t_exp))
    print('cost={0:.3e} for opt'.format(cost))
    plot_d_hat(d_hat, title='d_hat from cvxopt.qp(P,q)')
    
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
