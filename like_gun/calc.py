"""calc.py: Classes for EOS and gun simulation.  Derived from calc.py
in parent directory.  Goals are to develop, analyze and understand a
procedure for estimating an isentrope on the basis of data and simulations.

Run using python3 and scipy 0.14

I should figure out and document the constraints on the coefficients at
the repeated knots at the edges of splines.  Perhaps I should modify the
code to obey the constraints and operate with fewer degrees of freedom.

"""
import numpy as np
from scipy.integrate import quad, odeint
import scipy.interpolate # InterpolatedUnivariateSpline
#https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py

# Next line provides easy plotting, eg in pdb:
#> new_ax('eos').plot(x,f)
#> plt.show()
import matplotlib.pyplot as plt
new_ax = lambda n : plt.figure(n).add_subplot(1,1,1)

from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
from cmf_models import Component, Provenance
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
    def set_c(self,c):
        '''Return a new Spline instance that is copy of self except
        that the coefficients for the basis functions are c and
        provenance is updated.'''
        import copy
        from inspect import stack
        rv = copy.deepcopy(self)
        rv._eval_args = self._eval_args[0], c, self._eval_args[2]
        # stack()[1] is context that called set_c
        rv.provenance = Provenance(
            stack()[1], 'New coefficients', branches=[self.provenance],
            max_hist=10)
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
        from markup import oneliner
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
           )

from cmf_models import Float
from markup import oneliner
class GUN:
    '''Represents an imagined experiment and actual simulations.

    Section 2 of the document ../juq.tex describes the imagined
    experiment.    
    '''
    def __init__(
            self,             # GUN instance
            C=Float(2.56e10,'Constant in nominal equation of state',
                    max_hist=10),
            xi=Float(0.4,'Initial position of projectile / cm'),
            xf=Float(4.0,'Final/muzzle position of projectile /cm'),
            m=Float(100.0,'{0} of projectile / g'.format(oneliner.a(
                'Mass', href='http://en.wikipedia.org/wiki/Mass'))),
            N=400,            # Number of intervals between xi and xf
            sigma_sq_v=1.0e5, # Variance attributed to v measurements
            ):
        x = Float(1e10,'Just for exercising Float operations')
        C = C + x
        C = C - x
        C = C/x
        C = C*x
        C = -C
        C = -C
        self.C = C + 0.0 # Vacuous demonstration of Float + float
        self.xi = xi
        self.xf = xf
        self.m = m
        self._set_N(N)
        self.sigma_sq_v = sigma_sq_v
        self.components = set(('C','xi','xf','m'))#Component instance names
        return
    def _set_N(
            self,                 # GUN instance
            N,                    # Number of intervals between xi and xf
            stretch=magic.stretch # Extension of EOS beyond barrel
    ):
        '''Interval lengths are uniform on a log scale ie
               self.x[i+1]/self.x[i] = constant.
        '''
        # N points and N-1 intervals equal spacing on log scale
        self.x = np.logspace(
            np.log(self.xi.value/stretch),
            np.log(self.xf.value*stretch), N, base=np.e)
        self.set_eos(lambda x:self.C.value/x**3)
        return
    def set_eos(self, func):
        self.eos = func
    def set_eos_spline(self, x, f):
        s = '''Initialize the eos with a spline fit to the arrays x
and f with elements that are positions along the gun
barrel and the forces at those positons respectively.'''
        self.eos = Spline(x, f, s)
        self.components.add('eos')
        return
    def E(self, # GUN instance
          x     # Scalar position at which to calculate energy
          ):
        ''' Integrate eos between xi and x using numpy.integrate.quad
        to get energy of projectile at position x.
        '''
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
        '''Run a simulation to build a spline for mapping times to velocities
        '''
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
        xv = odeint(F,[self.xi.value,0],t, atol=1.0e-11, rtol=1.0e-11)
        assert xv.shape == (len(t),2)
        # xv is array of calculated positions and velocities at times in t
        self.t2v = Spline(t,xv[:,1])
        return self.t2v
    def set_D(self, fraction=magic.D_frac):
        '''Calculate dv/df in terms of spline coefficients and save as self.D
        '''
        eos_nom = self.eos
        c_f_nom = eos_nom.get_c() # Vector of nominal coefficients for eos
        n_f = len(c_f_nom)
        t2v_nom = self.set_t2v()
        c_v_nom = t2v_nom.get_c()
        n_v = len(c_v_nom)
        self.D = np.empty((n_v,n_f)) # To be dv/df matrix
        for i in range(n_f):
            c_f = c_f_nom.copy()
            # Next set size of finite difference for derivative approximation
            if c_f[i] == 0.0: # Funny end point knots
                d_f=fraction*float(c_f.max())*(self.xi.value/self.xf.value)**3
            else:
                d_f = float(c_f[i]*fraction)
            c_f[i] += d_f
            self.eos = eos_nom.set_c(c_f)
            # Next line runs a simulation to get a new spline for v(t)
            c_v = self.set_t2v().get_c()
            self.D[:,i] = (c_v - c_v_nom)/d_f
        self.eos = eos_nom
        self.t2v = t2v_nom
        return self.D
    def set_Be(self, vt):
        '''Map experimental velocities v and times t to the following:
        
        self.e[i]=v[i] - t2v(t[i]) Difference between simulation and data
        
        self.B[i,j] = b_j(t[i]) Where b_j is the jth basis function for
                                the t2v spline
        '''
        v,t = vt
        n_vt = len(v)
        assert len(t) == n_vt
        c_ = np.zeros(self.set_t2v().get_c().shape) # Also sets nominal t2v
        n_c = len(c_)
        self.B = np.zeros((n_vt,n_c))
        for j in range(n_c):
            c_[j] = 1.0
            delta_t2v = self.t2v.set_c(c_)
            self.B[:,j] = delta_t2v(t)
            c_[j] = 0.0
        self.e = v - self.t2v(t) # Calculate errors
    def set_ddd(self):
        '''Calculate the derivative (wrt to c) of the second derivative
        of the eos wrt x at the knots.  Store result as self.ddd and return it.
        self.ddd is passed to scipy.optimize.fmin_slsqp as the derivative
        of theconvexity constraint
        '''
        t_all = self.eos.get_t()
        c = np.zeros(t_all.shape)
        t = t_all[3:-3]  # Ignore edge knots that are at same position
        self.ddd = ddd = np.empty((len(t), len(c)))
        for i in range(len(c)):
            c[i] = 1.0
            ddd[:,i] = self.eos.set_c(c).derivative(2)(t)
            c[i] = 0.0
        return ddd
    def opt(self, vt):
        ''' Do a constrained optimization step
        ''' 
        from scipy.optimize import fmin_slsqp as fmin
        self.set_D() # Expensive
        self.set_Be(vt)
        self.BD = np.dot(self.B, self.D)
        self.set_ddd()
        def func(d, self):
            '''The objective function, S(d) in notes.tex.
            '''
            r = self.e - np.dot(self.BD,d) # Residual
            return float(np.dot(r.T,r))
        def d_func(d, self):
            '''Derivative of the objective function.
            '''
            r = self.e - np.dot(self.BD,d)
            return -2*np.dot(self.BD.T,r)
        def dd(d, self):
            '''Return the vector of constraint function values, ie, the second
            derivitive of f at the knots ([3:-3] ignores repeated edge knot
            locations).
            '''
            c = self.eos.get_c()
            t = self.eos.get_t()
            return self.eos.set_c(c+d).derivative(2)(t[3:-3])
        c = self.eos.get_c()
        d_hat, ss, its, lmode, smode = fmin(
            func,
            np.zeros(c.shape),
            f_ieqcons=dd,
            args=(self,),
            fprime=d_func,
            iter=2000,
            fprime_ieqcons=lambda d, self: self.ddd,
            full_output=True,
            iprint=0,
            )
        new_c = c + d_hat
        self.eos = self.eos.set_c(new_c)
        return
    def log_like(
            self, # GUN instance
            vt,   # Arrays of measured times and velocities
            ):
        ''' Assume t are accurate and that for model velocities m

            L(m) = log(p(v|m)) =\sum_t - \frac{(v-m)^2}{2\sigma^2}

            g = dL/dm = \frac{v-m}{\sigma^2}

            h = \frac{d^2 L(m + ag)}{d a^2} = -\frac{1}{\sigma^2}

        This problem is easy because H is -\frac{1}{\sigma^2}\cdot I
        In general:

            m_hat = m + H^{-1} g

        Return: L, g, h
        '''
        v,t = vt
        t2v = self.set_t2v()
        m = t2v(t)
        d = v-m
        g = d/self.sigma_sq_v
        L = -np.dot(d,g)/2
        h = -1.0/self.sigma_sq_v
        return L, g, h
def plot_f_v_e(data, vt, e, fig):
    '''Plot nominal, perturbed and fit eoses and the consequent
    velocities.  Also plot the errors used for fitting.
    '''
    v,t = vt
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
    x_D = gun.t2v.get_t()*1.0e6

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
            gun.eos = eos_nom.set_c(c_)
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
        if n_x == len(t):
            y_ = y_/magic.cm2km
            if y_.max() < 0.1:
                y_ *= 1e3
                ax.set_ylabel(r'$\Delta v/(\rm{m/s})$')
            x_ = x_*1.0e6
        for i in range(n_y):
            if l_y == '$\Delta f$' or l_y == '$f$':
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
def best_fit(vt):
    '''Create new gun, adjust spline description of eos to make simulated
    velocities match vt and return adjusted gun.
    '''
    gun = GUN(N=magic.fit_dim)
    # Start with spline approximation to nominal eos
    gun.set_eos_spline(gun.x, gun.eos(gun.x))
    last, g,h = gun.log_like(vt)
    print('L[-1]=%e'%(last,))
    e = []
    for i in range(magic.like_iter):
        old_eos = gun.eos
        gun.opt(vt)
        e.append(gun.e)
        L, g, h = gun.log_like(vt)
        Delta = L - last
        print('L[%2d]=%e, delta=%e'%(i,L,Delta))
        if Delta <= abs(L)*magic.converge:
            gun.eos = old_eos
            e.pop()
            return gun,e
        last=L
    raise RuntimeError('best_fit failed to converge')
def main():
    ''' Diagnostic plots
    '''
    # Make nominal gun and get data for plotting
    nom = GUN()
    x = nom.x    # Positions for plots
    plot_data = {'nominal':((x, nom.eos(x), 'f'),(x, nom.x_dot(x)/1e5, 'v'))}
    
    # Get "experimental" data and data for plotting "true" eos
    vt = experiment(plot_data)

    # Calculate best fit to experiment and get data for plotting
    fit,e = best_fit(vt)
    plot_data['fit']=((x, fit.eos(x), 'f'),(x, fit.x_dot(x)/magic.cm2km, 'v'))

    fig_fve = plt.figure('fve',figsize=(8,10))
    plot_f_v_e(plot_data, vt, e, fig_fve)

    # Plot study of derivatives
    fig_d = plt.figure('derivatives', figsize=(14,16))
    study = GUN(N=10)
    study.set_eos_spline(study.x, study.eos(study.x))
    plot_dv_df(study, x, fig_d)

    if False:
        fig_d.savefig('fig_d.pdf', format='pdf')
        fig_fve.savefig('fig_fve.pdf', format='pdf')
    else:
        plt.show()
    return
    
if __name__ == "__main__":
    main()

#---------------
# Local Variables:
# mode: python
# End:
