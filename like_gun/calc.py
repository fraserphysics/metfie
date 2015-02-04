"""calc.py: Classes for EOS and gun simulation.  Derived from calc.py
in parent directory.

Run using python3 and scipy 0.14

"""
import numpy as np
from scipy.integrate import quad, odeint
import matplotlib.pyplot as plt # For plots for debugging
new_ax = lambda n : plt.figure(n).add_subplot(1,1,1)
import scipy.interpolate # InterpolatedUnivariateSpline
#https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py
class spline(scipy.interpolate.InterpolatedUnivariateSpline):
    '''From the source:

    data = dfitpack.fpcurf0(x,y,k,w=w,xb=bbox[0],xe=bbox[1],s=s)
    n,t,c,k,ier = data[7],data[8],data[9],data[5],data[-1]
    self._eval_args = t[:n],c[:n],k
    
    t is the array of knots, c is the array of coefficients and k is the
    order.    
    '''
    def get_t(self):
        return self._eval_args[0]
    def get_c(self):
        return self._eval_args[1]
    def set_c(self,c):
        self._eval_args = self._eval_args[0], c, self._eval_args[2]
        return
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
    stretch=1.25,          # Expansion of x beyond (xi,xf)
    v_var=1.0e5,          # Variance attributed to v measurements
    k_factor=1.0e6,       # (f(xi)/f(xf))^2
    c_weight=1.0e8,
           )

class GUN:
    def __init__(self,      # GUN instance
                 C=2.56e10, # Constant in nominal equation of state
                 xi=0.4,    # Initial position of projectile / cm
                 xf=4.0,    # Final/muzzle position of projectile /cm
                 m=100.0,   # Mass of projectile / g
                 N=400,     # Number of intervals between xi and xf
                 sigma_sq_v = magic.v_var
                 ):
        self.C = C
        self.xi = xi
        self.xf = xf
        self.m = m
        self._set_N(N)
        self.sigma_sq_v = sigma_sq_v
        return
    def _set_N(
            self,                 # GUN instance
            N,                    # Number of intervals between xi and xf
            stretch=magic.stretch # Extension of EOS beyond barrel
    ):
        '''Interval lengths are uniform on a log scale ie constant ratio.  x_c
        are points in the centers of the intervals and x are the end
        points.

        '''
        # N points and N-1 intervals equal spacing on log scale
        self.x = np.logspace(
            np.log(self.xi/stretch), np.log(self.xf*stretch), N, base=np.e)
        self.set_eos(self.x, self.C/self.x**3)
        c = self.eos.get_c()
        c_sq = c*c
        k = c_sq.sum()/len(c_sq)
        self.sigma_f = (c_sq + k*magic.k_factor)/magic.c_weight
        self.mu_f = c
        return
    def set_eos(self,   # GUN instance
                x,
                f
                ):
        self.eos = spline(x, f)
        return
    def E(self, # GUN instance
          x     # Scalar position at which to calculate energy
          ):
        ''' Integrate eos between xi and x using numpy.integrate.quad
        to get energy of projectile at position.
        '''
        rv, err = quad(self.eos,self.xi, min(x,self.xf))
        if rv != rv: # test for nan
            raise RuntimeError
        return rv
    def x_dot(self, # GUN instance
              x     # Position[s] /cm at which to calculate velocity
              ):
        ''' Calculate the projectile velocity at position x
        '''
        if isinstance(x,np.ndarray):
            return np.array([self.x_dot(x_) for x_ in x])
        elif isinstance(x,np.float):
            if x <= self.xi:
                return 0.0
            assert(self.E(x)>=0.0),'E(%s)=%s'%(x,self.E(x))
            return np.sqrt(2*self.E(x)/self.m)
        else:
            raise RuntimeError('x has type %s'%(type(x),))
    def set_t2v(self):
        '''Build spline for mapping times to velocities
        '''
        m = self.m
        f = self.eos
        xf = self.xf
        def F(x,t):
            '''x = (position,velocity)
            '''
            if t<0:
                return np.zeros(2)
            if x[0] > xf:
                acceleration = 0.0
            else:
                acceleration = f(x[0])/m
            return np.array([x[1], acceleration])
        t = np.linspace(magic.t_min, magic.t_max, magic.n_t)
        self.t_max = t[-1]
        xv = odeint(F,[self.xi,0],t, atol=1.0e-11, rtol=1.0e-11)
        self.t2v = spline(t,xv[:,1])
        return self.t2v
    def set_D(self, fraction=magic.D_frac):
        '''Calculate dv/df in terms of spline coefficients and save as self.D
        '''
        c_f_nom = self.eos.get_c()
        n_f = len(c_f_nom)
        c_v_nom = self.set_t2v().get_c()
        n_v = len(c_v_nom)
        self.D = np.empty((n_v,n_f))
        for i in range(n_f):
            c_f = c_f_nom.copy()
            d_f = float(c_f[i]*fraction)
            if d_f == 0.0:
                d_f = fraction*float(c_f.max())*(self.xi/self.xf)**3
            c_f[i] += d_f
            self.eos.set_c(c_f)
            c_v = self.set_t2v().get_c() # *** expensive ***
            self.D[:,i] = (c_v - c_v_nom)/d_f
        self.eos.set_c(c_f_nom)
        self.t2v.set_c(c_v_nom)
        return self.D
    def set_Be(self, vt):
        '''Given experimental velocities and times, vt: calculate errors,
        self.e, and the matrix of basis functions for t2v applied to
        times, self.B.

        '''
        v,t = vt
        n_vt = len(v)
        assert len(t) == n_vt
        c_v = self.set_t2v().get_c() # Calculate nominal t2v function
        c_ = np.zeros(c_v.shape)
        n_c = len(c_v)
        self.B = np.zeros((n_vt,n_c))
        # b[i,j] = b_t2v[j](t[i]) jth function of ith point
        for j in range(n_c):
            c_[j] = 1.0
            self.t2v.set_c(c_)
            self.B[:,j] = self.t2v(t)
            c_[j] = 0.0
        self.t2v.set_c(c_v)      # Restore nominal t2v function
        self.e = v - self.t2v(t) # Calculate errors
    def set_ddd(self):
        '''Calculate the derivative (wrt to c) of the second derivative of eos
        wrt x at the knots t.  Store result as self.ddd and return it.
        '''
        s = self.eos
        c_old = s.get_c()
        c = np.zeros(c_old.shape)
        t_all = s.get_t()
        t = t_all[3:-3]
        self.ddd = ddd = np.empty((len(t), len(c)))
        for i in range(len(c)):
            c[i] = 1.0
            s.set_c(c)
            ddd[:,i] = s.derivative(2)(t)
            c[i] = 0.0
        s.set_c(c_old)
        return ddd
    def mse(self,vt):
        from numpy.linalg import lstsq
        self.set_D()
        self.set_Be(vt)
        BD = np.dot(self.B, self.D)
        b = np.dot(BD.T,self.e)
        a = np.dot(BD.T, BD)
        d_hat = lstsq(a,b)[0]
        c = self.eos.get_c()
        new_c = self.eos.get_c() + d_hat
        self.eos.set_c(new_c)
    def m_ap(self,vt):
        '''Maximum a posterior probability.  See eq:dmap in notes.tex.
        '''
        from numpy.linalg import lstsq
        self.set_D()
        self.set_Be(vt)
        BD = np.dot(self.B, self.D)
        c = self.eos.get_c()
        b = np.dot(BD.T,self.e)/self.sigma_sq_v + (self.mu_f-c)/self.sigma_f
        a = np.dot(BD.T, BD)/self.sigma_sq_v + np.diag(1/self.sigma_f)
        d_hat = lstsq(a,b)[0]
        new_c = c + d_hat
        self.eos.set_c(new_c)
        return
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
            r = self.e - np.dot(self.BD,d)
            return float(np.dot(r.T,r))
        def d_func(d, self):
            '''Derivative of the objective function.
            '''
            r = self.e - np.dot(self.BD,d)
            return -2*np.dot(self.BD.T,r)
        def dd(d, self):
            '''Return the vector of constraint function values, ie, the second
            derivitive of f at the knots.
            '''
            s = self.eos
            c = s.get_c()
            t = s.get_t()
            s.set_c(c+d)
            rv = s.derivative(2)(t[3:-3])
            s.set_c(c)
            return rv
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
        self.eos.set_c(new_c)
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
        sigma_sq = magic.v_var
        d = v-m
        g = d/sigma_sq
        L = -np.dot(d,g)/2
        h = -1.0/sigma_sq
        return L, g, h
def plot_f_v_e(gun, data, t, e, fig):
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
    x_D = gun.t2v.get_t()*1.0e6

    c = gun.eos.get_c()
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
            gun.eos.set_c(c_)
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
        
def main():
    ''' Diagnostic plots
    '''
    import matplotlib.pyplot as plt
    
    # Unperturbed gun
    gun = GUN()
    f_nom = gun.eos(gun.x)
    v_nom = gun.x_dot(gun.x)

    # Plot study of derivatives
    fig_d = plt.figure('derivatives', figsize=(14,16))
    plot_dv_df(GUN(N=10), gun.x, fig_d)

    plot_data = {'initial':((gun.x, f_nom, 'f'),(gun.x, v_nom/1e5, 'v'))}
    
    # Select samples in x for perturbation
    x = np.logspace(
        np.log(gun.xi/magic.stretch),
        np.log(gun.xf*magic.stretch),
        magic.n_x_pert, base=np.e)
    
    # Create perturbed gun
    f = gun.eos(x)
    x_off = 0.6
    y = x-x_off
    freq=.2
    w = .2
    D = 2*np.sin(freq*y)*np.exp(-y**2/(2*w**2))*gun.eos(x_off)/freq
    gun_p = GUN(N=magic.n_x_pert)
    gun_p.set_eos(x,f+D)
    # plot f(x) and v(x) for perturbed gun
    f_p = gun_p.eos(gun_p.x)
    v_p = gun_p.x_dot(gun_p.x)
    plot_data['actual'] = (
        (gun_p.x, f_p, 'f'),(gun_p.x, v_p/magic.cm2km, 'v'))
    # Make simulated measurements
    t2v = gun_p.set_t2v()
    t = np.linspace(0,gun_p.t_max,magic.n_t_sim)
    v = t2v(t)
    vt= (v, t)
    # Set gun for fitting and derivative
    gun_fit = GUN(N=magic.fit_dim)
    last, g,h = gun_fit.log_like(vt)
    print('L[-1]=%e'%(last,))
    e = []
    for i in range(magic.like_iter):
        old_c = gun_fit.eos.get_c()
        #gun_fit.mse(vt)
        #gun_fit.m_ap(vt)
        gun_fit.opt(vt)
        e.append(gun_fit.e)
        L, g, h = gun_fit.log_like(vt)
        Delta = L - last
        print('L[%d]=%e, delta=%e'%(i,L,Delta))
        if Delta <= abs(L)*magic.converge:
            gun_fit.eos.set_c(old_c)
            e.pop()
            break
        last=L
    f_hat = gun_fit.eos(gun.x)
    v_hat = gun_fit.x_dot(gun.x)
    plot_data['fit'] = (
        (gun.x, f_hat, 'f'),
        (gun.x, v_hat/magic.cm2km, 'v'))
    # Plot fit, nominal, perturbed and fit and dL
    L, g, h = gun.log_like(vt)
    fig_fve = plt.figure('fve',figsize=(8,10))
    plot_f_v_e(gun, plot_data, t, e, fig_fve)

    if True:
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
