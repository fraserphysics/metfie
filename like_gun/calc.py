"""calc.py: Classes for EOS and gun simulation.  Derived from calc.py
in parent directory.

Run using python3.

"""
import numpy as np
from scipy.integrate import quad, odeint
import matplotlib.pyplot as plt # For plots for debugging
new_ax = lambda n : plt.figure(n).add_subplot(1,1,1)
import scipy.interpolate #.InterpolatedUnivariateSpline
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
class GUN(object):
    def __init__(self,      # GUN instance
                 C=2.56e10, # Constant in nominal equation of state
                 xi=0.4,    # Initial position of projectile / cm
                 xf=4.0,    # Final/muzzle position of projectile /cm
                 m=100.0,   # Mass of projectile / g
                 N=400,     # Number of intervals between xi and xf
                 sigma_sq_v = 1.0e6
                 ):
        self.C = C
        self.xi = xi
        self.xf = xf
        self.m = m
        self._set_N(N)
        self.sigma_sq_v = sigma_sq_v
        return
    def _set_N(
            self,         # GUN instance
            N,            # Number of intervals between xi and xf
            stretch = 1.1 # Extension of EOS beyond barrel
    ):
        '''Interval lengths are uniform on a log scale ie constant ratio.  x_c
        are points in the centers of the intervals and x are the end
        points.

        '''
        # N points and N-1 intervals equal spacing on log scale
        log_x = np.linspace(np.log(self.xi/stretch),np.log(self.xf*stretch),N)
        self.x = np.exp(log_x)
        self.eos = spline(self.x, self.C/self.x**3)
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
    def T(self,         # GUN instance
          x,            # Array of positions /cm at which to calculate time
          fudge=1.0e-4  # avoid 1/0 in dT_dx
          ):
        ''' Calculate projectile time as a function of position
        '''
        x_d = self.xi + fudge     # Divide x into parts at x_d
        w = np.where(x < x_d)[0]
        d = w[-1]
        x_a = np.maximum(0.0, x[:d+2]-self.xi)
        x_b = x[d+1:]
        a = self.eos((self.xi+x_d)/2)/self.m #acceleration
        t_a = np.sqrt(2*x_a/a)               # invert x = (1/2)*a*t**2
        # Two arguments for dT_dx because odeint calls func with two arguments
        dT_dx = lambda t,x:np.sqrt((self.m/(2*self.E(x))))
        t_b = odeint(dT_dx,   # RHS of ODE
                     t_a[-1], # Initial value of time
                     x_b,     # Solve for time at each position in x
            )
        rv= np.concatenate((t_a[:-1],t_b.flatten()))
        return rv
    def set_eos(self,   # GUN instance
                x,
                f
                ):
        self.eos = spline(x, f)
        return
    def set_t2v(self):
        '''Build spline for mapping times to velocities
        '''
        log_x = np.linspace(np.log(self.xi),np.log(self.xf),500)
        x = np.exp(log_x)
        # Calculate t and v for x
        t = self.T(x) # *** expensive ***
        self.t_max = t[-1]
        v = self.x_dot(x)
        v_max = v[-1]
        v_spline = spline(t,v)
        def v_func(t):
            if t <= 0.0:
                return 0.0
            if t >= self.t_max:
                return v_max
            return v_spline(t)
        # Extend range of t by frac
        frac = .1
        t_frac = self.t_max*frac
        t = np.linspace(-t_frac, self.t_max+t_frac, 250)
        v = np.array([v_func(t_) for t_ in t])
        self.t2v = spline(t,v)
        return self.t2v
    def set_D(self, fraction=2.0e-2):
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
    def mse(self,vt,label):
        from numpy.linalg import lstsq # solve
        self.set_Be(vt)
        self.set_D()
        BD = np.dot(self.B, self.D)
        b = np.dot(BD.T,self.e)
        a = np.dot(BD.T, BD)
        d_hat = lstsq(a,b)[0]
        c = self.eos.get_c()
        new_c = self.eos.get_c() + d_hat
        self.eos.set_c(new_c)
        print('B.shape=%s, D.shape=%s, e.shape=%s,'%(
            self.B.shape, self.D.shape,self.e.shape))
        v,t = vt
        ax = new_ax('e')
        ax.plot(t,self.e,label=label)
        ax.legend()
    def m_ap(self,vt):
        '''Maximum a posterior probability.  See eq:dmap in notes.tex.
        '''
        from numpy.linalg import lstsq # solve
        self.set_Be(vt)
        self.set_D()
        c = self.eos.get_c()
        c_sq = c*c
        sigma_c = (c_sq + c_sq.sum()/(1.0e8*len(c_sq)))/1.0e-8
        BD = np.dot(self.B, self.D)
        b = np.dot(BD.T,self.e)/self.sigma_sq_v
        a = np.dot(BD.T, BD)/self.sigma_sq_v + np.diag(1/sigma_c)
        d_hat = lstsq(a,b)[0]
        new_c = c + d_hat
        self.eos.set_c(new_c)
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
        sigma_sq = 1.0e5
        t_ = self.T(self.x)
        i = np.where(t_ == 0.0)[0][-1]
        x_t = spline(t_[i:],self.x[i:])
        x = x_t(t)
        m = self.x_dot(x)
        d = v-m
        g = d/sigma_sq
        L = -np.dot(d,g)
        h = -1.0/sigma_sq
        return L, g, h
def plot_f_v_dL(gun, data, t, dL, fig):
    '''Plot nominal, perturbed and fit eoses and the consequent
    velocities.  Also plot the gradient of the log likelihood used for
    fitting.

    '''
    ax_d = {
        'f':{'ax':fig.add_subplot(3,1,1), 'l_x':r'$x$','loc':'upper right'},
        'v':{'ax':fig.add_subplot(3,1,2),'l_x':r'$x$','loc':'lower right'},
        'dL':{'ax':fig.add_subplot(3,1,3),'l_x':r'$t$'}
    }
    for name,d in ax_d.items():
        d['ax'].set_ylabel(r'$%s$'%name)
        d['ax'].set_xlabel(d['l_x'])
    for mod,xyn in data.items():
        for x,y,name in xyn:
            ax_d[name]['ax'].plot(x,y,label=r'$\rm %s$'%mod)
    for name in ('f','v'):
        ax_d[name]['ax'].legend(loc=ax_d[name]['loc'])
    ax_d['dL']['ax'].plot(t*1e6,dL)
    return
def plot_dv_df(gun, x, DA, DB, fig):
    '''Plot analysis of 2 finite difference approximations to dv/df
    '''
    f_i_A, v_i_A, f_all_nom_A, v_x_nom_A, Dv_Df_A = DA
    f_i_B, v_i_B, f_all_nom_B, v_x_nom_B, Dv_Df_B = DB

    n_i, n_x = f_i_B.shape
    # Positions for add_subplot(4,3,n_)
    #  1  2  3
    #  4  5  6
    #  7  8  9
    # 10 11 12
    for n_,x_,y_,l_ in (
            (1,gun.x,f_i_A,'$f$'),
            (2,gun.x,f_i_A-f_all_nom_A,'$Df$'),
            (3,gun.x,f_i_B-f_all_nom_B,'$Df$'),
            (4,x,v_i_A,'$v$'),
            (5,x,v_i_A-v_x_nom_A,'$Dv$'),
            (6,x,v_i_B-v_x_nom_B,'$Dv$'),
            (7,x,Dv_Df_A.T-Dv_Df_B.T,r'$\rm Difference$'),
            (8,x,Dv_Df_A.T,'$Dv/Df$'),
            (9,x,Dv_Df_B.T,'$Dv/Dc$'),
            (11,x,v_i_A*v_i_A-v_x_nom_A*v_x_nom_A,'$DE$'),
            (12,x,v_i_B*v_i_B-v_x_nom_B*v_x_nom_B,'$DE$'),
            ):
        ax = fig.add_subplot(4,3,n_)
        ax.set_ylabel(l_)
        ax.set_xlabel('$x$')
        for i in range(n_i):
            ax.plot(gun.x, y_[i])
        
def main():
    ''' Diagnostic plots
    '''
    import matplotlib.pyplot as plt
    
    # Unperturbed gun
    gun = GUN()
    f_nom = gun.eos(gun.x)
    v_nom = gun.x_dot(gun.x)

    plot_data = {'nominal':((gun.x, f_nom, 'f'),(gun.x, v_nom/1e5, 'v'))}
    
    # Select samples in x for perturbation
    n = 1000
    stretch = 1.1
    log_x = np.linspace(np.log(gun.xi/stretch),np.log(gun.xf*stretch),n)
    x = np.exp(log_x)
    
    # Create perturbed gun
    f = gun.eos(x)
    x_off = 0.6
    y = x-x_off
    freq=.2
    w = .2
    D = np.sin(freq*y)*np.exp(-y**2/(2*w**2))*gun.eos(x_off)/freq
    gun_p = GUN(N=1000)
    gun_p.set_eos(x,f+D)
    # plot f(x) and v(x) for perturbed gun
    f_p = gun_p.eos(gun_p.x)
    v_p = gun_p.x_dot(gun_p.x)
    t_p = gun_p.T(gun_p.x)
    plot_data['perturbed'] = ((gun_p.x, f_p, 'f'),(gun_p.x, v_p/1e5, 'v'))

    # Make simulated measurements
    t2v = gun_p.set_t2v()
    t = np.linspace(0,gun_p.t_max*1.05,1000)
    v = t2v(t)
    vt= (v, t)
    # Set gun for fitting and derivative
    gun_fit = GUN(N=12)
    for i in range(4):
        #gun_fit.m_ap(vt)
        gun_fit.mse(vt,'%d'%i)
    f_hat = gun_fit.eos(gun.x)
    v_hat = gun_fit.x_dot(gun.x)
    plot_data['fit'] = ((gun.x, f_hat, 'f'),(gun.x, v_hat/1e5, 'v'))
    
    # Plot fit, nominal, perturbed and fit and dL
    L, g, h = gun.log_like(vt)
    fig1 = plt.figure(figsize=(8,10))
    plot_f_v_dL(gun, plot_data, t, g, fig1)

    plt.show()
    return
    # Calculate derivatives
    # fraction = 2.0e-2 about max for convex f(x)
    # fraction = 1.0e-3 about min for bugless T(x) integration
    DA = gun.dv_df(gun.x, x, 2.0e-2)
    dv_df = DA[-1]
    DB = gun.dv_df(gun.x, x, 1.0e-2)
    fig2 = plt.figure(figsize=(14,16))
    plot_dv_df(gun, x, DA, DB, fig2)

    plt.show()
    #fig.savefig('fig.pdf', format='pdf')
    
if __name__ == "__main__":
    main()
    #_test()

#---------------
# Local Variables:
# mode: python
# End:
