"""calc.py: Classes for EOS and gun simulation.  Derived from calc.py
in parent directory.

Run using python3.

"""
import numpy as np
from scipy.integrate import quad, odeint
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
                 ):
        self.C = C
        self.xi = xi
        self.xf = xf
        self.m = m
        self._set_N(N)
        return
    def _set_N(self,  # GUN instance
               N,     # Number of intervals between xi and xf
               ):
        '''Interval lengths are uniform on a log scale ie constant ratio.  x_c
        are points in the centers of the intervals and x are the end
        points.

        '''
        # N points and N-1 intervals equal spacing on log scale
        stretch = 1.1
        log_x = np.linspace(np.log(self.xi/stretch),np.log(self.xf*stretch))
        self.x = np.exp(log_x)
        self.eos = spline(self.x, self.C/self.x**3)
        # Fudge to avoid 1/0 in self.dT_dx
        self._xi_ = self.xi + (self.x[1] - self.x[0])/10.0
        def dT_dx(t, x):
            ''' a service function called by odeint to calculate
            muzzle time T.  Since odeint calls dT_dx with two
            arguments, dT_dx has the unused t argument.
            '''
            if x < self._xi_:
                x = self._xi_
            return np.sqrt((self.m/(2*self.E(x))))
        self.dT_dx = dT_dx
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
    def set_eos(self,   # GUN instance
                x,
                f
                ):
        self.eos = spline(x, f)
        return
    def set_t2v(self):
        '''Build spline for mapping times to velocities
        '''
        # Calculate t and v for range of self.x
        t = self.T(self.x) # *** expensive ***
        t_max = t[-1]
        v = self.x_dot(self.x)
        v_max = v[-1]
        # Extend range of t by frac
        frac = .1
        n_frac = int(len(t)*frac)
        t_frac = t_max*frac
        t_pre = np.linspace(-t_frac, 0.0, n_frac,endpoint=False)
        t_post = np.linspace(t_max, t_frac+t_max, n_frac)[1:]
        v_pre = np.zeros(t_pre.shape)
        v_post = np.ones(t_post.shape)*v_max
        t_ = np.concatenate((t_pre, t, t_post))
        v_ = np.concatenate((v_pre, v, v_post))
        self.t2v = spline(t_,v_)
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
        self.e, and the matrix of basis functions applied to times, self.B.
        '''
        v,t = vt
        n_vt = len(v)
        assert len(t) == n_vt
        c_v = self.set_t2v().get_c() # Calculate nominal t2v function
        c_ = c_v * 0.0
        n_c = len(c_v)
        self.B = np.empty((n_vt,n_c)) # b[i,j] = b_t2v[j](t[i])
        for j in range(n_c):
            c_[j] = c_v[j]
            self.t2v.set_c(c_)
            self.B[:,j] = self.t2v(t)
            c_[j] = 0.0
        self.t2v.set_c(c_v)      # Restore nominal t2v function
        self.e = v - self.t2v(t) # Calculate errors
    def _solve_(self, a, b):
        from numpy.linalg import lstsq # solve
        d_hat = lstsq(a,b)[0]
        new_c = self.eos.get_c() + d_hat
        self.eos.set_c(new_c)
    def mse(self,vt):
        self.set_Be(vt)
        self.set_D()
        BD = np.dot(self.B, self.D)
        b = np.dot(BD.T,self.e)
        a = np.dot(BD.T, BD)
        self._solve_(a,b)
    def map(self,vt):
        ''' Need calculations of a and b appropriate to eq:hat in notes.tex
        '''
        from numpy.linalg import lstsq # solve
        self.set_Be(vt)
        self.set_D()
        BD = np.dot(self.B, self.D)
        BDTe = np.dot(BD.T,self.e)
        BDTBD = np.dot(BD.T, BD)
        d_hat = lstsq(BDTBD,BDTe)[0]
        new_c = self.eos.get_c() + d_hat
        self.eos.set_c(new_c)
    def log_like(
            self, # GUN instance
            tv,   # Arrays of measured times and velocities
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
        t,v = tv
        sigma_sq = 1.0e5
        t_ = self.T(self.x)
        x_t = spline(t_,self.x)
        x = x_t(t)
        m = self.x_dot(x)
        d = v-m
        g = d/sigma_sq
        L = -np.dot(d,g)
        h = -1.0/sigma_sq
        return L, g, h
    def max_like(self, tv, dv_df):
        '''Adjust self.eos to maximize the likelihood of tv with the
        assumption that dv_df is right.
        '''
        from numpy.linalg import lstsq, norm
        L_old, g, h = self.log_like(tv)
        c_old = self.eos.get_c()
        for i in range(5):
            print('%2d L=%f, norm(g)=%e, h=%e'%(i, L_old, norm(g), h))
            c = lstsq(dv_df, -g/h)[0] + c_old
            self.eos.set_c(c)
            L, g, h = self.log_like(tv)
            if L < L_old:
                print('Quit with L=%f, norm(g)=%e, h=%e\n'%(L, norm(g), h))
                self.eos.set_c(c_old)
                return
            L_old = L
            c_old = c
    def dv_df(
            self, # GUN instance
            x_v,  # Positions of velocity measuerments
            x_f,  # Positions of eos specifications
            fraction=1.0e-2
            ):
        '''Derivative of velocity at points x wrt. self.f at points x_f.
        Returns a len(x_v) \times len(x_f) matrix.
        
        '''
        eos_orig = self.eos
        f_x_nom = np.array([self.eos(x) for x in x_f])
        self.set_eos(x_f, f_x_nom)
        f_all_nom = self.eos(self.x) # Use spline as reference
        v_x_nom = self.x_dot(x_v)
        f = self.eos
        f_nom_c = f.get_c()
        f_nom_t = f.get_t()
        # f_nom_t is 4 longer than x_f but matches x_f[2:-2] in the middle
        rv = np.zeros((len(x_v), len(f_nom_c)))
        
        # For debug plots: f_i, df_i, v_i
        f_i = np.empty((len(f_nom_c),len(self.x)))
        v_i = np.empty((len(f_nom_c),len(x_v)))

        # Should pick pertubations that maintain convexity
        D_f = np.empty(f_nom_c.shape)
        for i in range(len(f_nom_c)):
            f_c = f_nom_c.copy()
            D_f[i] = f_c[i]*fraction
            if D_f[i] == 0.0:
                D_f[i] = D_f[i-1]
            f_c[i] += D_f[i]
            f.set_c(f_c)
            v = self.x_dot(x_v)
            Dv_Df = (v - v_x_nom)/D_f[i]
            rv[:,i] = Dv_Df
            # Debug data
            f_i[i,:] = f(self.x)
            v_i[i,:] = v
        self.eos = eos_orig
        return f_i, v_i, f_all_nom, v_x_nom, rv
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
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
    gun_p = GUN()
    gun_p.set_eos(x,f+D)
    # Calculate f(x), v(t) and c for perturbed gun
    f_p = gun_p.eos(gun.x)
    v_p = gun_p.x_dot(gun.x)
    t_p = gun_p.T(gun.x)
    plot_data['perturbed'] = ((gun.x, f_p, 'f'),(gun.x, v_p/1e5, 'v'))

    # Select samples in x for fitting and derivative
    n = 12
    log_x = np.linspace(np.log(gun.xi/stretch),np.log(gun.xf*stretch),n)
    x = np.exp(log_x)
    f = gun.eos(x)
    gun_fit = GUN()
    gun_fit.set_eos(x,f)

    # Estimate eos based on measured v
    tv = (t_p, v_p)
    gun_fit.mse(tv)
    f_hat = gun_fit.eos(gun.x)
    v_hat = gun_fit.x_dot(gun.x)
    plot_data['fit'] = ((gun.x, f_hat, 'f'),(gun.x, v_hat/1e5, 'v'))
    
    # Plot fit, nominal, perturbed and fit and dL
    L, g, h = gun.log_like(tv)
    fig1 = plt.figure(figsize=(8,10))
    plot_f_v_dL(gun, plot_data, t_p, g, fig1)

    plt.show()
    return
    # Calculate derivatives
    # fraction = 2.0e-2 about max for convex f(x)
    # fraction = 1.0e-3 about min for bugless v(x) integration
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
