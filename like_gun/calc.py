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
        # N+1 points and N intervals equal spacing on log scale
        log_x = np.linspace(np.log(self.xi),np.log(self.xf),N+1)
        step = (log_x[-1] - log_x[0])/N
        assert len(log_x) == N+1
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
    def E(self, # GUN instance
          x     # Scalar position at which to calculate energy
          ):
        ''' Integrate eos between xi and x using numpy.integrate.quad
        to get energy of projectile at position.
        '''
        rv, err = quad(self.eos,self.xi,x)
        return rv
    def x_dot(self, # GUN instance
              x     # Position[s] /cm at which to calculate velocity
              ):
        ''' Calculate the projectile velocity at position x
        '''
        if isinstance(x,np.ndarray):
            return np.array([self.x_dot(x_) for x_ in x])
        elif isinstance(x,np.float): 
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
    def log_like(
            self, # GUN instance
            t,    # Array of measured times
            v     # Array of measured velocities
            ):
        ''' Assume t are accurate and that for model velocities m
            log(p(v|m)) =\sum_t -((v-m)/1e5)^2
        '''
        t_ = self.T(self.x)
        x_t = spline(t_,self.x)
        x = x_t(t)
        m = self.x_dot(x)
        d = (v-m)/1e5
        return -np.dot(d,d),-2*d
    def dv_df(
            self, # GUN instance
            x_v,  # Positions of velocity measuerments
            x_f,  # Positions of eos specifications
            fraction=1.0e-2
            ):
        '''Derivative of velocity at points x wrt. self.f at points x_f.
        Returns a len(x_v) \times len(x_f) matrix.
        
        '''
        f_all_nom = self.eos(self.x)
        f_x_nom = np.array([self.eos(x) for x in x_f])
        v_x_nom = self.x_dot(x_v)
        self.set_eos(x_f, f_x_nom)
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
        return f_i, v_i, f_all_nom, v_x_nom
        
def plot():
    '''Plot velocity as a function of time.
    '''
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$v$')
    ax2 = fig.add_subplot(3,1,2)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f$')
    ax3 = fig.add_subplot(3,1,3)
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('$dL$')
    
    # Plot v(t) and f(x) for unperturbed gun
    gun = GUN()
    t = gun.T(gun.x)
    v = gun.x_dot(gun.x)
    ax1.plot(t*1e6, # microseconds
             v/1e5,# km/s
             label=r'$v_{\rm nominal}$')
    ax2.plot(gun.x, [gun.eos(x) for x in gun.x],label=r'$f_{\rm nominal}$')

    # Create perturbed gun
    n = 30
    log_x = np.linspace(np.log(gun.xi),np.log(gun.xf),n)
    x = np.exp(log_x)
    
    f_i, v_i, f_all_nom, v_x_nom = gun.dv_df(gun.x, x, 2.0e-2)
    n_i, n_x = f_i.shape
    fig2 = plt.figure()
    for n_,x_,y_,l_ in (
            (1,gun.x,f_i,'$f$'),
            (2,gun.x,f_i-f_all_nom,'$Df$'),
            (3,x,v_i,'$v$'),
            (4,x,v_i-v_x_nom,'$Dv$'),
            ):
        ax = fig2.add_subplot(2,2,n_)
        ax.set_ylabel(l_)
        ax.set_xlabel('$x$')
        for i in range(n_i):
            ax.plot(gun.x, y_[i])
        
    f = gun.eos(x)
    x_off = 0.6
    y = x-x_off
    freq=.2
    w = .2
    D = np.sin(freq*y)*np.exp(-y**2/(2*w**2))*gun.eos(x_off)/freq
    gun_p = GUN()
    gun_p.set_eos(x,f+D)
    
    # Plot v(t) and f(x) for perturbed gun
    t = gun.T(x)
    v = np.array([gun_p.x_dot(x_) for x_ in x])
    L,dL = gun.log_like(t,v)
    print('lengths: gun.x=%d, t=%d, v=%d, dL=%d'%(len(gun.x), len(t), len(v), len(dL)))
    ax1.plot(t*1e6, # microseconds
             v/1e5,# km/s
             label=r'$v_{\rm perturbed}$')
    ax2.plot(
        gun.x,
        [gun_p.eos(x) for x in gun_p.x],
        label=r'$f_{\rm perturbed}$')
    ax3.plot(t*1e6, dL)
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')

    
    plt.show()
    #fig.savefig('fig.pdf', format='pdf')
    
if __name__ == "__main__":
    plot()
    #_test()

#---------------
# Local Variables:
# mode: python
# End:
