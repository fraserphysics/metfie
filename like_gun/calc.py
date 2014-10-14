"""calc.py: Classes for EOS and gun simulation.  Derived from calc.py
in parent directory.

"""
import numpy as np
from scipy.integrate import quad, odeint
import numpy.linalg.linalg as LA, scipy.interpolate
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
def plot():
    '''Plot velocity as a function of time.
    '''
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$v$')
    ax2 = fig.add_subplot(2,1,2)
    gun = GUN()
    t = gun.T(gun.x)*1e6 # microseconds
    v = np.array([gun.x_dot(x) for x in gun.x])/1e5 # km/s
    ax1.plot(t,v,label=r'$v(t)$')
    #ax1.legend(loc='lower left')
    #plt.show()
    fig.savefig('fig.pdf', format='pdf')
    
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    plot()
    #_test()

#---------------
# Local Variables:
# mode: python
# End:
