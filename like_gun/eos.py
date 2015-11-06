"""eos.py: Properties of the "EOS" function volume -> pressure.
Used in calc.py. Goals:

1. Develop, analyze and understand a procedure for estimating an
   isentrope on the basis of data and simulations.

"""
import numpy as np

class Go:
    ''' Generic object.
    '''
    def __init__(self, **kwargs):
        self.add(**kwargs)
    def add(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
magic = Go(
    C=2.56e9,                # For nominal eos. Pa at 1 cm^3 / gram
    spline_N=50,             # Number of samples (number of knots - 4)
    spline_min=0.1,          # Minimum volume
    spline_max=100,          # Maximum volume
    spline_uncertainty=5.e-3,# Multiplicative uncertainty of prior (1/2 %)
    spline_end=4,            # Last 4 coefficients of splines are 0
    freq=.2,                 # sin term of perturbation
    w=.2,                    # Perturbation width
    v_0=0.6,                 # Center of perturbation
    scale=2.0,               # Scale of perturbation
           )

R = lambda vel, vol_0, vol: vel*vel*(vol_0-vol)/(vol_0*vol_0)
from scipy.optimize import brentq
class Isentrope:
    '''Provides method to calculate CJ conditions
    '''
    def CJ(self, vol_0):
        '''Find CJ conditions using two nested line searches.
        '''
        # Search for det velocity between 1 and 10 km/sec
        d_min = 1.0e5
        d_max = 1.0e6
        v_min = magic.spline_min
        v_max = magic.spline_max

        # R is Rayleigh line
        R = lambda vel, vol: vel*vel*(vol_0-vol)/(vol_0*vol_0)
        # F is self - R
        F = lambda vel, vol, self : self(vol) - R(vel, vol)
        # d_F is derivative of F wrt vol
        d_F = lambda vol, vel, self: self.derivative(1)(vol) + (
            vel/vol_0)**2
        
        # arg_min(vel, self) finds volume that minimizes self(v) - R(v)
        arg_min = lambda vel,self: brentq(d_F,v_min,v_max,args=(vel,self))
        
        E = lambda vel, self: F(vel, arg_min(vel,self), self)
        
        vel_CJ = brentq(E, d_min, d_max, args=(self,))
        vol_CJ = arg_min(vel_CJ,self)
        p_CJ = self(vol_CJ)
        return vel_CJ, vol_CJ, p_CJ
class Nominal(Isentrope):
    '''Model of pressure as function of volume.  This is the nominal eos

    p = C/v^3
    '''
    def __init__(
            self,      # Nominal instance
            C=magic.C,
            ):
        self.C = C
        return
    def __call__(
            self,      # Nominal instance
            v          # Specific volume in cm^3/gram
            ):
        return self.C/v**3
    def derivative(self, n):
        assert n == 1
        return lambda v: -3*self.C/v**4
class Experiment(Isentrope):
    '''This is the "true" eos used for the experiments.

    P(v) =
    C/v^3 + 2 sin(freq*(v-v_0)) * e^{(v-v_0)^2/(2*w^2)} * C/(freq*(v-v_0)^3)
    '''
    def __init__(
            self,              # Experiment instance
            C=magic.C,
            v_0=magic.v_0,
            freq=magic.freq,
            w=magic.w,
            ):
        self.C = C
        self.v_0 = v_0
        self.freq = freq
        self.w = w
        return
    def __call__(
            self,      # Experiment instance
            v          # Specific volume in cm^3/gram
            ):
        freq = self.freq
        v_0 = self.v_0
        w = self.w
        sin = lambda Dv: np.sin(freq*Dv)
        exp = lambda Dv: np.exp(-Dv**2/(2*w**2))
        pert = lambda Dv: np.sin(freq*Dv)*np.exp(-Dv**2/(2*w**2))
        factor = 2*self.C/(freq*v_0**3)
        f_Dv = lambda Dv : factor * sin(Dv)*exp(Dv)
        return self.C/v**3 + f_Dv(v-v_0)
    def derivative(self, n):
        assert n == 1
        freq = self.freq
        v_0 = self.v_0
        w = self.w
        sin = lambda Dv: np.sin(freq*Dv)
        dsin = lambda Dv: freq*np.cos(freq*Dv)
        exp = lambda Dv: np.exp(-Dv**2/(2*w**2))
        dexp = lambda Dv: -Dv/w*exp(Dv)
        pert = lambda Dv: np.sin(freq*Dv)*np.exp(-Dv**2/(2*w**2))
        factor = 2*self.C/(freq*v_0**3)
        df = lambda Dv: factor*(dsin(Dv)*exp(Dv) + sin(Dv)*dexp(Dv))            
        return lambda v:-3*self.C/v**4 + df(v-v_0)
    
from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py
class Spline(IU_Spline):
    def get_t(self):
        'Return the knot locations'
        return self._eval_args[0]
    def get_c(self):
        'Return the coefficients for the basis functions'
        return self._eval_args[1][:-magic.spline_end]
    def new_c(self,c):
        '''Return a new Spline_eos instance that is copy of self except
        that the coefficients for the basis functions are c.'''
        import copy
        from inspect import stack
        rv = copy.deepcopy(self)
        c_ = np.zeros(self._eval_args[1].shape)
        c_[:-magic.spline_end] = c
        rv._eval_args = self._eval_args[0], c_, self._eval_args[2]
        return rv
class Spline_eos(Spline, Isentrope):
    '''Model of pressure as function of volume implemented as a spline.
    
    Methods:

    Pq_prior  Describes portion of cost function due to the prior
    Pq_like   Describes portion of cost function due to a likelihood
    Gh        Describes constraints: convex, monotonic and positive
    '''
    def __init__(
            self,   # Spline_eos instance
            P,      # Pressure function (usually the nominal eos)
            N=magic.spline_N,
            v_min=magic.spline_min,
            v_max=magic.spline_max,
            uncertainty=magic.spline_uncertainty,
            precondition=False,
            comment='',
            ):
        v = np.logspace(np.log10(v_min), np.log10(v_max), N)
        IU_Spline.__init__(self,v,P(v))
        self.prior_mean = self.get_c().copy()
        dev = self.prior_mean*uncertainty
        self.prior_var_inv = np.diag(1.0/(dev*dev))
        self.precondition = precondition
        if precondition:
            self.U_inv = np.diag(dev)
        return
    def dev(self):
        '''Calculate and return a spline for plotting uncertainty of
        the pressure function.

        The function is the square root of the marginal covariance.
        '''
        c = np.zeros(len(self.get_c()))
        c_dev = np.zeros(len(c))
        for i in range(len(c)):
            c[i] = 1.0
            var_x_i = np.linalg.solve(self.prior_var_inv, c)
            c_dev[i] = np.sqrt(var_x_i[i])
            c[i] = 0.0
        return self.new_c(c_dev)
    def Pq_prior(
        self,  # Spline_eos instance
        c_P,   # Current spline coefficients
        ):
        '''Return P and q for the cost of the prior:

           (x^T P x)/2 + q^T x
        '''
        P = self.prior_var_inv
        ep_f = c_P - self.prior_mean
        q = np.dot(ep_f, self.prior_var_inv)
        if self.precondition:
            P = np.dot(self.U_inv, np.dot(P, self.U_inv))
            q = np.dot(q, self.U_inv)
        else:
            P = P.copy() # Will be used as "P += ..."
        # Could scale by largest singular value of P
        # np.linalg.svd(P, compute_uv=False)[0]
        return P, q
    def Pq_like(
            self,        # Eos instance
            D_k,         # d sim /d f matrix
            ep_k,        # v_exp - v_sim
            Sigma_inv_k, # Covariance martix for exp
            ):
        '''See the section "A Posteriori Probability" of notes.tex,

        P = Sigma_f^{-1} + \sum_k D_k^T Sigma^{-1}_k D_k
        q^T = ep_f^T Sigma_f^{-1} - \sum_k ep_k^T Sigma_v^{-1} D_k
        R = ep_f^T Sigma_f^{-1} ep_f + ep_k^T Sigma_k^{-1} ep_k

        This method calculates the contributions to P and q from the
        likelihood of an experiment, ie:
        
        P_k =  D_k^T Sigma^{-1}_k D_k
        q_k^T =  - ep_k^T Sigma_k^{-1} D_k

        If eos.precondition use these \tilde values instead:

        \tilde P = U^{-1} P U^{-1} (variable t_P here)
        \tilde q^T = q^T U^{-1} (variable t_q here)
        '''
        Sigma_inv_D = np.dot(Sigma_inv_k, D_k)
        P = np.dot(D_k.T, Sigma_inv_D)
        q = -np.dot(ep_k, Sigma_inv_D)
        if self.precondition:
            P = np.dot(self.U_inv, np.dot(P, self.U_inv))
            q = np.dot(self.U_inv, q)
        return P, q
    def Gh(
            self,  # Spline_eos instance
            c_P,   # Current spline coefficients
    ):
        ''' Calculate constraint matrix G and vector h.  The
        constraint enforced by cvxopt.solvers.qp is

        G*x leq_component_wise h

        Equivalent to max(G*x - h) leq 0

        Since

        c_f_new = c_f+x,

        G(c_f+x) leq_component_wise 0

        is the same as

        G*x leq_component_wise -G*c_f,

        and h = -G*c_f

        Here are the constraints for p(v):

        p'' positive for all v
        p' negative for v_max
        p positive for v_max

        For cubic splines between knots, f'' is constant and f' is
        affine.  Consequently, f''*rho + 2*f' is affine between knots
        and it is sufficient to check eq:star at the knots.
        
        '''
        dim = len(self.prior_mean)
        v_all = self.get_t()
        v_unique = v_all[magic.spline_end-1:1-magic.spline_end]
        n_v = len(v_unique)
        n_constraints = n_v + 2
        G = np.zeros((n_constraints, dim))
        c = np.zeros(dim)
        for i in range(dim):
            c[i] = 1.0
            P_work = self.new_c(c)
            G[:-2,i] = -P_work.derivative(2)(v_unique)
            G[-2,i] = P_work.derivative(1)(v_unique[-1])
            G[-1,i] = -P_work(v_unique[-1])
            c[i] = 0.0
        h = -np.dot(G,c_P)

        scale = np.abs(h)
        scale = np.maximum(scale, scale.max()*1e-15)
        # Scale to make |h[i]| = 1 for all i
        HI = np.diag(1.0/scale)
        h = np.dot(HI,h)
        G = np.dot(HI,G)

        if self.precondition: # If P is preconditioned, must modify G
            G = np.dot(G, self.U_inv)
        return G,h
    def log_prior(
            self, # Spline_eos instance
            c,    # Proposed new spline coefficients
            ):
        ''' log of prior evaluated at c is ep_f^T * P/2 * ep_f where:
             ep_f is c - mu and
             P is inverse covariance
        '''
        ep_f = c - self.prior_mean
        return -np.dot(ep_f, np.dot(self.prior_var_inv, ep_f))/2
    
# Test functions
close = lambda a,b: a*(1-1e-7) < b < a*(1+1e-7)
import numpy.testing as nt
def test_CJ():
    v_0 = 1/1.835
    velocity, volume, pressure = Nominal().CJ(v_0)
    #print('D_CJ={0:.4e}'.format( velocity))
    assert close(volume, 0.408719346049324)
    assert close(pressure, 3.749422497177544e10)
    assert close(velocity,                            2.858868318e+05)
    assert close(Spline_eos(Nominal()).CJ(v_0)[0],    2.858856308e+05)
    assert close(Spline_eos(Experiment()).CJ(v_0)[0], 2.744335208e+05)
    assert close(Experiment().CJ(v_0)[0],             2.745241393e+05)
    return 0
def test_spline():
    '''For convex combinations of nominal and experimental, ensure that
    minimum cost is at nominal and that feasible boundary is between
    .55 and .60.  Also check preconditioning.
    '''

    nominal = Nominal()
    experiment = Experiment()
    for pre_c in (True, False):
        s_nom = Spline_eos(nominal,precondition=pre_c)
        c_nom = s_nom.get_c() # Coefficients for nominal spline
    
        s_exp = Spline_eos(experiment, precondition=pre_c)
        c_exp = s_exp.get_c() # Coefficients for experiment spline
        # spline should match data at the knots
        t = s_exp.get_knots()
        nt.assert_allclose(s_exp(t), experiment(t), rtol=1e-15)
    
        d_c = c_exp - c_nom
        P,q = s_nom.Pq_prior(c_nom)
        G,h = s_nom.Gh(c_nom)

        cost = np.empty(21)
        feasible = np.empty(21)
        for i,a in enumerate(np.linspace(0,1,21)):
            c = d_c*a
            if pre_c:
                c = np.linalg.solve(s_nom.U_inv, c)
            cost[i] = np.dot(c,np.dot(P,c))/2 + np.dot(q,c)
            y = np.dot(G,c) - h
            feasible[i] = float(y.max())
        assert np.argmin(cost) == 0
        assert feasible[11]*feasible[12] < 0
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
    nominal = Nominal()
    experiment = Experiment()
    
    spline = Spline_eos(nominal)
    spline.display()
    
    spline = Spline_eos(experiment)
    spline.display()
    
    plt.show()
    return 0
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) >1 and sys.argv[1] == 'work':
        sys.exit(work())
    rv = test()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
