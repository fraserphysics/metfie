"""eos.py: Properties of the "EOS" function volume -> pressure.
Used in calc.py. Goals:

1. Develop, analyze and understand a procedure for estimating an
   isentrope on the basis of data and simulations.

2. Demonstrate provenance tracking using Component, Provenance and Float
   from cmf_models

"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
from cmf_models import Component, Provenance, Float
from markup import oneliner
#import matplotlib.pyplot as plt

class go:
    ''' Generic object.  For storing magic numbers.
    '''
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
magic = go(
    C=2.56,                  # For nominal eos. GPa at 1 cm^3 / gram
    spline_N=50,             # Number of samples (number of knots - 4)
    spline_min=0.1,          # Minimum volume
    spline_max=100,          # Maximum volume
    spline_uncertainty=.005, # Multiplicative uncertainty of prior (1/2 %)
    spline_end=4,            # Last 4 coefficients of splines are 0
    freq=.2,                 # sin term of perturbation
    w=.2,                    # Perturbation width
    v_0=0.6,                 # Center of perturbation
    scale=2.0,               # Scale of perturbation
           )

class Nominal:
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
class Experiment:
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
        pert = 2*np.sin(freq*(v-v_0))*np.exp(-(v-v_0)**2/(2*w**2))
        return self.C/v**3 + pert*self.C/(freq*v_0**3)
class Spline(IU_Spline, Component):
    '''Model of pressure as function of volume implemented as a spline.
    
    Methods:

    Pq
    Gh
    '''
    def __init__(
            self,   # Spline instance
            P,      # Pressure function (usually the nominal eos)
            N=magic.spline_N,
            v_min=magic.spline_min,
            v_max=magic.spline_max,
            uncertainty=magic.spline_uncertainty,
            precondition=False,
            comment='',
            ):
        self.end = magic.spline_end
        v = np.logspace(np.log10(v_min), np.log10(v_max), N)
        IU_Spline.__init__(self,v,P(v))
        Component.__init__(self, self.get_c(), comment)
        c_P = self.get_c()[:-self.end]
        self.prior_mean = c_P.copy()
        dev = c_P*uncertainty
        self.prior_var_inv = np.diag(1.0/(dev*dev))
        self.precondition = precondition
        if precondition:
            self.U_inv = np.diag(dev)
        return
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
    def display(
            self,      # Spline instance
            show=False # Call plt.show() for debugging
            ):
        '''This method serves the Component class and the make_html
        function defined in the cmf_models module.  It returns an html
        description of self and writes a plot to 'eos.jpg'.
        '''
        x_label = r'$v*{\rm cm}^3/{\rm g}$'
        y_label = r'$p/{\rm GPa}$'
        t = self.get_t()
        
        fig = plt.figure('eos', figsize=(7,10))
        ax = fig.add_subplot(2,1,1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        low = magic.v_0/2
        high = magic.v_0*3
        x = np.linspace(low,high,200)
        mask = np.where((low < t)*(t < high))[0]
        x_ = t[mask]
        ax.plot(x, self(x))
        ax.plot(x_, self(x_),'rx')
        
        ax = fig.add_subplot(2,1,2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        x = np.logspace(
            np.log10(magic.spline_min),
            np.log10(magic.spline_max),
            500)
        ax.loglog(x, self(x))
        ax.loglog(t, self(t), 'rx')
        if show:
            plt.show()
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

    def Pq(
        self,  # Spline instance
        c_P,   # Current spline coefficients
        ep_f,  # c_f - c_{\tilde f}
        BD,    # Derivative of v wrt c_f
        ep_v,  # Array of errors
        ):
        '''From eq:PqR (Eqn17) of metfie/like_gun/notes.tex
        '''
        P = self.Sigma_f_inv
        ep_f = c_P - self.prior_mean
        q = np.dot(ep_f, self.prior_var_inv)
        if self.precondition:
            P = np.dot(self.U_inv, np.dot(P, self.U_inv))
            q = np.dot(self.U_inv, q)
        s = np.linalg.svd(P, compute_uv=False)[0] # Largest singular value
        return P/s,q/s
    
    def Gh(
            self,  # Spline instance
            c_P,   # Current spline coefficients
    ):
        ''' Calculate constraint matrix G and vector h.  The
        constraint enforced by cvxopt.solvers.qp is

        G*x leq_component_wise h

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
        P_work = self.spline.copy()
        c_work = P_work.get_c()
        dim = len(self.prior_mean)
        v_all = self.spline.get_t()
        v_unique = v_all[self.end-1:1-self.end]
        n_v = len(v_unique)
        n_constraints = n_v + 2
        G = np.zeros((n_constraints, dim))
        for i in range(dim):
            c[i] = 1.0
            P_work.set_c(c)
            G[:-2,i] = -P_work.derivative(2)(t_unique)
            G[-2,i] = P_work.derivative(1)(t_unique[-1])
            G[-1,i] = -P_work(t_unique[-1])
            c[i] = 0.0
        h = -np.dot(G,c_P)

        # Scale to make |h[i]| = 1 for all i
        HI = np.diag(1.0/np.abs(h))
        h = np.dot(HI,h)
        G = np.dot(HI,G)

        if self.precondition: # If P is preconditioned, must modify G
            G = np.dot(G, self.U_inv)
        return -G,h
    
# Test functions
import numpy.testing as nt
def test_spline():
    nominal = Nominal()
    experiment = Experiment()
    spline = Spline(experiment)

    assert spline.provenance.line == 'spline = Spline(experiment)'
    x = spline.get_knots()
    nt.assert_allclose(spline(x), experiment(x), rtol=1e-15)
    return 0
def test():
    ''' Execute collection of test functions
    '''
    test_spline()
    return 0
def work():
    ''' This code for debugging stuff will change often
    '''
    nominal = Nominal()
    experiment = Experiment()
    spline = Spline(experiment)

    spline.display(show=True)
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
