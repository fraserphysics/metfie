'''sym.py for the distribution of monotonic functions.
'''
plot_dict = {} # Keys are those keys in args.__dict__ that ask for
               # plots.  Values are the functions that make those
               # plots.
class go:
    ''' Generic object.  EG, the value of go(x=5).x is 5.
    '''
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
import sys
import numpy as np
import sympy
params = {
    'axes.labelsize': 18,     # Plotting parameters for latex
    'font.size': 15,
    'legend.fontsize': 15,
    'text.usetex': True,
    'font.family':'serif',
    'font.serif':'Computer Modern Roman',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15}
import matplotlib as mpl

class Eigenfunction:
    def __init__(
            self,
            tau,  # A float
            start=None,
            ):
        print('''
tau={0}'''.format(tau))
        if start==None:
            start = tau -.05 + 2*tau
        x,lam = sympy.symbols('x lam'.split())
        self.tau = tau
        k = 0
        term = 1.0 + x*0   # \frac{1}{k!} * (-\frac{x}{lam})^k
        pk = 0             # sum_{n=0}^k term_k
        cum_prod = 1       # prod_{n=0}^k pk(tau)
        integral = 0       # integral_0^? rho(s) ds
        while tau*k < 1.0:
            pk += term              # Polynomial of x/lam
            pk_tau = pk.subs(x,tau) # Polynomial of tau/lam
            print('''  k={0}
  term={1}
  pk={2}
  '''.format(k, term, pk))
            if tau*(k+1) >= 1.0:
                d = 1.0 - tau*k
                integral += sympy.integrate(pk, (x, 0, d))/cum_prod
                break
            integral += sympy.integrate(pk, (x, 0, tau))/cum_prod
            # integral is a rational function of tau and lam
            cum_prod *= pk_tau      # Polynomial of tau/lam
            k += 1
            term *= -x/(k*lam)      # Monomial of x/lam
        print('''  cum_prod={2}  Solving
  {0}={1}'''.format(lam,integral.simplify(), cum_prod))
        self.eigenvalue = sympy.solve(lam-integral, lam)
        #FixMe: what is start?
            
class Interval:
    '''Represents eigenfunction in range [k/n, (k+1)/n)
    '''
    def __init__(
            self,              # Interval instance
            n=None,            # Number of segments between 0 and 1
            predecessor=None   # Previous interval
            ):
        '''If n given, initialize self as first interval, otherwise
         initialize self by integrating predecessor.
        '''
        assert bool(n)^bool(predecessor) # Exclusive or
        t,lam = sympy.symbols('t lam'.split())
        if bool(n):
            k = 0
            self.f = 1+0*t      # The function
        else:
            n = predecessor.n
            k = predecessor.k+1
            self.f = predecessor.f_f - sympy.integrate(
                predecessor.f,(t,predecessor.t_f,t))/lam
        self.n = n
        self.k = k
        self.t_i = float(k)/float(n)
        self.t_f = float(k+1)/float(n)
        self.f_f = self.f.subs(t,self.t_f) # Function of lam
        return
    def __call__(
            self,  # Interval instance
            t_val, # float value of t
            e_val  # float value of eigenvalue
            ):
        ''' Calculate and return float evaluation of self at t_val, e_val
        '''
        t,lam = sympy.symbols('t lam'.split())
        return self.f.subs(((t,t_val),(lam,e_val)))

class Piecewise:
    '''Represents whole eigenfunction in n pieces
    '''
    def __init__(
            self,  # Piecewise instance
             n     # Number of segments between 0 and 1
            ):
        last = Interval(n=n)
        self.intervals = []
        for k in range(n):
            self.intervals.append(last)
            last = Interval(predecessor=last)
        self.bins = np.linspace(0, 1, n+1)
    def __call__(
            self,   # Piecewise instance
            t_vals, # 1-d numpy array
            e_val,  # eigenvalue
            ):
        ''' Calculate and return value of function at t_vals and e_val
        '''
        n_t = len(t_vals)
        i = np.digitize(t_vals, self.bins) - 1
        assert i.shape == (n_t,)
        y = np.empty(n_t)
        #print 'n={0:d}'.format(len(self.intervals))
        for k in range(n_t):
            i_ = i[k]
            assert 0 <= i_ < len(self.intervals)
            y[k] = self.intervals[i_](t_vals[k], e_val)
        return y
    def integrate(self):
        '''Calculate and return definite integral from 0 to 1 with
        lam as free variable.
        '''
        t = sympy.symbols('t')
        rv = 0.0
        for i in self.intervals:
            rv += sympy.integrate(i.f,(t,i.t_i,i.t_f))
        return rv

def e_val(tau, start=None):
    '''Calculate approximate eigenvalue using eq:2
    '''
    t,lam = sympy.symbols('t lam'.split())
    f = t - lam*sympy.exp((t-1)/lam)
    f_tau = f.subs(t,tau)
    if start == None:
        start = tau
    rv = sympy.nsolve(f_tau, lam, start)#, tol=1e-8)
    return rv
vals = {}
funcs = {}
vals_n = {}
funcs_n = {}
def calculate(args):
    '''For 20 values of n, set vals[n] to float representation of eigenvalue
    and set funcs[n] to corresponding eigenfunction with lambda as free
    variable.
    '''
    from first import LO
    if len(vals) != 0:
        return
    t,lam = sympy.symbols('t lam'.split())
    old = 1.0
    delta = 0
    ns = np.arange(1, args.n_calculate)
    for n in ns:
        T = 1.0/n
        A = LO(T, 1.0/3000)
        A.power(op=A.matvec, small=1.0e-8)
        vals_n[n] = A.eigenvalue
        funcs_n[n] = A.eigenvector
        F = Piecewise(n)
        funcs[n] = F
        new_ = sympy.nsolve(lam-F.integrate(), lam, old+delta)
        vals[n] = new_
        delta = new_ - old
        old = new_
        print('For n={0:d}, lambda={1}, delta={2}'.format(n,new_,delta))
def eigenvalues(args, plt):
    '''
    '''
    from first import LO
    calculate(args)
    ns_v = np.arange(1,len(vals)+1)
    y = []
    y_lo = []
    for n in ns_v:
        y.append(vals[n])
        y_lo.append(vals_n[n])
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(1.0/ns_v, y, label=r'$\lambda$')
    ax.plot(1.0/ns_v, y_lo, label=r'numeric')
    ns_a = np.arange(1,1000)
    approx = []
    val = 1.0 # first guess
    for n in ns_a:
        val = e_val(1.0/n, val)
        approx.append(val)
    ax.plot(1.0/ns_a, approx, label=r'$\tilde \lambda$')
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$\lambda_\tau$')
    ax.legend(loc='lower right')
    return fig
plot_dict['eigenvalues']=eigenvalues

def eigenfunctions(args, plt):
    '''Broken code.
    '''
    calculate(args)
    lam = sympy.symbols('lam')
    ns = args.ns
    x = np.linspace(0, 1, 100, endpoint=False)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)
    for n in ns:
        y = funcs[n](x,vals[n])
        ax.plot(x,y,lw=2,label=r'$n=%d$'%n)
        y_ = funcs_n[n]
        x_ = np.linspace(0,1,len(y_))
        ax.plot(x_,y_/y_[0], linestyle='dotted',color='k')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(r'$g$')
    ax.set_ylabel(r'$_{_L}\rho_{\frac{1}{n}}(g)$')
    ax.legend(loc='lower left')
    return fig
plot_dict['eigenfunctions']=eigenfunctions

def test():
    for tau,start in ((1.0,1.0), (.5,.85), (1.0/3, .65)):
        f = Eigenfunction(tau, start=start)
        print('tau={0}, lambda={1}'.format(tau, f.eigenvalue))
    return 0

def main(argv=None):
    '''Makes plots of eigenfunctions

    '''
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--ns', nargs='*', type=int, default=(1,2,3,5,10),
        help='number of segments/samples in interval [0,1)')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--n_calculate', type=int, default=21,
        help='Calculate eigenfunctions and eigenvalues for 0 < n < n_calculate')
    #
    for plot in 'eigenfunctions eigenvalues'.split():
        parser.add_argument('--{0}'.format(plot), type=str, default=None,
            help="Write {0} plot to this file".format(plot))
    args = parser.parse_args(argv)
    assert args.ns[-1] < args.n_calculate
    
    if args.test:
        test()
        return 0
    mpl.use('PDF', warn=False)
    import matplotlib.pyplot as plt  # must be after mpl.use
    mpl.rcParams.update(params)

    for key in args.__dict__:
        if key not in plot_dict:
            continue
        if args.__dict__[key] == None:
            continue
        fig = plot_dict[key](args, plt)
        fig.savefig(getattr(args, key), format='pdf')
    return 0

if __name__ == "__main__":
    #rv = main()
    rv = test()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
