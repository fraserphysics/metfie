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

class Interval:
    '''Represents eigenfunction in range [k/n, (k+1)/n)
    '''
    def __init__(self, n=None, predecessor=None):
        '''If n given, initialize self as first interval, otherwise
         initialize self by integrating predecessor.
        '''
        assert bool(n)^bool(predecessor) # Exclusive or
        t = sympy.symbols('t')
        if bool(n):
            k = 0
            self.f = 1+0*t      # The function
        else:
            n = predecessor.n
            k = predecessor.k+1
            self.f = predecessor.f_f - sympy.integrate(
                predecessor.f,(t,predecessor.t_f,t))
        self.n = n
        self.k = k
        self.t_i = float(k)/float(n)
        self.t_f = float(k+1)/float(n)
        self.f_f = self.f.subs(t,self.t_f).evalf()
        return
    def __call__(self,x):
        t = sympy.symbols('t')
        return self.f.subs(t,x).evalf()

class Piecewise:
    '''Represents whole eigenfunction in n pieces
    '''
    def __init__(self, n):
        last = Interval(n=n)
        self.intervals = []
        for k in range(n):
            self.intervals.append(last)
            last = Interval(predecessor=last)
        self.bins = np.linspace(0, 1, n+1)
    def __call__(
            self, # Piecewise instance
            x,    # 1-d numpy array
            ):
        n_x = len(x)
        i = np.digitize(x, self.bins) - 1
        assert i.shape == (n_x,)
        y = np.empty(n_x)
        #print 'n={0:d}'.format(len(self.intervals))
        for k in range(n_x):
            i_ = i[k]
            assert 0 <= i_ < len(self.intervals)
            y[k] = self.intervals[i_](x[k])
            #print '{0:d}  {1:6.3f}  {2:6.3f}'.format(i_, x[k], y[k])
        return y

def eigenfunctions(args, plt):
    ns = args.ns
    x = np.linspace(0, 1, 500, endpoint=False)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)
    for n in ns:
        y = Piecewise(n)(x)
        ax.plot(x,y,label=r'$n=%d$'%n)
    y = np.exp(-x)
    ax.plot(x,y,label=r'$n=\infty$')
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(r'$g$')
    ax.set_ylabel(r'$\rho_n(g)$')
    ax.legend(loc='lower left')
    return fig
plot_dict['eigenfunctions']=eigenfunctions

def test():
    import matplotlib as mpl
    mpl.use('Qt4Agg', warn=False)
    import matplotlib.pyplot as plt  # must be after mpl.use
    mpl.rcParams.update(params)
    fig = eigenfunctions(go(ns=(1,2,3,10,50)), plt)
    plt.show()
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
    #
    parser.add_argument('--eigenfunctions', type=str, default=None,
        help="Write plot to this file")
    args = parser.parse_args(argv)
    
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
    rv = main()
    #rv = test()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
