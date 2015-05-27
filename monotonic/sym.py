'''sym.py for the distribution of monotonic functions.
'''
import sys
import numpy as np
import sympy
from sympy import collect
params = {'axes.labelsize': 18,     # Plotting parameters for latex
          'text.fontsize': 15,
          'legend.fontsize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
import matplotlib as mpl
import numpy.linalg as LA

class Interval:
    def __init__(self, n=None, predecessor=None):
        '''
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
    def __init__(self, n):
        last = Interval(n=n)
        self.intervals = []
        for k in range(n):
            self.intervals.append(last)
            last = Interval(predecessor=last)
        self.bins = np.linspace(0, 1, n+1)
    def __call__(self, x):
        i = np.digitize([x], self.bins) - 1
        assert 0 <= i < len(self.intervals)
        return self.intervals[i](x)
        
def main(argv=None):
    import matplotlib as mpl
    mpl.use('Qt4Agg', warn=False)
    import matplotlib.pyplot as plt  # must be after mpl.use
    f = Piecewise(5)
    x = np.linspace(0, 1, 200, endpoint=False)
    y = tuple(f(t) for t in x)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y)
    plt.show()
    return 0

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
