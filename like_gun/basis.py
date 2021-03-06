'''Illustrate spline basis function and its derivatives.
'''

from eos import Spline
import numpy as np
def make_figure(plt):
    x_n=10      # Xvalues from 0 to 10
    n_d = 4     # Number of derivatives f, f', f'', f'''
    n_x = 1000  # Number of x points for plots
    
    fig = plt.figure('basis functions',figsize=(8,10))
    ax = tuple(fig.add_subplot(3,1,i) for i in (1,2,3))
    x = np.linspace(0,x_n,x_n+1)
    s = Spline(x,np.zeros(x.shape)) # Function is zero everywhere
    c = s.get_c()                   # The coefficients
    t = s.get_t()[3:-3]             # The knot locations
    x = np.linspace(0, x_n, n_x)
    n_x = len(x)
    d = np.empty((n_d,n_x))
    d2 = np.empty((len(t),len(c)))
    for k in range(len(c)):
        c[k] = 1.0
        s.new_c(c)
        d2[:,k] = s.derivative(2)(t)
        for i in range(n_x):
            d[:,i] = s.derivatives(x[i])
        for i in range(3):
            if k == 5:
                ax[i].plot(x,d[i,:], lw=2, linestyle='--', label=str(k))
            else:
                ax[i].plot(x,d[i,:], label=str(k))
            ax[i].plot(t,np.zeros(t.shape),'rx')
        c[k] = 0.0
    for axis, label in zip(ax, "f \\frac{df}{dx} \\frac{d^2f}{df^2}".split()):
        axis.set_ylabel(r'${0}$'.format(label))
    return fig
  
if __name__ == "__main__":
    import sys
    import matplotlib as mpl
    if len(sys.argv) == 2:
        mpl.use('PDF')
    import matplotlib.pyplot as plt
    fig = make_figure(plt)
    if len(sys.argv) == 2:
        fig.savefig(sys.argv[1], format='pdf')
    else:
        plt.show()
    sys.exit(0)

#---------------
# Local Variables:
# mode: python
# End:
