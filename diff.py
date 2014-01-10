'''diff.py Plots the difference between two estimates of the Perron
Frobenius function, v_{PF}, ie, the eigenfunction corresponding to the
largest eigenvector of the first order Markov integral operator.

Default arguments have n_g and n_h take values (200,200) and
(225,225).

'''
def plot(op, # Linear operator
         f   # Difference of v_{PF}
    ):
    '''Function to plot a result of main.  
    '''
    import numpy as np
    import matplotlib as mpl
    g,h = op.gh()
    z = op.vec2z(f)
    G,H = np.meshgrid(g, h)
    params = {'axes.labelsize': 18,     # Plotting parameters for latex
              'text.fontsize': 15,
              'legend.fontsize': 15,
              'text.usetex': True,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15}
    mpl.rcParams.update(params)
    if True:
        DEBUG = True
        mpl.rcParams['text.usetex'] = False
    else:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use
    from mpl_toolkits.mplot3d import Axes3D  # Mysteriously necessary
                                             #for "projection='3d'".
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    fig = plt.figure(figsize=(24,12))
    ax = fig.add_subplot(1,1,1, projection='3d', elev=5, azim=15)
    surf = ax.plot_surface(
            G, H, z, rstride=1, cstride=1, cmap=mpl.cm.jet, linewidth=1,
            antialiased=False)
    ax.set_xlabel(r'$g$')
    ax.set_ylabel(r'$h$')
    plt.show()
import sys
from first_c import LO
def main(argv=None):
    import argparse
    import numpy as np
    global DEBUG
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='''Plot differences between estimates of the Perron
        Frobenius function of an integral operator''')
    parser.add_argument('--u', type=float, default=(2.0e-5),
                       help='log fractional deviation')
    parser.add_argument('--dy', type=float, default=3.2e-4,
                       help='y_1-y_0 = log(x_1/x_0)')
    parser.add_argument('--n_g0', type=int, default=200,
                       help='number of integration elements in value')
    parser.add_argument('--n_h0', type=int, default=200, help=
'number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    parser.add_argument('--n_g1', type=int, default=225)
    parser.add_argument('--n_h1', type=int, default=225)
    args = parser.parse_args(argv)

    d_g = {'big':2*args.u/args.n_g0, 'small':2*args.u/args.n_g1}
    h_lim = np.sqrt(48*args.u)
    d_h = {'big':2*h_lim/args.n_h0, 'small':2*h_lim/args.n_h1}
    
    from scipy.sparse.linalg import LinearOperator
    
    tol = 5e-6
    maxiter = 150
    LO_ = {}
    for size in ('small', 'big'):
        op = LO( args.u, args.dy, d_g[size], d_h[size])
        op.power(small=tol, n_iter=maxiter)
        print('For %s, n_g=%d and n_h=%d\n'%(size, op.n_g, op.n_h))
        LO_[size] = op
    diffs = {}
    for a,b in (('big','small'),('small','big')):
        x = LO_[a].xyz()
        d, f = LO_[b].diff(x[0],x[1],x[2],rv=True)
        print('%s.diff(%s)=%g'%(b, a, d))
        plot(LO_[a], f)
    return 0
    
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
