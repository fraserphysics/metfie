import sys
from first_c import LO # user    0m10.177s
#from first import LO   # user    1m35.858s
DEBUG = False
def main(argv=None):
    '''For looking at sensitivity of time and results to u, dy, n_g, n_h.

    '''
    import argparse
    global DEBUG
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Plot eigenfunction of the integral equation')
    parser.add_argument('--u', type=float, default=(2.0e-5),
                       help='log fractional deviation')
    parser.add_argument('--dy', type=float, default=3.2e-4,
                       help='y_1-y_0 = log(x_1/x_0)')
    parser.add_argument('--n_g', type=int, default=200,
                       help='number of integration elements in value')
    parser.add_argument('--n_h', type=int, default=200,
help='number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    parser.add_argument('--n_g2', type=int, default=220,
                       help='number of integration elements in value for 2')
    parser.add_argument('--n_h2', type=int, default=275,
help='number of integration elements in slope for 2')
    parser.add_argument('--m_g', type=int, default=50,
                       help='number of points for plot in g direction')
    parser.add_argument('--m_h', type=int, default=50,
                       help='number of points for plot in h direction')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--file', type=str, default=None,
        help="where to write result")
    args = parser.parse_args(argv)
    import matplotlib as mpl
    params = {'axes.labelsize': 18,     # Plotting parameters for latex
              'text.fontsize': 15,
              'legend.fontsize': 15,
              'text.usetex': True,
              'font.family':'serif',
              'font.serif':'Computer Modern Roman',
              'xtick.labelsize': 15,
              'ytick.labelsize': 15}
    mpl.rcParams.update(params)
    if args.debug or args.file == None:
        DEBUG = True
        mpl.rcParams['text.usetex'] = False
    else:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use
    from scipy.sparse.linalg import LinearOperator
    from mpl_toolkits.mplot3d import Axes3D  # Mysteriously necessary
                                             #for "projection='3d'".
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np
    
    def plot(L, floor, ax):
        g,h = L.gh(args.m_g, args.m_h)
        G,H = np.meshgrid(g, h)
        b = np.log10(np.fmax(L.eigenvector, floor))
        z = L.vec2z(b, g, h, np.log10(floor))
        surf = ax.plot_surface(
            G, H, z.T, rstride=1, cstride=1, cmap=mpl.cm.jet, linewidth=1,
            antialiased=False)
        ax.set_xlabel(r'$g$')
        ax.set_ylabel(r'$h$')
        ax.set_title(r'$n_g=%d$ $n_h=%d$ $dy=%8.3g$ $\lambda=%8.3g$'%(
            L.n_g, L.n_h, L.dy, L.eigenvalue))

    fig = plt.figure(figsize=(24,12))
    ax1 = fig.add_subplot(1,2,1, projection='3d', elev=15, azim=-45)
    ax2 = fig.add_subplot(1,2,2, projection='3d', elev=15, azim=-45)
    
    tol = 5e-6
    maxiter = 1000
    
    A = LO( args.u, args.dy, args.n_g, args.n_h)
    A.power(small=tol, n_iter=maxiter, verbose=True)
    
    B = LO( args.u, args.dy, args.n_g2, args.n_h2)
    B.power(small=tol, n_iter=maxiter, verbose=True)

    from first import sym_diff
    import time
    t1 = time.time()
    d = sym_diff(A,B)
    t2 = time.time()
    print('\ncalculated sym_diff(A,B) = %f in %f seconds\n'%(d, t2-t1))

    floor = 1e-20*max((A.eigenvector).max(), (B.eigenvector).max())
    plot(A, floor, ax1)
    plot(B, floor, ax2)
    print('''
    b.max()=%f, b2.max()=%f, floor=%e
    '''%(A.eigenvector.max(), B.eigenvector.max(), floor))
    if DEBUG:
        plt.show()
    else:
        File = open(args.file, 'w')
        fig.savefig(File, format='pdf')

if __name__ == "__main__":
    rv = main()
    if DEBUG:
        import matplotlib.pyplot as plt
        plt.show()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
