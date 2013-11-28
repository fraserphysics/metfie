'''converge.py study convergence of numerical estimates of the
eigenfunction corresponding to the largest eigenvector of the first
order Markov integral operator.
'''
import sys
from first_c import LO
def main(argv=None):
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
    parser.add_argument('--n_g0', type=int, default=200,
                       help='number of integration elements in value')
    parser.add_argument('--n_h0', type=int, default=200,
help='number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    parser.add_argument('--n_g_step', type=int, default=25)
    parser.add_argument('--n_h_step', type=int, default=25)
    parser.add_argument('--n_g_final', type=int, default=225)
    parser.add_argument('--n_h_final', type=int, default=225)
    parser.add_argument('--out_file', type=str, default='result_converge',
        help="where to write result")
    args = parser.parse_args(argv)
    
    from scipy.sparse.linalg import LinearOperator
    import numpy as np
    from first import sym_diff
    import pickle
    
    tol = 5e-6
    maxiter = 150
    error = {}

    g = np.arange(args.n_g_final, args.n_g0-1, -args.n_g_step)
    h = np.arange(args.n_h_final, args.n_h0-1, -args.n_h_step)
    g_ref = args.n_g_final + args.n_g_step
    h_ref = args.n_h_final + args.n_h_step
    ref_LO = LO( args.u, args.dy, g_ref, h_ref)
    ref_LO.power(small=tol, n_iter=maxiter,verbose=True)
    for n_g in g:
        for n_h in h:
            key = 'n_g=%d n_h=%d'%(n_g, n_h)
            assert not key in error
            A = LO( args.u, args.dy, n_g, n_h)
            A.power(small=tol, n_iter=maxiter,verbose=True)
            d = sym_diff(A,ref_LO)
            error[key] = d
            print('%s error=%f'%(key, d))

    pickle.dump(error, open( args.out_file, "wb" ) )
    return 0
def plot(file_name='result_converge'):
    '''Function to plot result of main.  Invoke with
    python3 -c "from converge import plot; plot('result_converge')"
    '''
    import pickle
    import numpy as np
    error = pickle.load( open( file_name, "rb" ) )
    gs = []
    hs = []
    for key,d in error.items():
        n_g,n_h = (int(s.split('=')[-1]) for s in key.split())
        gs.append(n_g)
        hs.append(n_h)
        print('%s error=%f'%(key, d))
    g = sorted(set(gs))
    h = sorted(set(hs))
    G,H = np.meshgrid(g, h)
    z = np.empty(G.shape)
    assert G.shape == (len(h), len(g))
    for i in range(len(h)):
        for j in range(len(g)):
            try:
                key = 'n_g=%d n_h=%d'%(g[j], h[i])
            except:
                print('j=%d i=%d'%(j,i))
                key = 'n_g=%d n_h=%d'%(g[j], h[i])
            z[i,j] = error[key]

    import matplotlib as mpl
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
    ax.set_xlabel(r'$n_g$')
    ax.set_ylabel(r'$n_h$')
    plt.show()
    
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
