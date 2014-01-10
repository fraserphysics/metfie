'''converge.py study convergence of numerical estimates of the
eigenfunction corresponding to the largest eigenvector of the first
order Markov integral operator.

After calculating errors, look at them with:

->python3 -c "from converge import plot; plot('result_converge')"

Default arguments have n_g and n_h take values 200 and 225.  With the
default arguments, script calls LO.power() 5 times and on watcher, the
run time is

real    2m57.357s
user    2m56.151s
sys     0m0.820s

'''
import sys
from first_c import LO
def main(argv=None):
    import argparse
    import numpy as np
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
    parser.add_argument('--n_h0', type=int, default=200, help=
'number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    parser.add_argument('--n_g_step', type=int, default=2, help=
                        'Number of different delta_gs')
    parser.add_argument('--n_h_step', type=int, default=2, help=
                        'Number of different delta_hs')
    parser.add_argument('--n_g_final', type=int, default=225)
    parser.add_argument('--n_h_final', type=int, default=225)
    parser.add_argument('--ref_frac', type=float, default=0.9, help=
                        'Fraction of finest resoultion used for reference')
    parser.add_argument('--out_file', type=str, default='result_converge',
        help="where to write result")
    args = parser.parse_args(argv)

    d_g_big = 2*args.u/args.n_g0
    d_g_small = 2*args.u/args.n_g_final
    dd_g = (d_g_big - d_g_small)/args.n_g_step
    d_g_ref = args.ref_frac*d_g_small

    h_lim = np.sqrt(48*args.u)
    d_h_big = 2*h_lim/args.n_h0
    d_h_small = 2*h_lim/args.n_h_final
    dd_h = (d_h_big - d_h_small)/args.n_h_step
    d_h_ref = args.ref_frac*d_h_small
    
    from scipy.sparse.linalg import LinearOperator
    from first import sym_diff
    import pickle
    
    tol = 5e-6
    maxiter = 150
    error = {}
    text = ''

    ref_LO = LO( args.u, args.dy, d_g_ref, d_h_ref)
    text += 'For ref, n_g=%d and n_h=%d\n'%(ref_LO.n_g,ref_LO.n_h)
    ref_LO.power(small=tol, n_iter=maxiter,verbose=True)
    for d_g in np.arange(d_g_small, d_g_big, dd_g):
        for d_h in np.arange(d_h_small, d_h_big, dd_h):
            key = 'd_g=%g d_h=%g'%(d_g, d_h)
            assert not key in error
            A = LO( args.u, args.dy, d_g, d_h)
            A.power(small=tol, n_iter=maxiter,verbose=True)
            d = sym_diff(A,ref_LO)
            error[key] = d
            text += 'n_g=%d and n_h=%d, error=%g\n'%(A.n_g,A.n_h,d)

    pickle.dump((args,text,error), open( args.out_file, "wb" ) )
    return 0
def read_ghz(file_name):
    '''Get arrays g (1-d array of n_g values), h (1-d array of n_h values)
    and z (2-d array of error values from pickled dict.

    '''
    import pickle
    import numpy as np
    args,text,error = pickle.load( open( file_name, "rb" ) )
    print('args=\n%s\ntext=%s\n'%(args,text))
    #error = pickle.load( open( file_name, "rb" ) )
    gs = set([])
    hs = set([])
    for key,d in error.items():
        d_g,d_h = (s.split('=')[-1] for s in key.split())
        gs.add((float(d_g),d_g))
        hs.add((float(d_h),d_h))
    gs = sorted(set(gs)) # Sort on floats and keep strings for keys
    hs = sorted(set(hs))
    z = np.empty((len(hs), len(gs)))
    for i in range(len(hs)):
        for j in range(len(gs)):
            key = 'd_g=%s d_h=%s'%(gs[j][1], hs[i][1])
            z[i,j] = error[key]
    h = [x[0] for x in hs]
    g = [x[0] for x in gs]
    return g,h,z

def plot(file_name='result_converge'):
    '''Function to plot result of main.  Invoke with
    python3 -c "from converge import plot; plot('result_converge')"
    '''
    import numpy as np
    import matplotlib as mpl
    g,h,z = read_ghz(file_name)
    for j in range(len(g)):
        for i in range(len(h)):
            print('d_h=%g, d_g=%g,  %f'%(h[i],g[j],z[i,j]))
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
    ax.set_xlabel(r'$d_g$')
    ax.set_ylabel(r'$d_h$')
    plt.show()
    
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
