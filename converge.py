'''converge.py study convergence of numerical estimates of the
eigenfunction corresponding to the largest eigenvector of the first
order Markov integral operator.

After calculating errors, look at them with converge_plot.py

Default arguments have n_g and n_h take values 200 and 225.  With the
default arguments, script calls LO.power() 5 times and on watcher, the
run time is

real    2m57.357s
user    2m56.151s
sys     0m0.820s

'''
import sys
from first_c import LO_step as LO
def main(argv=None):
    import argparse
    import numpy as np
    import time
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    t_start = time.time()
    parser = argparse.ArgumentParser(description=
    '''Calculate eigenvalues and eigenfuction errors for ranges of resolution
in g and h''')
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

    d_g_small = 2*args.u/args.n_g_final
    if args.n_g0 == args.n_g_final or args.n_g_step == 1:
        d_g_big = 1.1*d_g_small 
        dd_g = 0.2*d_g_small
    else:
        d_g_big = 2*args.u/args.n_g0
        dd_g = (d_g_big - d_g_small)/args.n_g_step
    d_g_ref = args.ref_frac*d_g_small

    h_lim = np.sqrt(48*args.u)
    d_h_big = 2*h_lim/args.n_h0
    d_h_small = 2*h_lim/args.n_h_final
    dd_h = (d_h_big - d_h_small)/args.n_h_step
    d_h_ref = args.ref_frac*d_h_small
    
    from first import sym_diff
    import pickle
    
    tol = 5e-6
    maxiter = 1000
    error = {}
    text = ''

    ref_LO = LO( args.u, args.dy, d_g_ref, d_h_ref)
    ref_LO.power(small=tol, n_iter=maxiter,verbose=True)
    text += 'For ref, n_g=%d, n_h=%d and e_val=%9.3e\n'%(
        ref_LO.n_g,ref_LO.n_h, ref_LO.eigenvalue)
    for d_g in np.arange(d_g_small, d_g_big, dd_g):
        for d_h in np.arange(d_h_small, d_h_big, dd_h):
            key = 'd_g=%g d_h=%g'%(d_g, d_h)
            assert not key in error
            A = LO( args.u, args.dy, d_g, d_h)
            A.power(small=tol, n_iter=maxiter,verbose=True)
            d = sym_diff(A,ref_LO)
            error[key] = (d,A.eigenvalue)
            text += 'n_g=%5d and n_h=%4d, error=%6.4f, e_val=%9.3e\n'%(
                A.n_g,A.n_h,d, A.eigenvalue)

    elapsed = time.time() - t_start
    pickle.dump((args,text,error,elapsed), open( args.out_file, "wb" ) )
    return 0
def read_study(file_name, verbose=1):
    '''Get data from pickled dict.
    '''
    import pickle
    import numpy as np
    from datetime import timedelta
    
    args,text,dict_,elapsed = pickle.load( open( file_name, "rb" ) )
    if verbose>0:
        print('time=%s'%(timedelta(seconds=elapsed)))
        for key,value in vars(args).items():
            print('%s=%s'%(key,value))
    if verbose>1:
        print('%s'%(text,))
    gs = set([])
    hs = set([])
    for key in dict_:
        d_g,d_h = (s.split('=')[-1] for s in key.split())
        gs.add((float(d_g),d_g))
        hs.add((float(d_h),d_h))
    gs = sorted(set(gs)) # Sort on floats and keep strings for keys
    hs = sorted(set(hs))
    error = np.empty((len(hs), len(gs)))
    eigenvalue = np.empty((len(hs), len(gs)))
    for i in range(len(hs)):
        for j in range(len(gs)):
            key = 'd_g=%s d_h=%s'%(gs[j][1], hs[i][1])
            error[i,j], eigenvalue[i,j] = dict_[key]
    h = [x[0] for x in hs]
    g = [x[0] for x in gs]
    return g,h,error,eigenvalue

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
