import sys
                                  # Clock time, default parameters on watcher
from first_c import LO_step as LO #   4.3 sec
#from first import LO_step as LO  # 689.3 sec
def main(argv=None):
    import argparse
    import numpy as np
    import time
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    t_start = time.time()
    parser = argparse.ArgumentParser(description=
    '''Initialize LO_step instance, calculate eigenfuction and store in
archive directory''')
    parser.add_argument('--u', type=float, default=(2.0e-5),
                       help='log fractional deviation')
    parser.add_argument('--dy', type=float, default=3.2e-4,
                       help='y_1-y_0 = log(x_1/x_0)')
    parser.add_argument('--n_g', type=int, default=200,
                       help='number of integration elements in value')
    parser.add_argument('--n_h', type=int, default=200, help=
'number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    parser.add_argument('--out_file', type=str, default=None,
        help="Name of result file.  Default derived from other args.")
    args = parser.parse_args(argv)
    
    if args.out_file == None:
        args.out_file = '%d_g_%d_h_%d_y'%(args.n_g, args.n_h, int(args.dy/1e-5))
    d_g = 2*args.u/args.n_g
    h_lim = np.sqrt(48*args.u)
    d_h = 2*h_lim/args.n_h
    
    tol = 5e-6
    maxiter = 1000

    op = LO( args.u, args.dy, d_g, d_h )
    op.power(small=tol, n_iter=maxiter,verbose=True)
    op.archive(args.out_file)
    # FixMe: Modify archive method to accept comments and pass text
    # like converge.py
   
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:

