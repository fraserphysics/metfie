import sys
                                  # Clock time, 200x200 on watcher
from first_c import LO_step as LO #   4.3 sec
#from first import LO_step as LO  # 689.3 sec
def main(argv=None):
    import argparse
    import numpy as np
    import time
    import resource
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=
    '''Initialize LO_step instance, calculate eigenfuction and store in
archive directory''')
    parser.add_argument('--u', type=float, default=(48.0),
        help='log fractional deviation')
    parser.add_argument('--dy', type=float, default=0.2,
        help='y_1-y_0 = log(x_1/x_0)')
    parser.add_argument('--n_g', type=int, default=400,
                       help='number of integration elements in value')
    parser.add_argument('--n_h', type=int, default=400, help=
'number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    parser.add_argument('--max_iter', type=int, default=2000,
                       help='Maximum number power iterations')
    parser.add_argument('--tol', type=float, default=(1.0e-6),
                       help='Stopping criterion for power')
    parser.add_argument('--out_file', type=str, default=None,
        help="Name of result file.  Default derived from other args.")
    parser.add_argument('--out_dir', type=str, default='archive',
        help="Directory for result")
    args = parser.parse_args(argv)
    
    if args.out_file == None:
        args.out_file = '%d_g_%d_h_%d_y'%(args.n_g, args.n_h, int(args.dy/1e-5))
    d_g = 2*args.u/args.n_g
    h_lim = np.sqrt(48*args.u)
    d_h = 2*h_lim/args.n_h
    
    t_start = time.time()
    op = LO( args.u, args.dy, d_g, d_h )
    op.power(small=args.tol, n_iter=args.max_iter, verbose=True)
    t_stop = time.time()
    more = {
        'eigenvalue':float(op.eigenvalue),
        'iterations':op.iterations,
        'time':(t_stop-t_start),
        'memory':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    op.archive(args.out_file, more, args.out_dir)
   
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:

