'''test.py Make sure that first.py and first_c.pyx make the same LO

'''
import sys
import first
import first_c
def main(argv=None):
    import argparse
    import numpy as np
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
    parser.add_argument('--n_h', type=int, default=200, help=
'number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    args = parser.parse_args(argv)

    d_g = 2*args.u/args.n_g

    h_lim = np.sqrt(48*args.u)
    d_h = 2*h_lim/args.n_h
    
    tol = 5e-6
    maxiter = 150
    error = {}
    text = ''

    LO_cython = first_c.LO_step( args.u, args.dy, d_g, d_h)
    print('cython: n_states=%d, n_pairs=%d'%(
        LO_cython.n_states,LO_cython.n_pairs))
    LO_python = first.LO_step( args.u, args.dy, d_g, d_h)
    print('python: n_states=%d, n_pairs=%d'%(
        LO_python.n_states,LO_python.n_pairs))
    return 0
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
