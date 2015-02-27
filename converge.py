'''converge.py study convergence of numerical estimates of the
conditional distribution given an initial point.  Either d_h or d_g or
both can be varied.  To study dependence on the number of iterations,
use conditional.py

'''
import sys
import numpy as np
import matplotlib as mpl
from first_c import LO_step
from first import Archive
def main(argv=None):
    '''For looking at the probability distribution over the image of a
    point under A
    '''
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=float, default=100,
        help='Max g')
    parser.add_argument('--d_h', nargs='*', type=float, default=(4,),
        help='list of element sizes')
    parser.add_argument('--d_g', nargs='*', type=float, default=(4,),
        help='list of element sizes')
    parser.add_argument('--iterations', type=int, default=2,
        help='Apply operator n times and scale d, d_h and d_g')
    parser.add_argument(
        '--point', type=float, nargs=2, default=(0.0, 0.0),
        help='Analyze the image of this point (h and g as fractions of maxima')
    parser.add_argument('--out', type=str, default=None,
        help="Write plot to this file")
    args = parser.parse_args(argv)

    params = {'axes.labelsize': 18,     # Plotting parameters for latex
              'text.fontsize': 15,
              'legend.fontsize': 15,
              'text.usetex': True,
              'font.family':'serif',
              'font.serif':'Computer Modern Roman',
              'xtick.labelsize': 15,
              'ytick.labelsize': 15}
    mpl.rcParams.update(params)
    if args.out != None:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use

    archive = Archive(LO_step)
    # Initialize operator
    d = args.d*args.iterations**2
    d_h_list = args.d_h
    d_g_list = args.d_g
    n_d_h = len(d_h_list)
    n_d_g = len(d_g_list)
    
    from level import ellipse
    from conditional import fit
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1, aspect='equal')
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$g$')
    for i,d_h in enumerate(d_h_list):
        args.d_h = d_h
        for j,d_g in enumerate(d_g_list):
            args.d_g = d_g
            print('working on {0}, {1}'.format(d_h, d_g))
            mu, sigma, vals, theta_sigma, A = fit(
                args, archive, args.iterations)
            h,g = ellipse(sigma)
            ax.plot( h+mu[0], g+mu[1], label=
                     r'$d_h={0:.2f}, d_g={1:.2f}$'.format(d_h,d_g))
    ax.legend(loc='upper left')
    
    if args.out == None:
        plt.show()
    else:
        fig.savefig( open(args.out, 'wb'), format='pdf')
    return 0

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
