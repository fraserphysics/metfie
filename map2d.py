'''map2d.py: Makes plots in 2d of regions to which points map and
regions that map to points.

'''
import sys
import numpy as np
import matplotlib as mpl
def two_d(w,A, uniform=True):
    'return 2-d version of state vector suitable for plotting'
    if uniform:
        # Next line makes images have same color regardless of overlap
        w = (w>0)*A.h_step*A.g_step
    z = A.vec2z(np.ones((A.n_states,))) # Make mask for plots
    u = A.vec2z(w.reshape((A.n_states,)))
    m = u.max()
    w = u*z + m*z
    return w

def main(argv=None):
    '''For looking at images of points under A and A.T

    '''
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=float, default=(200.0),
        help='Max g')
    parser.add_argument('--d_g', type=float, default=1,
        help='element size')
    parser.add_argument('--d_h', type=float, default=1,
        help='element size')
    parser.add_argument('--iterations', type=int, default=1,
        help='Apply operator n times and scale d, d_h and d_g')
    parser.add_argument(
        '--points', type=float, nargs='*', default=(
                -0.9, 0.98,
                 -0.8, -0.4,
                    0, 0,
                   .9, 0.5,
                   0.0, 0.995),
                        help='Plot these points and their images')
    parser.add_argument('--backward', action='store_true',
        help='Use transpose of operator')
    parser.add_argument('--out', type=str, default=None,
        help="Write plot to this file")
    parser.add_argument('--archive', type=str, default=None,
                        help="Write self and vector to archive/name")
    args = parser.parse_args(argv)

    assert len(args.points)%2 == 0
    f_sources = ((args.points[2*i], args.points[2*i+1]) for i in 
                 range(int(len(args.points)/2)))
    
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
    from scipy.sparse.linalg import LinearOperator
    from first_c import LO_step
    #from first import LO_step

    A = LO_step( args.d*args.iterations**2, args.d_h, args.d_g)
    v = np.zeros(A.n_states)
    i_sources = []           # For tagging sources in plot
    for h_,g_ in f_sources:
        g = A.d * g_
        h_max = A.h_lim(g)
        h = h_max * h_
        G = A.g2G(g)
        H = A.h2H(h)
        i_sources.append((H,G))
        k = A.state_dict[(H,G)]  # get index of state vector
        v[k] = 1                 # Set component for (g, h) to 1.0
    if args.backward:
        op = A.rmatvec
        op_string = '$A^T$'
    else:
        op = A.matvec
        op_string = '$A$'
    if args.iterations > 1:
        uniform = False
        iterations_string = ', {0:d} iterations'.format(args.iterations)
        for i in range(1,args.iterations):
            v = op(v)
            v /= v.max()
    else:
        uniform = True
        iterations_string = ''
        v = op(v)
    if args.archive != None:
        A.archive(args.archive, v=v)
    suptitle = 'Points and images under %s%s'%(
        op_string, iterations_string)
    data = two_d(v,A, uniform)
    fig = plt.figure(figsize=(8,10))
    fig.suptitle(suptitle)
    ax = fig.add_subplot(1,1,1)
    for H,G in i_sources:
        t = data[H,G]
        data[H-2:H+3,G-1:G+2] = 0 # Make big markers for source points
        data[H,G] = t
    h_max = A.h_lim(-A.d)
    ax.imshow(
        data.T[-1::-1,:], interpolation="nearest",
        extent=[-h_max,h_max,-A.d,A.d], aspect='auto')
    ax.set_ylabel('$g$')
    ax.set_xlabel('$h$')
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
