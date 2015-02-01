'''map2d.py: Makes plots in 2d of regions to which points map and
regions that map to points.

'''
import sys
import numpy as np
import matplotlib as mpl
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
    parser.add_argument('--symmetry', action='store_true',
        help='Same result as forward, but from conjugate of backwards')
    parser.add_argument('--out', type=str, default=None,
        help="Write result to this file")
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

    A = LO_step( args.d, args.d_h, args.d_g)
    v = np.zeros(A.n_states)
    i_sources = []         # For tagging sources in plot
    for h_,g_ in f_sources:
        g = A.d * g_
        h_max = np.sqrt(24*(A.d-g))
        h = h_max * h_
        G = int(np.floor((g+A.d)/A.g_step + .5))
        H = int(np.floor(h/A.h_step + .5))
        if not (H,G) in A.state_dict:
            print('skipping ({0}, {1})'.format(G,H))
            continue
        i_sources.append((H+A.n_h/2,G))
        k = A.state_dict[(H,G)]  # get index of state vector
        v[k] = 1                 # Set component for (g, h) to 1.0
    if args.backward:
        image = A.rmatvec(v)
        suptitle = 'Points and images under $A$ for $d$=%.2f'%A.d
    elif args.symmetry:
        image = A.symmetry(A.matvec(A.symmetry(v)))
        suptitle = 'Symmetry ops: $A$ from $A^T$ $d$=%.2f'%A.d
    else:
        image = A.matvec(v)
        suptitle = 'Points and images under $A^T$ for $d$=%.2f'%A.d
    z = A.vec2z(np.ones((A.n_states,))) # Make mask for plots
    def two_d(w):
        'return 2-d version of state vector suitable for plotting'
        # Next line makes images have same color regardless of overlap 
        w = np.minimum(A.h_step*A.g_step, w)
        u = A.vec2z(w.reshape((A.n_states,)))
        m = u.max()
        w = u*z + m*z
        return w
    fig = plt.figure(figsize=(8,10))
    fig.suptitle(suptitle)
    h_max = np.sqrt(24*(2*A.d))
    ax = fig.add_subplot(1,1,1)
    data = two_d(image)
    for H,H in i_sources:
        t = data[H,G]
        data[H-2:H+3,G-1:G+2] = 0 # Make big markers for source points
        data[H,G] = t
    ax.imshow(
        data.T[-1::-1,:], interpolation="nearest",
        extent=[-A.d,A.d,-h_max,h_max], aspect='auto')
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
