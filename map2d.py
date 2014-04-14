'''map2d.py derived from explore.py.  For making plots in 2d of
regions to which points map and regions that map to points.

'''
import sys
import numpy as np
import matplotlib as mpl
DEBUG = False
def main(argv=None):
    '''For looking at sensitivity of time and results to u, dy, n_g, n_h.

    '''
    import argparse
    global DEBUG
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--u', type=float, default=(48.0),
        help='log fractional deviation')
    parser.add_argument('--dy', type=float, default=0.3,
        help='y_1-y_0 = log(x_1/x_0)')
    parser.add_argument('--n_g', type=int, default=500,
        help='number of integration elements in value')
    parser.add_argument('--n_h', type=int, default=240,
        help='number of integration elements in slope.')
    parser.add_argument('--line', type=float, nargs=2,
        default=None,
        help='number of integration elements in slope.')
    parser.add_argument(
        '--points', type=float, nargs='*', default=(
                -0.98, 0.9,
                 -0.4, -0.8,
                    0, 0,
                   .5, 0.9,
                0.995, 0.0),
                        help='Plot these points and their images')
    parser.add_argument('--backward', action='store_true',
        help='Plot pre-images of points.')
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
    from mpl_toolkits.mplot3d import Axes3D  # Mysteriously necessary
                                             #for "projection='3d'".
    from first_c import LO_step
    #from first import LO_step

    g_step = float(2*args.u)/args.n_g
    h_max = (24*args.u)**.5
    h_step = float(2*h_max)/args.n_h
    A = LO_step( args.u, args.dy, g_step, h_step)
    v = np.zeros(A.n_states)
    i_sources = []         # For tagging sources in plot
    for g_,h_ in f_sources:
        g = A.g_max * g_
        h_max = np.sqrt(24*(A.g_max-g))
        h = h_max * h_
        G = int(np.floor((g-A.g_min)/A.g_step + .5))
        H = int(np.floor(h/A.h_step + .5))
        i_sources.append((G, H+A.n_h/2))
        k = A.state_dict[(G,H)]  # get index of state vector
        v[k] = 1                 # Set component for (g, h) to 1.0
    if args.backward:
        image = A.rmatvec(v)
        title = '$A^T(v)$'
        suptitle = 'Points and images under $A^T$ for $dy$=%.2f'%A.dy
    else:
        image = A.matvec(v)
        title='$A(v)$'
        suptitle = 'Points and images under $A$ for $dy$=%.2f'%A.dy
    z = A.vec2z(np.ones((A.n_states,))) # Make mask for plots
    def two_d(w):
        'return 2-d version of state vector suitable for plotting'
        # Next line makes images have same color regardless of overlap 
        w = np.minimum(A.h_step*A.g_step, w)
        u = A.vec2z(w.reshape((A.n_states,)))
        m = u.max()
        w = u*z + m*z
        return w
    fig = plt.figure(figsize=(16,8))
    fig.suptitle(suptitle)
    h_max = (48*A.g_max)**.5 # FixMe?
    ax = fig.add_subplot(1,1,1)
    data = two_d(image)
    if args.line != None:
        m,b = args.line
        for H in range(A.n_h):
            h = A.h_step * (H-A.n_h/2)
            g = (h-b)/m
            h_lim = A.h_lim(g)
            if h >= h_lim or h <= -h_lim:
                continue
            G = int((g+A.g_max)/A.g_step)
            data[G,H] = 0
    for G,H in i_sources:
        t = data[G,H]
        data[G-1:G+2,H-2:H+3] = 0 # Make big markers for source points
        data[G,H] = t
    ax.imshow(
        data.T[-1::-1,:], interpolation="nearest",
        extent=[-A.g_max,A.g_max,-h_max,h_max], aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('$g$')
    ax.set_ylabel('$h$')
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
