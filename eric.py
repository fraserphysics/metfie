'''eric.py derived from explore.py.  For making plots of pie slices,
regions to which points map.

'''
import sys
import numpy as np
import matplotlib as mpl
def main(argv=None):
    '''For looking at sensitivity of time and results to u, dy, n_g, n_h.

    '''
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--u', type=float, default=(48.0),
        help='log fractional deviation')
    parser.add_argument(
        '--points', type=float, nargs='*', default=(
            #1.0, 0.0,
            -1.05, -0.9,
            0.6, -1.0,
            -.1, .5,
            -.2, 0,
            -0.15, 0.95,
            0.8, 0.95,
            0.95, -1.,
            #0.95,1.0,
        ),
                        help='Plot pie slices with these points at the apex')
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
              'xtick.labelsize': 15,
              'ytick.labelsize': 15}
    mpl.rcParams.update(params)
    if args.out != None:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use

    h_max = (48*args.u)**.5
    h =  np.linspace(-h_max, h_max, 1000)
    boundary = lambda x: args.u -x*x/24
    g = boundary(h)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(g,h,'b-')
    ax.plot(np.ones(h.shape)*(-args.u),h,'b-')
    
    i_sources = []         # For tagging sources in plot
    for g_,h_ in f_sources:
        g = args.u * g_
        h_max = np.sqrt(24*(args.u-g))
        h = h_max * h_
        H = np.linspace(h,h_max,1000)
        G = np.ones(H.shape)*g
        ax.plot(G,H,'r-')
        G = g + H - h
        ax.plot(G,H,'r-')
        h_0 = h+12
        H = np.linspace(h,h_0,20)
        G = np.linspace(g,g-h_0+6,20)
        ax.plot(G,H,'g--',lw=2)
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
