'''eric.py derived from explore.py.  For making plots of pie slices,
regions to which points map.

'''
import sys
import numpy as np
import matplotlib as mpl
def axplot(*args):
    ax,x,y = args[0:3]
    ax.plot(y,x,*args[3:])
def sym(clipped=False):
    import sympy
    x, y, x0, y0, a, d = sympy.symbols('x y x0 y0 a d')
    #y1 = sympy.Max(-d, y0 + x0 - 1/(4*a))
    if clipped:
        y1 = d
    else:
        y1 = y0 + x0 - 1/(4*a)
    x1 = x0 -1/(2*a)
    boundary = d - a*x*x - y
    slope = y1 + x - x1 - y
    both = boundary - slope
    x2 = sympy.solve(both,x)[0]
    y2 = x2 - x1 + y1
    y3 = y1
    x3 = sympy.solve(boundary.subs(y, y1),x)[1]
    print('''
    x3=%s
    '''%(x3,))
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
    f_sources = np.array(tuple((args.points[2*i], args.points[2*i+1]) for i in 
                 range(int(len(args.points)/2))))
    
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
    axplot(ax,g,h,'b-')
    axplot(ax,np.ones(h.shape)*(-args.u),h,'b-')

    g = args.u*f_sources[:,0]
    h_max = np.sqrt(24*(args.u-g))
    h = f_sources[:,1]*h_max
    h_0 = h + 12
    g_0 = g - h_0 + 6
    g_1 = np.maximum(-args.u, g)
    h_1 = g_1 - g_0 - 6
    for i in range(len(g)):
        H = np.linspace(h[i],h_max[i],1000)
        G = np.ones(H.shape)*g[i]
        axplot(ax,G,H,'r-')
        G = g[i] + H - h[i]
        axplot(ax,G,H,'r-')
        H = np.linspace(h[i],h_0[i],20)
        G = np.linspace(g[i],g_0[i],20)
        axplot(ax,G,H,'g--')
        axplot(ax,g_1[i],h_1[i],'rx')
        axplot(ax,g_0[i],h_0[i],'gx')
    ax.set_xlabel('$h$')
    ax.set_ylabel('$g$')
    if args.out == None:
        plt.show()
    else:
        fig.savefig( open(args.out, 'wb'), format='pdf')
    return 0

if __name__ == "__main__":
    rv = main()
    #rv = sym()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
