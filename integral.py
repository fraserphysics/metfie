"""integral.py Makes plots illustrating the integral equation to solve
to get the eigenfunction of the adjacency function.

"""
DEBUG = False
plot_dict = {} # Keys are those keys in args.__dict__ that ask for
               # plots.  Values are the functions that make those
               # plots.
import sys
import matplotlib as mpl
import numpy as np
def main(argv=None):
    import argparse
    global DEBUG

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Make some plots for the integral equation')
    parser.add_argument('--Dy', type=float, default=0.01,
                       help='Range of y')
    parser.add_argument('--dy', type=float, default=0.0005,
                       help='Resolution of y')
    parser.add_argument('--Dg', type=float, default=0.005,
                       help='Range of g')
    parser.add_argument('--dg', type=float, default=0.0002,
                       help='Resolution of g')
    parser.add_argument('--y_0', type=float, default=0.0,
                       help='First y')
    parser.add_argument('--y_1', type=float, default=0.005,
                       help='Second y')
    parser.add_argument('--g0', type=float, default=-0.001,
                       help='First g')
    parser.add_argument('--g1', type=float, default=-0.0013,
                       help='Second g')
    parser.add_argument('--debug', action='store_true')
    # Plot requests
    parser.add_argument(
        '--taylor', type=argparse.FileType('wb'),
        help='2nd order approximation to line in log-log coordinates')
    parser.add_argument(
        '--bounds1', type=argparse.FileType('wb'),
        help="bounds on g at y_1 given g and g' at y_0")
    parser.add_argument(
        '--bounds2', type=argparse.FileType('wb'),
        help="bounds on g' at y_1 given g at y_0 and g at y_1")
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
    if args.debug:
        DEBUG = True
        mpl.rcParams['text.usetex'] = False
    else:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use
    lines = Lines(args)

    # Make requested plots
    for key in args.__dict__:
        if key not in plot_dict:
            continue
        if args.__dict__[key] == None:
            continue
        fig = plot_dict[key](lines, plt)
        if not DEBUG:
            fig.savefig(args.__dict__[key], format='pdf')
    return 0

class Lines:
    ''' Provides data arrays for plotting lines
    '''
    def __init__(self, args):
        self.Dy = args.Dy
        self.dy = args.dy
        self.Dg = args.Dg
        self.dg = args.dg
        self.y_0 = args.y_0
        self.y_1 = args.y_1
        self.g0 = args.g0
        self.g1 = args.g1
        shift = (self.y_1-self.y_0)/2
        self.y = np.arange(shift-self.Dy, shift+self.Dy, self.dy)
        self.x = np.exp(self.y)
        self.n_y = len(self.y)
        upper = np.ones(self.n_y)*self.Dg
        self.upper = (upper, 'r')
        self.lower = (-upper, 'r')
        self.center = (upper*0, 'g')
    def tangent_xf_1_1(self):
        ''' In x,f (not log-log) coordinates make line tanget to nominal at 1.0
        '''
        f = 4-3*self.x
        self.tan_00 = (3*self.y + np.log(f), 'm')
        return
    def quad_xf_1_1(self):
        ''' Quadratic approximation to tan_00
        '''
        self.quad_00 = (-6*self.y**2, 'c')
        return
    def bounds_g_dg(self, g=-.001, dg=-.2):
        ''' Given the function and its derivative at y_0, illustrate the constraints on the values at y_1.
        '''
        root = lambda g, a, y: y - np.sqrt((a-g)/6)
        a = self.upper[0][0]
        b = root(g, a, self.y_0)
        self.left = (a -6*(self.y - b)**2, 'c')
        self.right = (a -6*(self.y + b)**2, 'm')
        b = self.y_0 + dg/12
        a = g + dg**2/24
        self.tan_g_dg = (a -6*(self.y - b)**2, 'g')
        self.g0 = g
        self.dg0 = dg
        return
    def bounds_g0_g1(self, g0, g1):
        ''' Given g at y_0 and g at y_1, illustrate the constraints on g'
        at y_1.  In particular, calculate self.right and self.g0_g1
        '''
        # calculate self.right
        root = lambda g, a, y: y + np.sqrt((a-g)/6)
        a = self.upper[0][0]
        b = root(g1, a, self.y_1)
        self.right = (a -6*(self.y - b)**2, 'g') # FixMe: Do not repeat yourself
        # calculate self.g0_g1
        y0 = self.y_0
        y1 = self.y_1
        b = (g0-g1)/(12*(y0-y1)) + (y0+y1)/2
        a = g0 + 6*(y0-b)**2
        self.g0_g1 = (a -6*(self.y - b)**2, 'b')
        return
    def set_g1(self, g1):
        self.g1 = g1
        return
    def set_g0(self, g1):
        self.g0 = g0
        return
def tangent0(lines, plt):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    lines.tangent_xf_1_1()
    lines.quad_xf_1_1()
    for y, color in (lines.upper, lines.center, lines.lower):
        ax.plot(lines.y, y, color=color)
    for (y,color), t in ((lines.tan_00, r'$3y+\log(4-3e^y)$'),
                    (lines.quad_00, r'$-6y^2$')):
        ax.plot(lines.y, y, color=color, label=t)
    ax.legend()
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$g$')
    fig.subplots_adjust(left=0.2)
    return fig
plot_dict['taylor'] = tangent0
def bounds1(lines, plt):
    '''Illustrate the range of g values at y_1 allowed given g and h at y_0.
    '''
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    lines.bounds_g_dg()
    ax.plot((lines.y_0,), (lines.g0,), 'ok')
    ax.plot((lines.y_1,), (lines.g1,), 'ok')
    for y,color in (lines.upper, lines.lower,
              lines.tan_g_dg):
        ax.plot(lines.y, y, color=color)
    for (y,color), t in (
            (lines.left, r'$\bar U_g(g_0,y_0)$'),
            (lines.right, r'$U_g(g_0,y_0)$'),
            (lines.tan_g_dg, r'$L_g(g_0,h_0,y_0)$')):
        ax.plot(lines.y, y, color=color, label=t)
    U = lines.upper[0]
    L = lines.lower[0]
    for y in (lines.y_0, lines.y_1):
        ax.plot((y,y), (L,U), color='k')
    ax.set_ylim(-.006, .006)
    #ax.set_xticks((lines.y_0, lines.y_1))
    ax.legend()
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$g$')
    fig.subplots_adjust(left=0.2)
    return fig
plot_dict['bounds1'] = bounds1

def bounds2(lines, plt):
    '''Illustrate the range of h values at y_1 that are allowed given g at y_0
    and g at y_1.
    '''
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    lines.bounds_g0_g1(lines.g0, lines.g1)
    ax.plot((lines.y_0,), (lines.g0,), 'ok')
    ax.plot((lines.y_1,), (lines.g1,), 'ok')
    for y,color in (lines.upper, lines.lower):
        ax.plot(lines.y, y, color=color)
    for (y,color),t in ((lines.right, r'$U_h(g_1)$'),
                        (lines.g0_g1, r'$L_h(g_1, g_0)$')):
        ax.plot(lines.y, y, color=color, label=t)
    U = lines.upper[0]
    L = lines.lower[0]
    for y in (lines.y_0, lines.y_1):
        ax.plot((y,y), (L,U), color='k')
    ax.set_ylim(-.006, .006)
    ax.set_xticks((lines.y_0, lines.y_1))
    ax.legend()
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$g$')
    fig.subplots_adjust(left=0.2)
    return fig
plot_dict['bounds2'] = bounds2

if __name__ == "__main__":
    rv = main()
    if DEBUG:
        import matplotlib.pyplot as plt
        plt.show()
    sys.exit(rv)

#---------------
# Local Variables:
# eval: (python-mode)
# End:
