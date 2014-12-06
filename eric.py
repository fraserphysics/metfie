'''eric.py derived from explore.py.  For making plots of pie slices,
regions to which points map.

'''
import sys
import numpy as np
import matplotlib as mpl
import sympy
from sympy import collect
def lineplot(*args):
    ax,za,zb = args[0:3]
    ax.plot((za[0],zb[0]),(za[1],zb[1]),*args[3:])
def symplot(*args,**kwargs):
    ax,za = args[0:2]
    ax.plot(za[0],za[1],*args[2:],**kwargs)
class sym:
    '''Symbolic calculations of characteristics of images of initial points
    z0=(x0,y0)
    '''
    def __init__(self, clipped=False):
        x, y, x0, y0, a, d = sympy.symbols('x y x0 y0 a d')
        #y1 = sympy.Max(-d, y0 + x0 - 1/(4*a))
        if clipped:
            y1 = d
        else:
            y1 = y0 + x0 - 1/(4*a)
        x1 = x0 - 1/(2*a)
        boundary = d - a*x*x - y
        slope = y1 + x - x1 - y
        both = boundary - slope
        x2 = sympy.solve(both,x)[0]
        y2 = x2 - x1 + y1
        y3 = y1
        x3 = sympy.solve(boundary.subs(y, y1),x)[1]
        def integrate(f):
            ia = sympy.integrate(
                sympy.integrate(f,(y, y1, y1+x-x1)),
                (x, x1, x2))
            ib = sympy.integrate(
                sympy.integrate(f,(y, y1, d - a*x*x)),
                (x, x2, x3))
            i = ia + ib
            return i.subs(a, sympy.Rational(1,24))
        i0 = integrate(1)               # Area = integral of pie slice
        E = lambda f:(integrate(f)/i0)  # Expected value wrt Lebesgue
        #sigma = lambda f,g:collect(E(f*g) - E(f)*E(g),x0)
        sigma = lambda f,g:E(f*g) - E(f)*E(g)
        for s in 'x0 y0 a d x1 y1 x2 y2 x3 y3'.split():
            setattr(self, s, locals()[s])
        self.Ex = E(x)
        self.Ey = E(y)
        self.Sigmaxx = sigma(x,x)
        self.Sigmaxy = sigma(x,y)
        self.Sigmayy = sigma(y,y)
        return
    def Sigma(self, *args):
        '''Call with float args for (x0 y0 a d).  Returns covariance of image of
        z0=(x0,y0) as 2x2 array of floats.
        '''
        # paris[0] = (self.x0 (a sympy variable), x0 (a float argument))
        pairs = zip((getattr(self, s) for s in 'x0 y0 a d'.split()),args)
        xx = self.Sigmaxx.subs(pairs).evalf()
        xy = self.Sigmaxy.subs(pairs).evalf()
        yy = self.Sigmayy.subs(pairs).evalf()
        return np.array([[xx,xy],[xy,yy]])
    def z1(self, *args):
        pairs = zip((getattr(self, s) for s in 'x0 y0 a d'.split()),args)
        x = self.x1.subs(pairs).evalf()
        y = self.y1.subs(pairs).evalf()
        return (x,y)
    def z2(self, *args):
        pairs = zip((getattr(self, s) for s in 'x0 y0 a d'.split()),args)
        x = self.x2.subs(pairs).evalf()
        y = self.y2.subs(pairs).evalf()
        return (x,y)
    def z3(self, *args):
        pairs = zip((getattr(self, s) for s in 'x0 y0 a d'.split()),args)
        x = self.x3.subs(pairs).evalf()
        y = self.y3.subs(pairs).evalf()
        return (x,y)
    def Ez(self, *args):
        pairs = zip((getattr(self, s) for s in 'x0 y0 a d'.split()),args)
        x = self.Ex.subs(pairs).evalf()
        y = self.Ey.subs(pairs).evalf()
        return (x,y)
def plot(args, s):
    '''Make a single figure to show the following for several starting points
    z_0:

    boundary  Parabolic boundary of the allowed region in G x H
    pie-slice Image of z_0
    z_1       Vertex of pie slice
    z_2       Intersection of diagonal edge of pie slice and boundary 
    z_3       Intersection of flat edge of pie slice and boundary
    mu        Mean of pie slice
    ellipse   Level set of quadratic z^t \Sigma^{-1} z
    '''
    assert len(args.points)%2 == 0
    f_sources = np.array(tuple((args.points[2*i], args.points[2*i+1]) for i in 
                 range(int(len(args.points)/2))))
    
    params = {'axes.labelsize': 18,     # Plotting parameters for latex
              'text.fontsize': 15,
              'legend.fontsize': 15,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15}
    mpl.rcParams.update(params)
    if args.out == 'show':
        mpl.use('Qt4Agg')
    else:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use
    from level import ellipse

    h_max = (48*args.u)**.5
    h =  np.linspace(-h_max, h_max, 1000)
    boundary = lambda x: args.u -x*x/24
    g = boundary(h)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    # Plot the boundary
    ax.plot(h, g, 'b-')
    ax.plot(h, np.ones(h.shape)*(-args.u),'b-')

    g = args.u*f_sources[:,0]
    h_max = np.sqrt(24*(args.u-g))
    h = f_sources[:,1]*h_max
    h_0 = h + 12
    g_0 = g - h_0 + 6
    for i in range(len(g)):
        v = h_0[i], g_0[i], 1.0/24, args.u
        z0 = (h_0[i],g_0[i])
        z1 = s.z1(*v)
        z2 = s.z2(*v)
        z3 = s.z3(*v)
        Ez = s.Ez(*v)
        Sigma = s.Sigma(*v)
        lineplot(ax,z1,z3,'r-')        # horizontal line
        lineplot(ax,z1,z2,'r-')        # diagonal line
        lineplot(ax,z0,z1,'g--')       # line from z0 to z1
        x,y = ellipse(Sigma*2)
        if i == 0:
            symplot(ax,z0,'gx',label='z0')
            symplot(ax,z1,'r.',label='z1')
            symplot(ax,Ez,'bo',label=r'$\mu$')
            ax.plot(x+Ez[0], y+Ez[1], 'c-',label=r'$2\Sigma$')
        else:
            symplot(ax,z1,'r.')
            symplot(ax,z0,'gx')
            symplot(ax,Ez,'bo')
            ax.plot(x+Ez[0], y+Ez[1], 'c-')
    ax.set_xlabel('$h$')
    ax.set_ylabel('$g$')
    ax.legend()
    if args.out == 'show':
        plt.show()
    else:
        fig.savefig( open(args.out, 'wb'), format='pdf')
    return 0

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
            #-1.05, -0.9,
            -0.99, -0.99,
            0.6, -1.0,
            -.1, .5,
            #-.2, 0,
            #-0.15, 0.95,
            #0.8, 0.95,
            0.95, -1.,
            #0.95,1.0,
        ),
        help='Plot pie slices with apex at (g,h) as fractions')
    parser.add_argument('--backward', action='store_true',
        help='Plot pre-images of points.')
    parser.add_argument('--out', type=str, default=None,
        help="Write result to this file")
    parser.add_argument('--latex', type=str, default=None,
        help="Write latex results to this file")
    parser.add_argument('--integrate', action='store_true')
    args = parser.parse_args(argv)
    
    s = sym()
    if args.latex != None:
        f = open(args.latex,'w')
        f.write('\\newcommand{\Ex}{%s}\n'%(sympy.latex(s.Ex.simplify())),)
        f.write('\\newcommand{\Ey}{%s}\n'%(sympy.latex(s.Ey.simplify())),)
    if args.integrate:
        #print('\nEx=%s'%(s.Ex,))
        #print('\nEy=%s'%(s.Ey,))
        #print('\nSigmaxx=%s'%(s.Sigmaxx,))
        #print('\nSigmaxy=%s'%(s.Sigmaxy,))
        #print('\nSigmayy=%s'%(s.Sigmayy,))
        #print('\nx2=%s'%(s.x2,))
        v = (0,0,1.0/24, args.u)
        print('''
        z1=%s
        z2=%s
        z3=%s
        Ez=%s
        Sigma=
%s'''%(s.z1(*v),s.z2(*v),s.z3(*v),s.Ez(*v),s.Sigma(*v)))
    if args.out != None:
        plot(args,s)
    return 0

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
