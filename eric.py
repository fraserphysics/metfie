'''eric.py for calculating moments of pie slices and plotting the
slices and their moments.
'''
import sys
import numpy as np
import sympy
from sympy import collect
params = {'axes.labelsize': 18,     # Plotting parameters for latex
          'text.fontsize': 15,
          'legend.fontsize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
import matplotlib as mpl
import numpy.linalg as LA

class sym:
    '''Symbolic calculations of characteristics of images of initial points
    z0=(h0,g0)
    '''
    def __init__(self):
        h, g, h0, g0, a, d = sympy.symbols('h g h0 g0 a d')
        #g1 = sympy.Max(-d, g0 + h0 - 1/(4*a))
        g1 = g0 + h0 - 1/(4*a)
        h1 = h0 - 1/(2*a)
        parabola = d - a*h*h       # =g on boundary
        slope = g1 + h - h1        # =g on line with slope=1 through z1
        h2 = sympy.solve(parabola-slope,h)[0] # First solution is above z1
        g2 = h2 - h1 + g1          # Line has slope of 1
        g3 = g1
        h3 = sympy.solve((parabola-g).subs(g, g1),h)[1] # Second is on right
        r_a = sympy.Rational(1,24) # a=1/24 always
        self.h = tuple(x.subs(a,r_a) for x in (h0, h1, h2, h3))
        self.g = tuple(x.subs(a,r_a) for x in (g0, g1, g2, g3))
        def integrate(f):
            ia = sympy.integrate(
                sympy.integrate(f,(g, g1, g1+h-h1)),
                (h, h1, h2)) # Integral of f over right triangle
            ib = sympy.integrate(
                sympy.integrate(f,(g, g1, parabola)),
                (h, h2, h3)) # Integral of f over region against parabola
            return (ia+ib).subs(a, r_a)
        i0 = integrate(1)               # Area = integral of pie slice
        E = lambda f:(integrate(f)/i0)  # Expected value wrt Lebesgue measure
        sigma = lambda f,g:E(f*g) - E(f)*E(g)
        self.d = d
        self.Eh = collect(E(h),sympy.sqrt(d-g0-h0+6))
        self.Eg = E(g)
        self.Sigmahh = sigma(h,h)
        self.Sigmahg = sigma(h,g)
        self.Sigmagg = sigma(g,g)
        return
    def Sigma(
            self,   # sym instance
            *args   # Float values for h0, g0 and d
    ):
        '''Returns covariance of image of z0=(h0,g0) as 2x2 array of floats.
        '''
        pairs = zip((self.h[0], self.g[0], self.d),args)
        # paris[0] = (self.h0 (a sympy variable), h0 (a float argument))
        hh = self.Sigmahh.subs(pairs).evalf()
        hg = self.Sigmahg.subs(pairs).evalf()
        gg = self.Sigmagg.subs(pairs).evalf()
        return np.array([[hh,hg],[hg,gg]])
    # FixMe: def zn(self, n, *args)
    def zn(
            self,  # sym instance
            n,     # index of point to return
            *args  # Float values for h0, g0 and d
    ):
        ''' Returns (h[n],g[n]) as pair of floats
        '''
        pairs = zip((self.h[0], self.g[0], self.d),args)
        h = self.h[n].subs(pairs).evalf()
        g = self.g[n].subs(pairs).evalf()
        return (h,g)
    def Ez(
            self,  # sym instance
            *args  # Float values for h0, g0 and d
    ):
        '''Returns expected value (ie, mean) of z over image of z0=(h0,g0) as
        pair of floats.
        '''
        pairs = zip((self.h[0], self.g[0], self.d),args)
        h = self.Eh.subs(pairs).evalf()
        g = self.Eg.subs(pairs).evalf()
        return (h,g)
    
def lineplot(*args,**kwargs):
    '''Plot a line from za to zb on ax and pass other args and kwargs to
    the plot call.
    '''
    ax,za,zb = args[0:3]
    ax.plot((za[0],zb[0]),(za[1],zb[1]),*args[3:],**kwargs)
def symplot(*args,**kwargs):
    '''Plot a point at za on ax and pass other args and kwargs to
    the plot call.
    '''
    ax,za = args[0:2]
    ax.plot(za[0],za[1],*args[2:],**kwargs)

def f_n(
        args,
        dummy,
        plt
):
    '''Make a figure to illustrate iterations of the function f that gives
    z_1 = f(z_0), namely f^n(h,g) = (h - 12n, g + n(h - 6n))

    '''
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    
    # Plot the boundary
    h_max = (48*args.d)**.5 # Scalar value of biggest h
    h =  np.linspace(-h_max, h_max, 41)
    boundary = lambda h: args.d -h*h/24
    g = boundary(h)
    ax.plot(h, g, 'b-', label=r'$\rm boundary$')

    F = lambda z: (z[0]-12, z[1]+z[0]-6)
    
    # Plot F^n
    h_0, g_0 = 0.0, args.d - 2.5
    n = np.linspace(-4, 4, 41)  # Step size 0.2
    h_n = h_0 - 12*n
    g_n = g_0 + n * (h_0 - 6*n)
    ax.plot(h_n, g_n, 'm.', label=r'$F^n$')

    # Plot images of points on h=0
    for z_0 in [(0,g) for g in np.linspace(-args.d, args.d, 11)]:
        z_1 = F(z_0)
        symplot(ax,z_0,'g.')
        symplot(ax,z_1,'r.')
        lineplot(ax,z_0,z_1,'g--')       # line from z0 to z1
    lineplot(ax,z_0,z_1,'g--', label=r'$[z_0,z_1]\,\rm lines$')
    ax.set_xlabel('$h$')
    ax.set_ylabel('$g$')
    ax.legend()
    return fig

def eric(
        args,
        s,     # A sym instance
        plt
    ):
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

    def ellipse(C,M=500):
        ''' Find M points on the level set x CI x = 1
        '''
        CI = LA.inv(C)
        a = CI[0,0]
        b = CI[0,1]
        c = CI[1,1]
        step = 2*np.pi/M
        theta = np.arange(0,2*np.pi+0.5*step,step)
        sin = np.sin(theta)
        cos = np.cos(theta)
        rr = 1/(a*cos*cos + 2*b*cos*sin + c*sin*sin)
        r = np.sqrt(rr)
        return (r*cos,r*sin)
    
    h_max = (48*args.d)**.5 # Scalar value of biggest h
    h =  np.linspace(-h_max, h_max, 1000)
    boundary = lambda h: args.d -h*h/24
    g = boundary(h)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    # Plot the boundary
    ax.plot(h, g, 'b-')
    ax.plot(h, np.ones(h.shape)*(-args.d),'b-')

    assert len(args.points)%2 == 0
    f_sources = np.array(tuple((args.points[2*i], args.points[2*i+1]) for i in 
                 range(int(len(args.points)/2))))
    g = args.d*f_sources[:,1]      # Verticies of pie slices, ie, g_1
    h_max = np.sqrt(24*(args.d-g)) # Array biggest h for each element of g
    h = f_sources[:,0]*h_max       # Verticies of pie slices, ie, h_1
    h_0 = h + 12
    g_0 = g - h_0 + 6              # (h_0[i],g_0[i])=preimage of ith pie slice
    for i in range(len(g)):
        v = h_0[i], g_0[i], args.d # Vector of args for calls to s.*()
        z = tuple(s.zn(n,*v) for n in range(4))
        Ez = s.Ez(*v)
        Sigma = s.Sigma(*v)
        lineplot(ax,z[1],z[3],'r-')        # horizontal line
        lineplot(ax,z[1],z[2],'r-')        # diagonal line
        lineplot(ax,z[0],z[1],'g--')       # line from z0 to z1
        h,g = ellipse(Sigma*2)
        if i == 0:
            symplot(ax,z[0],'g.',label=r'$z_0$')
            symplot(ax,z[1],'r.',label=r'$z_1$')
            symplot(ax,z[2],'m.',label=r'$z_2$')
            symplot(ax,z[3],'k.',label=r'$z_3$')
            symplot(ax,Ez,'b.',label=r'$\mu$')
            ax.plot(h+Ez[0], g+Ez[1], 'c-',label=r'$\sqrt{2}\Sigma$')
        else:
            symplot(ax,z[0],'g.')
            symplot(ax,z[1],'r.')
            symplot(ax,z[2],'m.')
            symplot(ax,z[3],'k.')
            symplot(ax,Ez,'b.')
            ax.plot(h+Ez[0], g+Ez[1], 'c-')
    ax.set_xlabel('$h$')
    ax.set_ylabel('$g$')
    ax.legend()
    return fig

def main(argv=None):
    '''For looking at map action and moments of images.

    '''
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=float, default=(48.0),
        help='Maximum of g')
    parser.add_argument(
        '--points', type=float, nargs='*', default=(
            -0.99,  -0.99,
            -1.0,  0.6,
            .5,-.1,
            -1.,  0.95,
        ),
        help='Specify pie slices with apexes at (fh,fg) as fractions')
    parser.add_argument('--eric', type=str, default=None,
        help="Write eric result to this file")
    parser.add_argument('--f_n', type=str, default=None,
        help="Write f_n result to this file")
    parser.add_argument('--latex', type=str, default=None,
        help="Write latex results to this file")
    parser.add_argument('--simplify', action='store_true',
        help="Call sympy.simplify.  Takes about 2 minutes.")
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args(argv)
    
    if args.eric == 'show' or args.f_n == 'show':
        assert args.eric == 'show' or args.eric == None
        assert args.f_n == 'show' or args.f_n == None
        mpl.use('Qt4Agg', warn=False)
    else:
        mpl.use('PDF', warn=False)
    import matplotlib.pyplot as plt  # must be after mpl.use
    mpl.rcParams.update(params)

    if args.test:
        print('No testing defined here')
        return 0
    s = sym()
    if args.latex != None:
        c = 'abcd'
        f = open(args.latex,'w')
        for i in range(1,len(s.h)):
            for q in 'h g'.split():
                r = sympy.latex(getattr(s,q)[i].simplify())
                f.write('\\newcommand{\\%s%c}{%s}\n\n'%(q,c[i],r))
        for q in 'Eh Eg Sigmahh Sigmahg Sigmagg'.split():
            if args.simplify:
                r = sympy.latex(getattr(s,q).simplify())
            else:
                r = sympy.latex(getattr(s,q))
            f.write('\\newcommand{\\%s}{%s}\n\n'%(q,r))
    for filename, function in ((args.eric, eric), (args.f_n, f_n)):
        
        if filename != None:
            fig = function(args,s,plt)
            if filename == 'show':
                plt.show()
            else:
                fig.savefig( open(filename, 'wb'), format='pdf')

    return 0

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
