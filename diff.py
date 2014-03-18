'''diff.py Plots the difference between two estimates of the Perron
Frobenius function, v_{PF}, ie, the eigenfunction corresponding to the
largest eigenvector of the first order Markov integral operator.

Default arguments yields:

    For small, n_g=226 and n_h=227

    For big, n_g=201 and n_h=202

    user	0m31.618s



'''
import numpy as np
import mayavi.mlab as ML
import sys
from first_c import LO_step as LO
def main(argv=None):
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='''Plot differences between estimates of the Perron
        Frobenius function of an integral operator''')
    parser.add_argument('--u', type=float, default=(2.0e-5),
                       help='log fractional deviation')
    parser.add_argument('--dy', type=float, default=3.2e-4,
                       help='y_1-y_0 = log(x_1/x_0)')
    parser.add_argument('--n_g0', type=int, default=200,
                       help='number of integration elements in value')
    parser.add_argument('--n_h0', type=int, default=200, help=
'number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    parser.add_argument('--n_g1', type=int, default=225)
    parser.add_argument('--n_h1', type=int, default=225)
    args = parser.parse_args(argv)

    d_g = {'big':2*args.u/args.n_g0, 'small':2*args.u/args.n_g1}
    h_lim = np.sqrt(48*args.u)
    d_h = {'big':2*h_lim/args.n_h0, 'small':2*h_lim/args.n_h1}
    
    tol = 5e-6
    maxiter = 150
    LO_ = {}
    for size in ('small', 'big'):
        op = LO( args.u, args.dy, d_g[size], d_h[size])
        op.power(small=tol, n_iter=maxiter)
        print('For %s, n_g=%d and n_h=%d\n'%(size, op.n_g, op.n_h))
        LO_[size] = op
    diffs = {}
    plots = [] # to keep safe from garbage collection
    for a,b in (('big','small'),('small','big')):
        x = LO_[a].xyz()
        d, f = LO_[b].diff(x[0],x[1],x[2],rv=True)
        print('%s.diff(%s)=%g'%(b, a, d))
        plots.append(plot(LO_[a], f))
    ML.show()
    return 0
    
def plot(op, # Linear operator
         f   # Difference of v_{PF}
    ):
    '''Function to plot a result of main.  
    '''
    assert len(f) == op.n_states

    def scale(*xyz):
        '''This is a ultility function to prepare data for mayavi surface plots.
        xyz should be a list of three numpy arrays each with the same
        shape.  This function calculates applies a scalar affine
        transformation to each so that the results range from 0 to 1.
        It returns those 3 arrays and the oringinal bounds.
        '''
        rv = []
        ranges = []
        for w in xyz:
            rv.append((w-w.min())/(w.max()-w.min()))
            ranges.append(w.min())
            ranges.append(w.max())
        rv.append(ranges)
        return rv
    
    g,h = op.gh()           # Get arrays of g and h values that occur
    G,H = np.meshgrid(g, h) # Make 2d arrays
    
    floor=1e-30             # Need positive floor because I use logs
    v = op.eigenvector
    b = np.log10(np.fmax(v, floor))
    z = op.vec2z(b, g, h, np.log10(floor))
    fig_0 = ML.figure()
    X,Y,Z,ranges = scale(G,H,z.T)
    ranges[-2:] = [floor, v.max()] # Make plot show max before log
    s_0 = ML.mesh(X,Y,Z, figure=fig_0) # This call makes the surface plot
    ML.axes(ranges=ranges,xlabel='G',ylabel='H',zlabel='v', figure=fig_0)

    z = np.zeros(G.T.shape)
    for i in range(op.n_states):
        g,h,g_int,h_int = op.state_list[i]
        z[g_int,h_int] = f[i]
    fig_1 = ML.figure()
    X,Y,Z,ranges = scale(G,H,z.T)
    s_1 = ML.mesh(X,Y,Z, figure=fig_1) # This call makes the surface plot
    ML.axes(ranges=ranges,xlabel='G',ylabel='H',zlabel='d', figure=fig_1)
    return (fig_0, fig_1)

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
