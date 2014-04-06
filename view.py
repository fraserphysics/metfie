'''view.py displays marginal densities and eigenfunctions using mayavi.
'''
import numpy as np
import mayavi.mlab as ML
import sys
import first
def main(argv=None):
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='''Display specified data using mayavi ''')
    parser.add_argument('--archive', type=str,
                        default='400_g_400_h_32_y',
                        help='Read LO_step instance from this file.')
    parser.add_argument('--log_floor', type=float, default=(1e-20),
                       help='log fractional deviation')
    args = parser.parse_args(argv)

    LO = first.read_LO_step(args.archive)
    LO.calc_marginal()

    plots = [] # to keep safe from garbage collection
    for f in (LO.eigenvector,LO.marginal):
        plots.append(plot(LO, f, log=True, log_floor=args.log_floor))
        plots.append(plot(LO, f, log=False))
    ML.show()
    return 0
    
def plot(op, # Linear operator
         f,  # data to plot
         log=True,
         log_floor=1e-20
    ):
    '''Make a surface plot of f(op_g,op_h)
    '''
    assert len(f) == op.n_states

    def scale(*xyz):
        '''This is a ultility function to prepare data for mayavi surface plots.
        xyz should be a list of three numpy arrays each with the same
        shape.  This function calculates and applies a scalar affine
        transformation to each so that the results range from 0 to 1.
        It returns those 3 arrays and the original bounds.
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
    
    if log:
        f = np.log10(np.fmax(f, log_floor))
        floor=np.log10(log_floor)
    else:
        floor=0.0
    #z = op.vec2z(f, g, h, floor=floor) # Uses/tests spline
    z = op.vec2z(f, floor=floor)
    X,Y,Z,ranges = scale(G,H,z.T)
    ranges[-2:] = [floor, z.max()] # Make plot show max before log
    fig_0 = ML.figure()
    s_0 = ML.mesh(X,Y,Z, figure=fig_0) # This call makes the surface plot
    ML.axes(ranges=ranges,xlabel='G',ylabel='H',zlabel='v', figure=fig_0)

    return fig_0

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
