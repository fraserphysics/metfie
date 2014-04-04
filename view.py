'''view.py displays marginal densities or eigenfunctions using mayavi.
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
                        default='200_g_200_h_32_y',
                        help='Read LO_step instance from this file.')
    args = parser.parse_args(argv)

    LO = first.read_LO_step(args.archive)
    LO.calc_marginal()

    plots = [] # to keep safe from garbage collection
    plots.append(plot(LO, LO.marginal))
    plots.append(plot(LO, LO.eigenvector))
    plots.append(plot(LO, LO.symmetry(LO.eigenvector)))
    ML.show()
    return 0
    
def plot(op, # Linear operator
         f   # data to plot
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
    b = np.log10(np.fmax(f, floor))
    z = op.vec2z(b, g, h, np.log10(floor))
    fig_0 = ML.figure()
    X,Y,Z,ranges = scale(G,H,z.T)
    print('b[0]=%e, b.max=%s'%(b[0],b.max()))
    print('z[0,0]=%e, z.max=%s'%(z[0,0],z.max()))
    print('Z[0,0]=%e, Z.max=%s\n'%(Z[0,0],Z.max()))
    ranges[-2:] = [floor, z.max()] # Make plot show max before log
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
