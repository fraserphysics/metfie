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
    parser.add_argument('--file', type=str,
                        default='400_g_400_h_32_y',
                        help='Read LO_step instance from this file.')
    parser.add_argument('--dir', type=str,
                        default='archive',
                        help='Read LO_step instance from this file.')
    parser.add_argument('--log_floor', type=float, default=(1e-20),
                       help='log fractional deviation')
    parser.add_argument('--fig_files', type=str, default=None,
                       help='Write results rather than display to screen')
    parser.add_argument(
        '--resolution', type=int, nargs=2, default=(None,None),
        help='Resolution in g and h')
    args = parser.parse_args(argv)

    import pickle, os.path
    from datetime import timedelta

    m_g = args.resolution[0]
    m_h = args.resolution[1]
    _dict = pickle.load(open(os.path.join(args.dir,args.file),'rb'))
    g_max, dy, g_step, h_step = _dict['args']
    _dict.update({'g_max':g_max, 'dy':dy, 'g_step':g_step, 'h_step':h_step})
    keys = list(_dict.keys())
    keys.sort()
    for key in keys:
        if key == 'time':
            print('%-16s= %s'%('time',timedelta(seconds=_dict[key])))
            continue
        if key in set(('g_step','h_step')):
            print('%-16s= %e'%(key,_dict[key]))
            continue
        print('%-16s= %s'%(key,_dict[key]))
    LO = first.read_LO_step(args.file, args.dir)
    LO.calc_marginal()

    plots = [] # to keep safe from garbage collection
    for f,name in ((LO.eigenvector,'vec'),(LO.marginal,'marg')):
        if args.fig_files == None:
            plots.append(plot(LO, f, log=True, log_floor=args.log_floor,
                              m_g=m_g,m_h=m_h))
            plots.append(plot(LO, f, log=False, m_g=m_g, m_h=m_h))
        else:
            plot(LO, f, log=False, m_g=m_g, m_h=m_h)
            ML.savefig('%s_%s.png'%(args.fig_files,name))
            plot(LO, f, log=True, log_floor=args.log_floor, m_g=m_g, m_h=m_h)
            ML.savefig('%s_%s_log.png'%(args.fig_files,name))
    if args.fig_files == None:
        ML.show()
    return 0
    
def plot(op,               # Linear operator
         f_,               # data to plot
         log=True,         # Plot log(f)
         log_floor=1e-20,  # Small f values and zeros beyond (g,h) range
         m_g=None,         # Interpolate g values
         m_h=None          # Interpolate h values
    ):
    '''Make a surface plot of f(op_g,op_h)
    '''
    f = op.symmetry(f_)
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
    
    g,h = op.gh(m_g,m_h)    # Get arrays of g and h values that occur
    G,H = np.meshgrid(g, h) # Make 2d arrays
    
    if log:
        f = np.log10(np.fmax(f, log_floor))
        floor=np.log10(log_floor)
    else:
        floor=0.0
    if m_g == None:
        assert m_h == None
        z = op.vec2z(f, floor=floor)
    z = op.vec2z(f, g, h, floor=floor)
    X,Y,Z,ranges = scale(G,H,z.T)
    ranges[-2:] = [floor, z.max()] # Make plot show max before log
    fig_0 = ML.figure(size=(800,700))
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
