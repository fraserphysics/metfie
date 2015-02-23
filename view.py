'''view.py displays marginal densities and eigenfunctions using mayavi.
'''
import numpy as np
import mayavi.mlab as ML
import matplotlib as mpl
import sys
from first_c import LO_step
#from first import LO_step
from first import Archive
def main(argv=None):
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='''Display specified data using mayavi ''')
    parser.add_argument('--d', type=float, default=100,
        help='Max g')
    parser.add_argument('--d_g', type=float, default=4,
        help='element size')
    parser.add_argument('--d_h', type=float, default=4,
        help='element size')
    parser.add_argument('--iterations', type=int, default=2,
        help='Apply operator n times and scale d, d_h and d_g')
    parser.add_argument('--dir', type=str,
                        default='archive',
                        help='Read LO instances from this directory.')
    parser.add_argument('--log_floor', type=float, default=(1e-20),
                       help='log fractional deviation')
    parser.add_argument('--fig_files', type=str, default=None,
                       help='Write results rather than display to screen')
    parser.add_argument(
        '--resolution', type=int, nargs=2, default=(None,None),
        help='Resolution in h and g')
    args = parser.parse_args(argv)
    m_h = args.resolution[0]
    m_g = args.resolution[1]

    import pickle, os.path
    from datetime import timedelta

    params = {'axes.labelsize': 18,     # Plotting parameters for latex
              'text.fontsize': 15,
              'legend.fontsize': 15,
              'text.usetex': True,
              'font.family':'serif',
              'font.serif':'Computer Modern Roman',
              'xtick.labelsize': 15,
              'ytick.labelsize': 15}
    mpl.rcParams.update(params)
    if args.fig_files != None:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use

    archive = Archive(LO_step)
    # Initialize operator
    d = args.d*args.iterations**2
    h_,g_ = 0,0
    A,image_dict = archive.get( d, args.d_h, args.d_g,
                                [(h_, g_, args.iterations)])
    keys = list(image_dict.keys())
    assert len(keys) == 1
    key = keys[0]
    assert key[2] == args.iterations
    plots = [] # to keep safe from garbage collection
    z_1 = A.eigenvector
    z_2 = image_dict[key]
    z_3 = z_1*z_2
    for f,name in ((z_1,'e_vec'),(z_3,'conditional')):
        if args.fig_files == None:
            plots.append(plot(A, f, log=True, log_floor=args.log_floor,
                              m_h=m_h,m_g=m_g))
            plots.append(plot(A, f, m_h=m_h,m_g=m_g, log=False))
        else:
            plot(A, f, m_h=m_h,m_g=m_g, log=False)
            ML.savefig('%s_%s.png'%(args.fig_files,name))
            plot(A, f, m_h=m_h,m_g=m_g, log=True, log_floor=args.log_floor)
            ML.savefig('%s_%s_log.png'%(args.fig_files,name))
    if args.fig_files == None:
        ML.show()
    return 0
    
def plot(op,               # Linear operator
         f,                # data to plot
         log=True,         # Plot log(f)
         log_floor=1e-20,  # Small f values and zeros beyond (g,h) range
         m_g=None,         # Interpolate g values
         m_h=None          # Interpolate h values
    ):
    '''Make a surface plot of f(op_g,op_h)
    '''
    #f = op.symmetry(f_)
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
    
    g,h = op.hg(m_h,m_g)    # Get arrays of g and h values that occur
    H,G = np.meshgrid(h, g) # Make 2d arrays
    
    if log:
        f = np.log10(np.fmax(f, log_floor))
        floor=np.log10(log_floor)
    else:
        floor=0.0
    if m_g == None:
        assert m_h == None
        z = op.vec2z(f, floor=floor)
    z = op.vec2z(f, h, g, floor=floor)
    X,Y,Z,ranges = scale(H,G,z)
    assert X.shape == Y.shape,'X:{0}, Y:{1}'.format(X.shape, Y.shape)
    assert Z.shape == Y.shape,'Z:{0}, Y:{1}'.format(Z.shape, Y.shape)
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
