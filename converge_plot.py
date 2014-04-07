description='''Make surface plots from pickled results of a convergence
study produced by converge.py.
'''
import sys


import matplotlib as mpl
params = {'axes.labelsize': 18,     # Plotting parameters for latex
          'text.fontsize': 15,
          'legend.fontsize': 15,
          'text.usetex': True,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
mpl.rcParams.update(params)
if True:
    DEBUG = True
    mpl.rcParams['text.usetex'] = False
else:
    mpl.use('PDF')
import matplotlib.pyplot as plt          # must be after mpl.use
from mpl_toolkits.mplot3d import Axes3D  # Mysteriously necessary
                                         # for "projection='3d'".
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
magnitude = lambda A: int(np.log10(np.abs(A).max()))-1

def main(argv=None):
    import argparse
    from converge import read_study
    from plot import axis
    
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--in_file', type=str, default='result_converge', 
        help="file of results")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--azim', type=float, default=-140, help=
                        'view angle for plots')
    parser.add_argument('--elev', type=float, default=25, help=
                        'view angle for plots')
    parser.add_argument('--verbose', type=int, default=0, help=
                        '0:silent, 1:args, 2:pickle_text, 3:all')
    args = parser.parse_args(argv)
    
    g,h,error,eigenvalue = read_study(args.in_file,args.verbose)
    if args.verbose>2:
        for j in range(len(g)):
            for i in range(len(h)):
                print('d_h=%8.2e, d_g=%8.2e, frac_error=%6.4f, e_val=%9.3e'%
                      (h[i],g[j],error[i,j],eigenvalue[i,j]))
    
    g_data = np.array(g)
    g_mag = magnitude(g_data)
    g_scale = g_data/10**g_mag
    g_ax = axis(data=g_scale, label='d_g', magnitude=g_mag, ticks=g_scale)
    
    h_data = np.array(h)
    h_mag = magnitude(h_data)
    h_scale = h_data/10**h_mag
    h_ax = axis(data=h_scale, label='d_h', magnitude=h_mag, ticks=h_scale)
    
    def de_f(e,f):
        de_df = (e[1]-e[0])/(f[1]-f[0])
        return de_df*f[0]
    ev_zero = (eigenvalue[0,0]
               - de_f(eigenvalue[:,0],h_data)
               - de_f(eigenvalue[0,:],g_data))
    print('%-16s= %e'%('eigenvalue(0,0)',ev_zero,))
    plot(g_ax,h_ax,error,'(|v - v_{ref}|/|v_{ref}|)',args)
    if eigenvalue.max() > 0: # Test for handling old files wo eigenvalues
        plot(g_ax,h_ax,eigenvalue,'\lambda',args)

    plt.show()
    return 0

def plot(g_ax,h_ax,z,z_label,args):
    '''Function to plot result of main.
    '''
    import numpy as np
    from plot import axis
    
    G,H = np.meshgrid(g_ax.data, h_ax.data)
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1,1,1, projection='3d', elev=args.elev, azim=args.azim)
    g_ax.set_label(ax.set_xlabel)
    g_ax.set_ticks(ax.set_xticks, ax.set_xticklabels)
    h_ax.set_label(ax.set_ylabel)
    h_ax.set_ticks(ax.set_yticks, ax.set_yticklabels)

    z_min = z.min()
    z_max = z.max()
    z_data = np.array((z_min,z_max))
    z_mag = magnitude(z_data)
    z_scale = z_data/10**z_mag
    z_ax = axis(data=z_scale, magnitude=z_mag, label=z_label)
    
    z_ax.set_label(ax.set_zlabel)
    z_ax.set_ticks(ax.set_zticks, ax.set_zticklabels)
    
    surf = ax.plot_surface(G, H, z/10**z_mag, rstride=1, cstride=1,
                           cmap=mpl.cm.jet, linewidth=1,
                antialiased=False)
    
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
