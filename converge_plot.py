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

def main(argv=None):
    import argparse
    import numpy as np
    from converge import read_study
    
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--in_file', type=str, default='result_converge', 
        help="file of results")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', type=int, default=0, help=
                        '0:silent, 1:args, 2:pickle_text, 3:all')
    args = parser.parse_args(argv)
    
    g,h,error,eigenvalue = read_study(args.in_file,args.verbose)
    if args.verbose>2:
        for j in range(len(g)):
            for i in range(len(h)):
                print('d_h=%8.2e, d_g=%8.2e, frac_error=%6.4f, e_val=%9.3e'%
                      (h[i],g[j],error[i,j],eigenvalue[i,j]))
    plot(g,h,error)
    plot(g,h,eigenvalue)

    plt.show()
    return 0

def plot(g,h,z):
    '''Function to plot result of main.
    '''
    import numpy as np
    G,H = np.meshgrid(g, h)
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1,1,1, projection='3d', elev=21, azim=-94)
    surf = ax.plot_surface(
            G, H, z, rstride=1, cstride=1, cmap=mpl.cm.jet, linewidth=1,
            antialiased=False)
    ax.set_xlabel(r'$d_g$')
    ax.set_ylabel(r'$d_h$')
    
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
