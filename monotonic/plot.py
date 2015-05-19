"""plot.py Makes plots for the documents in the monotonic directory.

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
        description='Make plots for monotonic docs')
    parser.add_argument('--debug', action='store_true')
    # Plot requests
    parser.add_argument('--plot_bounds', type=argparse.FileType('wb'),
                       help='Write figure to this file')
    args = parser.parse_args(argv)
    
    params = {'axes.labelsize': 18,     # Plotting parameters for latex
              'font.size': 15,
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

    # Make requested plots
    for key in args.__dict__:
        if key not in plot_dict:
            continue
        if args.__dict__[key] == None:
            continue
        print('work on %s'%(key,))
        fig = plot_dict[key](plt)
        if not DEBUG:
            fig.savefig(getattr(args, key), format='pdf')
    return 0


def plot_bounds(plt):
    slope = -.2
    fig = plt.figure('plot_bounds',figsize=(6,12))
    ax1 = plt.subplot2grid((4,1),(0,0))
    ax2 = plt.subplot2grid((4,1),(1,0))
    ax3 = plt.subplot2grid((4,1),(2,0),rowspan=2)
    x = np.arange(4,dtype=np.int32)
    t = np.linspace(-.3, 3.3, 500)
    
    for ax in (ax1, ax2):
        ax.set_xticks(x, minor=False)
    ax1.set_xticklabels([])
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlim(-.3, 3.3)
    ax1.set_yticks(np.linspace(-1.0, 1.0, 3))
    
    ax2.set_ylim(-.1, 1.1)
    ax2.set_xticklabels(['%d'%i for i in x])

    f_m = lambda t: np.maximum(-.3, 1-(t+0.3)**2/4)
    f_s = lambda b,t: b + slope*t
    for y in (f_s(0,t), f_s(1,t), f_m(t)):
        ax1.plot(t, y)
        ax2.plot(t, y-slope*t)
        
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    x = np.linspace(0,1,500)
    y = np.minimum(1, x - slope)
    ax3.plot(x, y)
    #fig.subplots_adjust(hspace=0.3) # Make more space for label
    return fig
plot_dict['plot_bounds'] = plot_bounds

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
