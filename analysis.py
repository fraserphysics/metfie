'''analysis.py plots numerically estimated eigenfunctions and analytic
approximations as functions of h at fixed g=0.

'''
import numpy as np
import matplotlib as mpl
import sys
import first_c
def main(argv=None):
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='''Line plots of eigenfunctions''')
    parser.add_argument('--file', type=str,
                        default='400_g_400_h_32_y',
                        help='Read LO_step instance from this file.')
    parser.add_argument('--dir', type=str,
                        default='archive',
                        help='Read LO_step instance from this file.')
    parser.add_argument('--out', type=str, default=None,
        help="Write result to this file")
    args = parser.parse_args(argv)
    params = {'axes.labelsize': 18,     # Plotting parameters for latex
              'text.fontsize': 15,
              'legend.fontsize': 15,
              'text.usetex': True,
              'font.family':'serif',
              'font.serif':'Computer Modern Roman',
              'xtick.labelsize': 15,
              'ytick.labelsize': 15}
    mpl.rcParams.update(params)
    if args.out != None:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use

    import pickle, os.path
    from datetime import timedelta

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
    LO = first_c.read_LO_step(args.file, args.dir)
    fig = line_plot(LO, _dict['dy'], _dict['eigenvalue'], plt)
    if args.out == None:
        plt.show()
    else:
        fig.savefig( open(args.out, 'wb'), format='pdf')
    return 0

def line_plot(op, dy, e_val, plt):
    gh = []
    for i in range(op.n_states):
        g,h,G,H = op.state_list[i]
        if G == int(op.n_g/2):
            gh.append((g,h))
    gh.sort()
    gh = np.array(gh, dtype=np.float64)
    g = gh[:,0]
    h = gh[:,1]
    op.spline()
    f = op.bs.ev(g,h)
    gamma = (dy/e_val)**.5
    h_lim = (24*op.g_max)**.5
    s = np.sinh(gamma*(h_lim - h))
    s *= f[0]/s[0]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    #ax.plot(h, f)
    ax.semilogy(h, f, label='eigenfunction')
    ax.semilogy(h, s, label='sinh')
    ax.legend(loc='lower left')
    return fig

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
