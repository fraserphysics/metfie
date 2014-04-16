'''analysis.py plots numerically estimated eigenfunctions and analytic
approximations as functions of h at fixed g=0.

'''
import numpy as np
import sys
import first
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
    args = parser.parse_args(argv)

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
    LO = first.read_LO_step(args.file, args.dir)
    line_plot(LO, _dict['dy'], _dict['eigenvalue'])
    return 0

def line_plot(op, dy, e_val):
    import matplotlib as mpl
    import matplotlib.pyplot as plt  # must be after mpl.use
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
    plt.show()
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
