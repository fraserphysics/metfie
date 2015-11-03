"""plot.py derived from ../plot.py makes plots for notes.pdf.

"""
plot_dict = {} # Keys are those keys in args.__dict__ that ask for
               # plots.  Values are the functions that make those
               # plots.
import sys
import matplotlib as mpl
import numpy as np
from eos import Go

figwidth = 8 # Determines apparent font size in figures
fig_y_size = lambda y: (figwidth, figwidth/9.0*y)
def main(argv=None):
    import argparse
    import os
    import os.path

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Make plots for notes.pdf')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--fig_dir', type=str, default='figs', help=
    'Directory of figures')
    # Plot requests
    h_format = lambda s:'File for figure of {0}'.format(s)
    parser.add_argument('--C_gun', type=str, help=h_format('d c_v/ d c_p'))
    parser.add_argument('--vt_gun', type=str, help=h_format('v(t)'))
    parser.add_argument('--BC_gun', type=str, help=h_format('d v(t)/ d c_p'))
    parser.add_argument('--opt_result', type=str, help=h_format(
        'one optimization step'))
    parser.add_argument('--big_d', type=str, help=h_format(
        'finite difference derivative with 9 subplots'))
    parser.add_argument('--fve_gun', type=str, help=h_format(
        'force, velocity and sequence of errors'))
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
    if args.show:
        mpl.rcParams['text.usetex'] = False
    else:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use

    if not os.path.exists(args.fig_dir):
        os.mkdir(args.fig_dir)
        
    # Do quick calculations to create exp, nom and opt_args
    import eos
    import gun
    
    t=np.linspace(0, gun.magic.t_max, gun.magic.n_t_sim)
    exp = Go(eos=eos.Experiment())
    nom = Go(eos=eos.Spline_eos(eos.Nominal(), precondition=True))
    for go in (exp, nom):
        go.add(t=t, gun=gun.Gun(go.eos))
        go.add(x=np.linspace(go.gun.x_i, go.gun.x_f, 500))
        go.add(t2v=go.gun.fit_t2v())
        go.add(v=go.t2v(t))
        go.add(vt=(go.v, go.t))
    C=nom.gun.fit_C()
    B,ep = nom.gun.fit_B_ep(exp.vt)
    nom.add(C=C, B=B, ep=ep, BC=np.dot(B,C))
    
    opt_args = (nom.eos, {'gun':nom.gun}, {'gun':exp.vt})
    
    # Make requested plots
    do_show = args.show
    for key in args.__dict__:
        if key not in plot_dict:
            continue
        if args.__dict__[key] == None:
            continue
        print('work on %s'%(key,))
        fig = plot_dict[key](exp, nom, opt_args, plt)
        file_name = getattr(args, key)
        if file_name == 'show':
            do_show = True
        else:
            fig.savefig(os.path.join(args.fig_dir, file_name), format='pdf')
    if do_show:
        plt.show()
    return 0

def C_gun(exp, nom, opt_args, plt):
    fig = plt.figure('C', figsize=fig_y_size(6.4))
    n_f = len(nom.eos.get_c())
    n_v = len(exp.t2v.get_c())
    assert nom.C.shape == (n_v, n_f)
    ax = fig.add_subplot(1,1,1)
    for j in range(n_f):
        ax.plot(nom.C[:,j]*1e7)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(
        r'$\left( \frac{\partial c_v[k]}{\partial c_f[i]} \right)\cdot 10^7$')
    return fig
plot_dict['C_gun'] = C_gun

def vt_gun(exp, nom, opt_args, plt):
    fig = plt.figure('vt', figsize=fig_y_size(8))
    ts = nom.t2v.get_t()
    ax = fig.add_subplot(1,1,1)
    ax.plot(nom.t*1e6, nom.v/1e5, label='simulation')
    ax.plot(exp.t*1e6, exp.v/1e5, label='experiment')
    ax.plot(nom.t*1e6, nom.ep/1e5, label=r'error $\epsilon$')
    ax.set_xlabel(r'$t/(\mu \rm{sec})$')
    ax.set_ylabel(r'$v/(\rm{km/s})$')
    ax.legend(loc='upper left')
    return fig
plot_dict['vt_gun'] = vt_gun

def BC_gun(exp, nom, opt_args, plt):
    fig = plt.figure('BC', figsize=fig_y_size(7))
    C = nom.C
    BC = np.dot(nom.B,nom.C)
    n_t,n_c = BC.shape
    ax = fig.add_subplot(1,1,1)
    for i in range(n_c):
        ax.plot(exp.t*1e6, BC[:,i]*1e7)
    ax.set_xlabel(r'$t/(\mu\rm{sec})$')
    ax.set_ylabel(
        r'$\frac{\partial v(t)}{\partial c_f[i]}/({\rm cm/s})\cdot 10^7$')
    return fig
plot_dict['BC_gun'] = BC_gun

def opt_result(exp, nom, opt_args, plt):
    from fit import Opt
    from gun import magic
    
    fig = plt.figure('opt_result', figsize=fig_y_size(6))    
    p2f = magic.newton2dyne*magic.area/1e11
    opt = Opt(*opt_args)
    eos_0 = opt.eos
    opt.fit(max_iter=1)
    eos_1 = opt.eos
    nom.gun.eos = nom.eos # Restore after optimization
    
    ax = fig.add_subplot(1,1,1)
    ax.plot(exp.x,eos_0(exp.x)*p2f,label=r'$f_0$')
    ax.plot(exp.x,eos_1(exp.x)*p2f,label=r'$f_1$')
    ax.legend()
    ax.set_xlabel(r'$x/{\rm cm}$')
    ax.set_ylabel(r'$f/({\rm dyn}\cdot 10^{11})$')
    return fig
plot_dict['opt_result'] = opt_result

def big_d(exp, nom, opt_args, plt):
    ''' 3x3 matrix of plot that illustrate finite difference derivative
    '''
    fig = plt.figure('big_d', figsize=(14,16))
    gun = nom.gun
    t2v = nom.t2v
    x = exp.x
    frac_a = 2.0e-2
    frac_b = 2.0e-3
    CA = gun.fit_C(fraction=frac_a)*1e7
    CB = gun.fit_C(fraction=frac_b)*1e7
    x_C = t2v.get_t()[2:-2]*1.0e6

    c = nom.eos.get_c()
    f = nom.eos(x)
    t = np.linspace(0,105.0e-6,500)
    v = t2v(t)
    def delta(frac):
        df = []
        dv = []
        for i in range(len(c)):
            c_ = c.copy()
            c_[i] = c[i]*(1+frac)
            gun.eos = nom.eos.new_c(c_)
            t2v_i = gun.fit_t2v()
            df.append(gun.eos(x)-f)
            dv.append(t2v_i(t) - v)
        return np.array(df),np.array(dv)
    dfa, dva = delta(frac_a)
    dfb, dvb = delta(frac_b)
    # Positions for add_subplot(4,3,n_)
    #  1  2  3
    #  4  5  6
    #  7  8  9
    mic_sec = r'$t/(\mu \rm{sec})$'
    for n_, x_, y_, l_x, l_y in (
            (1, x, np.array([f]), '$v$', '$f$'),
            (2, x, dfa, '$v$', '$\Delta f$'),
            (3, x, dfb, '$v$', '$\Delta f$'),
            (4, t, np.array([v]),mic_sec, r'$v/(\rm{km/s})$'),
            (5, t, dva,mic_sec, '$\Delta v$'),
            (6, t, dvb, mic_sec,'$\Delta v$'),
            (7, x_C, (CA.T-CB.T)*1e3, mic_sec, r'$\rm Difference\cdot 10^3$'),
            (8, x_C, CA.T, mic_sec, '$\Delta c_v/\Delta c_f\cdot 10^7$'),
            (9, x_C, CB.T, mic_sec, '$\Delta c_v/\Delta c_f\cdot 10^7$'),
            ):
        ax = fig.add_subplot(3,3,n_)
        ax.set_xlabel(l_x)
        ax.set_ylabel(l_y)
        n_y, n_x = y_.shape
        if n_ in (4,5,6):
            y_ = y_/1.0e5 # magic.cm2km
            if n_ in (5,6):
                y_ *= 1e3
                ax.set_ylabel(r'$\Delta v/(\rm{m/s})$')
            x_ = x_*1.0e6
        if n_ in (2,3):
            ax.set_ylim(ymin=1e4, ymax=1e9)
        for i in range(n_y):
            if n_ in (1,2,3):
                ax.loglog(x_, y_[i])
            else:
                ax.plot(x_, y_[i])
    fig.subplots_adjust(wspace=0.3) # Make more space for label

    return fig
plot_dict['big_d'] = big_d

def fve_gun(exp, nom, opt_args, plt):
    from fit import Opt
    from gun import magic
    from gun import Gun
    
    fig = plt.figure('fve_gun',figsize=fig_y_size(9))    
    p2f = magic.newton2dyne*magic.area/1e11    
    opt = Opt(*opt_args)
    cs,costs = opt.fit(max_iter=5)
    print('costs={0}'.format(costs))
    opt_gun = Gun(opt.eos)
    nom.gun.eos = nom.eos # Restore nominal eos after optimization
    t2vs = [Gun(eos).fit_t2v() for eos in [opt.eos.new_c(c) for c in cs]]
    e = [exp.v - t2v(exp.t) for t2v in t2vs[1:]]

    data = {'nominal':(
        (exp.x, nom.eos(exp.x)*p2f, 'f'),
        (exp.x, nom.gun.x_dot(exp.x)/magic.cm2km, 'v'))}
    data['experimental']=(
        (exp.x, exp.eos(exp.x)*p2f, 'f'),
        (exp.x, exp.gun.x_dot(exp.x)/magic.cm2km, 'v'))
    data['fit']=(
        (exp.x, opt.eos(exp.x)*p2f, 'f'),
        (exp.x, opt_gun.x_dot(exp.x)/magic.cm2km, 'v'))
    
    cm = r'$x/(\rm{cm})$'
    mu_sec = r'$t/(\mu\rm{sec})$'
    f_key = r'$f/({\rm dyn}\cdot 10^{11})$'
    v_key = r'$v/(\rm{km/s})$'
    e_key = r'$\epsilon_k/(\rm{m/s})$'
    ax_d = {
        'f':{'ax':fig.add_subplot(3,1,1), 'l_x':cm,
             'l_y':f_key, 'loc':'upper right'},
        'v':{'ax':fig.add_subplot(3,1,2), 'l_x':cm,
            'l_y':v_key, 'loc':'lower right'},
        'e':{'ax':fig.add_subplot(3,1,3), 'l_x':mu_sec,
             'l_y':e_key, 'loc':'upper right'}
    }
    for mod,xyn in data.items():
        for x,y,name in xyn:
            if mod == 'experimental':
                ax_d[name]['ax'].plot(x,y,'r-.',label=r'$\rm %s$'%mod)
            else:
                ax_d[name]['ax'].plot(x,y,label=r'$\rm %s$'%mod)
    for i in range(len(e)):
        ax_d['e']['ax'].plot(exp.t*1e6,e[i]/100,label=r'$\epsilon_%d$'%(i+1))
    for name,d in ax_d.items():
        d['ax'].legend(loc=ax_d[name]['loc'])
        d['ax'].set_xlabel(d['l_x'])
        d['ax'].set_ylabel(r'$%s$'%name)
        if 'l_y' in d:
            d['ax'].set_ylabel(d['l_y'])
    
    return fig
plot_dict['fve_gun'] = fve_gun

if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# eval: (python-mode)
# End:
