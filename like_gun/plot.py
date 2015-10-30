"""plot.py derived from ../plot.py makes plots for notes.pdf.

"""
plot_dict = {} # Keys are those keys in args.__dict__ that ask for
               # plots.  Values are the functions that make those
               # plots.
import sys
import matplotlib as mpl
import numpy as np

figwidth = 8 # Determines apparent font size in figures
fig_y_size = lambda y: (figwidth, figwidth/9.0*y)
class Go:
    ''' Generic object.
    '''
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
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
    parser.add_argument('--D_gun', type=str, help=h_format('d c_v/ d c_p'))
    parser.add_argument('--vt_gun', type=str, help=h_format('v(t)'))
    parser.add_argument('--BD_gun', type=str, help=h_format('d v(t)/ d c_p'))
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
        
    # Do quick calculations for all plots even if not necessary
    import eos
    import gun
    
    exp_eos = eos.Experiment()
    exp_gun = gun.Gun(exp_eos)
    exp_t2v = exp_gun.fit_t2v()
    exp_t = np.linspace(0, gun.magic.t_max, gun.magic.n_t_sim)
    exp_v = exp_t2v(exp_t)
    exp_vt = (exp_v, exp_t)
    exp_x = np.linspace(exp_gun.x_i, exp_gun.x_f, 500)
    
    nom_eos = eos.Spline_eos(eos.Nominal(), precondition=True)
    nom_gun = gun.Gun(nom_eos)
    nom_t2v = nom_gun.fit_t2v()
    nom_D = nom_gun.fit_D()
    nom_B, nom_ep = nom_gun.fit_B_ep(exp_vt)

    opt_args = (nom_eos, {'gun':nom_gun}, {'gun':exp_vt})
    
    calc = Go(
        exp_eos=exp_eos,
        exp_gun=exp_gun,
        exp_t2v=exp_t2v,
        exp_t=exp_t,
        exp_v=exp_v,
        exp_x=exp_x,

        nom_gun=nom_gun,
        nom_eos=nom_eos,
        nom_t2v=nom_t2v,
        nom_D=nom_D,
        nom_B=nom_B,
        nom_ep=nom_ep,

        opt_args=opt_args,
        )
        
    # Make requested plots
    do_show = args.show
    for key in args.__dict__:
        if key not in plot_dict:
            continue
        if args.__dict__[key] == None:
            continue
        print('work on %s'%(key,))
        fig = plot_dict[key](calc, args, plt)
        file_name = getattr(args, key)
        if file_name == 'show':
            do_show = True
        else:
            fig.savefig(os.path.join(args.fig_dir, file_name), format='pdf')
    if do_show:
        plt.show()
    return 0

def D_gun(calc, args, plt):
    D = calc.nom_D
    eos = calc.nom_eos
    t2v = calc.exp_t2v
    fig = plt.figure('D', figsize=fig_y_size(6.4))

    n_f = len(eos.get_c())
    n_v = len(t2v.get_c())
    assert D.shape == (n_v, n_f)
    ax = fig.add_subplot(1,1,1)
    for j in range(n_f):
        ax.plot(D[:,j]*1e7)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(
        r'$\left( \frac{\partial c_v[k]}{\partial c_f[i]} \right)\cdot 10^7$')
    return fig
plot_dict['D_gun'] = D_gun

def vt_gun(calc, args, plt):
    t = calc.exp_t
    v = calc.exp_v
    t2v = calc.nom_t2v
    ep = calc.nom_ep

    ts = t2v.get_t()
    fig = plt.figure('vt', figsize=fig_y_size(6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(ts*1e6, t2v(ts)/1e5, label='simulation')
    ax.plot(t*1e6, (v+ep)/1e5, label='experiment')
    ax.plot(t*1e6, ep/1e5, label=r'error $\epsilon$')
    ax.set_xlabel(r'$t/(\mu \rm{sec})$')
    ax.set_ylabel(r'$v/(\rm{km/s})$')
    ax.legend(loc='upper left')
    return fig
plot_dict['vt_gun'] = vt_gun

def BD_gun(calc, args, plt):
    B = calc.nom_B
    D = calc.nom_D
    exp_t = calc.exp_t
    BD = np.dot(B,D)
    n_t,n_c = BD.shape
    
    fig = plt.figure('BD', figsize=fig_y_size(7))
    
    ax = fig.add_subplot(1,1,1)
    for i in range(n_c):
        ax.plot(exp_t*1e6, BD[:,i]*1e7)
    ax.set_xlabel(r'$t/(\mu\rm{sec})$')
    ax.set_ylabel(
        r'$\frac{\partial v(t)}{\partial c_f[i]}/({\rm cm/s})\cdot 10^7$')
    return fig
plot_dict['BD_gun'] = BD_gun

def opt_result(calc, args, plt):
    from fit import Opt
    from gun import magic
    
    p2f = magic.newton2dyne*magic.area/1e11
    opt = Opt(*calc.opt_args)
    
    x = calc.exp_x
    
    eos_0 = opt.eos
    opt.fit(max_iter=1)
    eos_1 = opt.eos
    
    fig = plt.figure('opt_result', figsize=fig_y_size(6))
    
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,eos_0(x)*p2f,label=r'$f_0$')
    ax.plot(x,eos_1(x)*p2f,label=r'$f_1$')
    ax.legend()
    ax.set_xlabel(r'$x/{\rm cm}$')
    ax.set_ylabel(r'$f/({\rm dyn}\cdot 10^{11})$')
    return fig
plot_dict['opt_result'] = opt_result

def big_d(calc, args, plt):
    ''' 3x3 matrix of plot that illustrate finite difference derivative
    '''
    gun = calc.nom_gun
    nom_eos = calc.nom_eos
    t2v = calc.nom_t2v
    x = calc.exp_x
    
    fig = plt.figure('big_d', figsize=(14,16))


    frac_a = 2.0e-2
    frac_b = 2.0e-3
    DA = gun.fit_D(fraction=frac_a)*1e7
    DB = gun.fit_D(fraction=frac_b)*1e7
    x_D = t2v.get_t()[2:-2]*1.0e6

    c = nom_eos.get_c()
    f = nom_eos(x)
    t = np.linspace(0,105.0e-6,500)
    v = t2v(t)
    def delta(frac):
        df = []
        dv = []
        for i in range(len(c)):
            c_ = c.copy()
            c_[i] = c[i]*(1+frac)
            gun.eos = nom_eos.new_c(c_)
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
            (1, x, np.array([f]), '$x$', '$f$'),
            (2, x, dfa, '$x$', '$\Delta f$'),
            (3, x, dfb, '$x$', '$\Delta f$'),
            (4, t, np.array([v]),mic_sec, r'$v/(\rm{km/s})$'),
            (5, t, dva,mic_sec, '$\Delta v$'),
            (6, t, dvb, mic_sec,'$\Delta v$'),
            (7, x_D, DA.T-DB.T, mic_sec, r'$\rm Difference$'),
            (8, x_D, DA.T, mic_sec, '$\Delta c_v/\Delta c_f\cdot 10^7$'),
            (9, x_D, DB.T, mic_sec, '$\Delta c_v/\Delta c_f\cdot 10^7$'),
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

def fve_gun(calc, args, plt):
    from fit import Opt
    from gun import magic
    from gun import Gun
    
    p2f = magic.newton2dyne*magic.area/1e11
    opt = Opt(*calc.opt_args)
    
    x = calc.exp_x
    nom_gun = calc.nom_gun
    nom_eos = calc.nom_eos
    v = calc.exp_v
    t = calc.exp_t
    exp_gun = calc.exp_gun
    exp_eos = calc.exp_eos

    cs = opt.fit(max_iter=5)
    opt_eos = opt.eos
    opt_gun = Gun(opt_eos)

    t2vs = [Gun(eos).fit_t2v() for eos in [opt_eos.new_c(c) for c in cs]]
    e = [v - t2v(t) for t2v in t2vs]
    
    fig = plt.figure('fve_gun',figsize=fig_y_size(9))


    data = {'nominal':(
        (x, nom_eos(x), 'f'),
        (x, nom_gun.x_dot(x)/magic.cm2km, 'v'))}
    data['experimental']=(
        (x, exp_eos(x), 'f'),
        (x, exp_gun.x_dot(x)/magic.cm2km, 'v'))
    data['fit']=(
        (x, opt_eos(x), 'f'),
        (x, opt_gun.x_dot(x)/magic.cm2km, 'v'))

    cm = r'$x/(\rm{cm})$'
    mu_sec = r'$t/(\mu\rm{sec})$'
    v_key = r'$v/(\rm{km/s})$'
    e_key = r'$e/(\rm{m/s})$'
    ax_d = {
        'f':{'ax':fig.add_subplot(3,1,1), 'l_x':cm,'loc':'upper right'},
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
        ax_d['e']['ax'].plot(t*1e6,e[i]/100,label='%d'%i)
    for name,d in ax_d.items():
        d['ax'].legend(loc=ax_d[name]['loc'])
        d['ax'].set_xlabel(d['l_x'])
        d['ax'].set_ylabel(r'$%s$'%name)
        if 'l_y' in d:
            d['ax'].set_ylabel(d['l_y'])
    
    return fig
plot_dict['fve_gun'] = fve_gun

def x(calc, args, plt):
    '''Template
    '''
    fig = plt.figure('x', figsize=fig_y_size(6.4))
    return fig
plot_dict['x'] = x

if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# eval: (python-mode)
# End:
