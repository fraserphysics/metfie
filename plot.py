"""plot.py Makes plots for the gun.pdf and juq.pdf documents.  It
imports class GUN from calc.py to do most of the calculations.

"""
DEBUG = False
plot_dict = {} # Keys are those keys in args.__dict__ that ask for
               # plots.  Values are the functions that make those
               # plots.
import calc    # Code that does the gun stuff
import sys
import matplotlib as mpl
import numpy as np
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D # Mysteriously necessary for
                                        # "projection='3d'".
def main(argv=None):
    import argparse
    global DEBUG

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Make plots for gun.pdf')
    parser.add_argument('--C', type=float, default=magic.C,
                       help='in cm^3 dyn. For EOS: f(x)=C/x^3')
    parser.add_argument('--xi', type=float, default=magic.xi,
                       help='Initial displacement in cm')
    parser.add_argument('--xf', type=float, default=magic.xf,
                       help='Final displacement in cm')
    parser.add_argument('--m', type=float, default=magic.m,
                       help='Projectile mass in grams')
    parser.add_argument('--N', type=int, default=magic.N_x,
                       help='Number of x samples')
    parser.add_argument('--Nf', type=int, default=magic.N_f,
                       help='Number of x samples')
    parser.add_argument('--dev', type=float, default=magic.DEV,
                       help='EOS tolerance')
    parser.add_argument('--debug', action='store_true')
    # Plot requests
    parser.add_argument('--plot_q0_1', type=argparse.FileType('wb'),
                       help='Write figure to this file')
    parser.add_argument('--plot_T_study', type=argparse.FileType('wb'),
                       help='Write figure to this file')
    parser.add_argument('--plot_q1R', type=argparse.FileType('wb'),
                       help='Write figure to this file')
    parser.add_argument('--plot_nominal', type=argparse.FileType('wb'),
                    help='Filename for plots of nominal EOS, Energy, and Time')
    parser.add_argument('--plot_mean', type=argparse.FileType('wb'),
                     help='Filename for plot of mean allowed EOS perturbation')
    parser.add_argument('--plot_allowedET', type=argparse.FileType('wb'),
                       help='Write ET study figure to this file')
    parser.add_argument(
        '--plot_allowed_tp2', type=argparse.FileType('wb'), help=
        'Write figure of bounding surfaces for f(t+2)|f(t),f(t+1) to this file')
    parser.add_argument(
        '--p_stat', type=argparse.FileType('wb'), help=
        'Make a surface plot of the stationary distribution')
    parser.add_argument('--plot_PCA', type=argparse.FileType('wb'),
                       help='Write PCA vals/vecs figure to this file')
    parser.add_argument('--plot_ellipsoid', type=argparse.FileType('wb'),
                       help='Write 2-d PCA figure to this file')
    parser.add_argument('--Plane', type=int, nargs='+',
                       help='Pairs of PCA components')
    parser.add_argument(
        '--plot_invariant', type=argparse.FileType('wb'),
        help='Write figure illustrating log-log invariance to this file')
    parser.add_argument('--plot_pert', type=argparse.FileType('wb'),
                       help='Write plot of fast transitions to this file')
    parser.add_argument('--plot_moments', type=argparse.FileType('wb'), help=
      'Write figure with 6 plots from stationary analysis to this file')
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
    if args.debug:
        DEBUG = True
        mpl.rcParams['text.usetex'] = False
    else:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use

    # Do the calculations
    gun = calc.GUN(args.C, args.xi, args.xf, args.m, args.N, args.dev)
    gun.stationary(args.Nf)
    gun.envelope_(args.Nf)
    
    # Make requested plots
    for key in args.__dict__:
        if key not in plot_dict:
            continue
        if args.__dict__[key] == None:
            continue
        print('work on %s'%(key,))
        if key=='plot_ellipsoid':
            fig = plot_dict[key](gun, args, plt)
        else:
            fig = plot_dict[key](gun, plt)
        if not DEBUG:
            fig.savefig(args.__dict__[key], format='pdf')
    return 0

magic = {     # Values that I invoke magically throughout module
'DEV':.05,    # Width of allowed variation. s[t]*(1-DEV/2) <z[t]< s[t]*(1+DEV/2)
'm':100.0,    # Mass of projectile in grams
'xi':0.4,     # Initial projectile position in cm
'xf':4.0 ,    # Muzzle position
'C':2.56e10,  # EOS constant
'N_x':1000,   # Number of x samples 1000
'N_f':4000,   # Number of bins for f 4000
'N_acf':50,   # Number of auto-correlation points to plot
'frac_acf':np.log10(1.5),# Fraction of x to use for plot
'N_plot':4,   # Number of eigenfunctions of ACF to plot
'samples':(0, 0.1) # (E,T) direct from ETz EOS's at these fractions of N_x
}
magic = calc.GO(magic)

# Utilities for axis labels in LaTeX with units in \rm font
label_magnitude_unit = lambda lab,mag,unit: (
    r'$%s/(10^{%d}\ {\rm{%s}})$'%(lab,mag,unit))
label_unit = lambda lab,unit: r'$%s/{\rm{%s}}$'%(lab,unit)
label_magnitude = lambda lab,mag: r'$%s/10^{%d}$'%(lab,mag)
magnitude = lambda A: int(np.log10(np.abs(A).max()))

class axis(object):
    ''' Class for managing scaling and labeling 1-d axes and data.
    '''
    def __init__(self,        # axis
                 **kwargs):   # Any keyword argument is legal
        ''' Hides some logic that figures out how to format axis
        labels, ticks and tick labels.
        '''
        self.__dict__.update(kwargs)
        defaults = dict(
            data=None,         # np array
            magnitude='auto',  # Power of 10.  False to suppress
            ticks='auto',      # np array.  False to suppress ticks
            label=False,       # string, eg 'force'
            units=None,        # eg, 'dyn'
            tick_label_flag=True)
        for key,value in defaults.items():
            if not key in self.__dict__:
                self.__dict__[key] = value
        if self.magnitude == 'auto' and isinstance(self.data,np.ndarray):
            self.magnitude = magnitude(self.data)
        if self.label == False:
            return
        # Calculate label string
        M = not (self.magnitude == 0 or self.magnitude == False)
        U = isinstance(self.units,str)
        self.label = {
         (True,True):label_magnitude_unit(self.label,self.magnitude,self.units),
         (True,False):label_unit(self.label,self.units),
         (False,True):label_magnitude(self.label,self.magnitude),
         (False,False):r'$%s$'%(self.label,)
            }[(U,M)]
        return
    def get_data(self,):
        if isinstance(self.magnitude,int):
            return self.data/10**self.magnitude
        return self.data
    def set_label(self,func):
        '''Apply func, eg, mpl.axis.set_xlabel(), to self._label
        '''
        if self.label != False:
            func(self.label)
        return
    def set_ticks(self,tick_func,label_func):
        '''Apply functions, eg, mpl.axis.set_xticks() and
        mpl.axis,set_xticklabels() to self.ticks
        '''
        if self.tick_label_flag == False:
            label_func([])
        if isinstance(self.ticks,str) and self.ticks == 'auto':
            return
        else:
            tick_func(self.ticks,minor=False)
        if self.tick_label_flag:
            if np.abs(self.ticks - self.ticks.astype(int)).sum() == 0:
                label_func([r'$%d$'%int(f) for f in self.ticks])
            else:
                label_func([r'$%1.1f$'%f for f in self.ticks])
        return
def SubPlot(fig,position,x,y,plot_flag=True,label=None,color='b'):
    ''' Make a subplot for fig using data and format in axis objects x and y
    '''
    ax = fig.add_subplot(*position)
    if plot_flag:
        if x.data == None:
            ax.plot(y.get_data(),color=color,label=label)
        else:
            ax.plot(x.get_data(),y.get_data(),color=color,label=label)
    y.set_label(ax.set_ylabel)
    y.set_ticks(ax.set_yticks,ax.set_yticklabels)
    x.set_label(ax.set_xlabel)
    x.set_ticks(ax.set_xticks,ax.set_xticklabels)
    return ax

def _plot_plane(fig,       # matplotlib fig
                location,  # matplotlib argument to subplot, eg, (2,1,1)
                ia,ib,     # Indices of vector coefficients to plot
                gun,
                center=None,
                n=500,     # Number of angles over which to find boundary
                ):
    ''' Plot ellipse and boundary of polytope in cross section
    defined by vectors a and b.
    '''
    def To_SS(i):
        '''Return a sequence of samples that represents the ith eigenfunction
        of the covariance.  Details of this function depend on the
        coordinates in which covariance was diagonalized.
        '''
        v = gun.cov_vecs[i]
        p = np.dot(v,v)
        q = gun.inner(v, gun.inner(gun.Cov,v))
        r = gun.cov_vals[i]
        assert abs(p-1) < 1e-10,'p=%f'%(p,)
        assert abs(q/r-1) < 1e-10,'q=%e r=%e'%(q,r) #FixMe: This should work
        return v/np.sqrt(gun.inner(v, v))
    a = To_SS(ia)
    b = To_SS(ib)
    if center == None:
        center = a*0
    x,y = gun.ellipse(a, b, M=n, Marginal=True)
    # Marginal/Conditional are the same since in diagonal coordinates
    coefficients,vectors = gun.edge(a, b, center, n)
    ax = fig.add_subplot(*location)
    ma = max(magnitude(x),magnitude(coefficients[0]))
    mb = max(magnitude(y),magnitude(coefficients[1]))
    ax.plot(2*x/10**ma,2*y/10**mb,label=r'$2\sigma$ ellipse')
    ax.plot(coefficients[0]/10**ma,coefficients[1]/10**mb,label=r'polytope')
    ax.set_xlabel(label_magnitude('C_{%d}'%ia,ma))
    ax.set_ylabel(label_magnitude('C_{%d}'%ib,mb))
    ax.legend()
    return

def plot_ellipsoid(gun, args, plt):
    N_p = int(len(args.Plane)/2)
    fig = plt.figure(figsize=(6*N_p,6))
    for i in range(N_p):
        a = args.Plane[2*i]
        b = args.Plane[2*i+1]
        _plot_plane(fig,
                    (1,N_p,i+1),
                    a,b,
                    gun,
                    None,       # Center FixMe should be mean?
                    500,        # Number of angles to plot
                    )
    return fig
plot_dict['plot_ellipsoid'] = plot_ellipsoid

def plot_pert(gun, plt):
    xi = 0.8
    xf = 1.4
    N = 500
    dev = 0.2
    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r'$x/\rm{cm}$')
    ax.set_xticks(np.arange(0.8, 1.41, 0.2), minor=False)
    ax.set_xlim(0.8, 1.4)
    ax.set_yticks(np.arange(0,6.1,2),minor=False)
    ax.set_ylim(-0.1, 6)
    mag = 10
    Y_label = label_magnitude_unit('f', mag, 'dyn')
    ax.set_ylabel(Y_label)
    gun_ = calc.GUN(gun.C,xi,xf,gun.m,N, dev) # New gun with custom xi and xf
    gun_.ET()
    x = gun_.x_c
    nom = gun_.s
    L,U = gun_.LU(nom)
    a = nom + gun_.ETz[int(.75*N),:] # Red L -> U
    b = nom + gun_.ETz[int(1.4*N),:]
    for y in (a, b):
        ax.plot(gun_.x_c, y/1e10)
    for y in (U, L):
        ax.plot(gun_.x_c, y/1e10, linestyle='dotted')
    fig.subplots_adjust(bottom=0.15) # Make more space for label
    return fig
plot_dict['plot_pert'] = plot_pert

def plot_invariant(gun, plt):
    xi = 0.7529
    xf = 1.3813
    N = 5
    mag = 10
    dev = 0.2
    fig = plt.figure(figsize=(9,8))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    ax1.set_yticks(np.arange(0,5.1,1),minor=False)
    ax1.set_xlim(0.8-3e-3, 1.3+3e-3)
    
    Y = np.arange(1.0e10, 5.1e10, 1e10)
    ax2.set_yticks(np.log10(Y),minor=False)
    ax2.set_yticklabels(['%5.0f'%(f/1e10) for f in Y])
    ax2.set_ylim(np.log10(1e10),np.log10(5.5e10))

    Y = [0.9, 1.0, 1.1]
    ax3.set_yticks(np.log10(Y),minor=False)
    ax3.set_yticklabels(['%5.2f'%f for f in Y])
    ax3.set_ylim(-0.05,0.05)
    
    gun_ = calc.GUN(gun.C,xi,xf,gun.m,N, dev) # New gun with custom xi and xf
    x = gun_.x_c
    logx = np.log10(x)
    nom = gun_.s
    L,U = gun_.LU(nom)
    Y_label = label_magnitude_unit('f',mag,'dyn')
    for y in (L,nom,U):
        ax1.plot(gun_.x_c,y/1e10)
        ax2.plot(np.log10(gun_.x_c),np.log10(y))
        ax3.plot(np.log10(gun_.x_c),np.log10(y/nom))
    for i in range(N):
        xx = (x[i],x[i])
        logxx = (logx[i],logx[i])
        ax1.plot(xx,(L[i]/1e10,U[i]/1e10),'k-')
        ax2.plot(logxx,np.log10((L[i],U[i])),'k-')
        ax3.plot(logxx,np.log10((L[i]/nom[i],U[i]/nom[i])),'k-')
    ax1.set_xlabel(r'$x$ linear scale')
    ax1.set_xticks(x,minor=False)
    ax1.set_xticklabels(['%4.2f'%f for f in x])
    ax1.set_ylabel(Y_label)
    
    ax2.set_xlabel(r'$x$ log scale')
    ax2.set_xticks(logx,minor=False)
    ax2.set_xlim(logx[0]-1e-3, logx[-1]+1e-3)
    ax2.set_xticklabels([])
    ax2.set_ylabel(r'$f/(10^{10})$~ log scale')
    
    ax3.set_xlabel(r'$x$ log scale')
    ax3.set_xticks(logx,minor=False)
    ax3.set_xlim(logx[0]-1e-3, logx[-1]+1e-3)
    ax3.set_xticklabels(['%4.2f'%f for f in x])
    ax3.set_ylabel(r'$f/\tilde f$~ log scale')
    fig.subplots_adjust(hspace=0.3) # Make more space for label
    return fig
plot_dict['plot_invariant'] = plot_invariant

def plot_T_study(gun, plt, orient=(3,1),shape=(6,9)):
    ''' This function makes plots that illustrate the
    linear approximation to muzzle exit time as a function of EOS.
    '''
    fig = plt.figure(figsize=shape)
    X = axis(data=gun.x_c,label='x', units='cm', ticks=np.arange(0,4.5,1))
    ax = SubPlot(fig,orient+(2,),X, axis(data=gun.q_1(),label='q_1',
          units='s/(dyn\, cm)',magnitude=-16,ticks=np.arange(-12,0.1,4)))
    ax.set_xlabel(ax.get_xlabel(),labelpad=1.5)
    X.tick_label_flag = False
    X.label = False
    SubPlot(fig,orient+(1,),X,axis(data=-1.0/gun.x_c,ticks=np.arange(-2,0.1,1),
                 label='\delta_1',units='(dyn)'))
    ax.set_ylim(-12,0)        # Clip because q_1(x_i) -> -\infty
    s = 1.5e9
    A = np.arange(-1.0,1.1,1.0)*s # 3 values for e_0
    B = np.arange(-1.0,1.1,1.0/10)*s
    T_0 = gun.T([gun.xi,gun.xf])[-1] # Nominal muzzle time
    X = axis(data=B,label='b',magnitude=8,
             ticks=np.arange(-15,16,5))
    Y = axis(label='(T(a,b)-T_0)',magnitude=-6,units='s',
             ticks=np.arange(-4,5,4))
    ax = SubPlot(fig,orient+(3,),X,Y,plot_flag=False)
    ax.set_xlim(-15,15)
    ax.set_ylim(-4,4)
    scale = 1e6
    b_scale = 1e8
    inner_1_0 = gun.inner(gun.q_1(), gun.e_0())
    inner_1_1 = gun.inner(gun.q_1(), gun.e_1())
    for a in A:
        T_ab = []
        for b in B:
            gun.set_a_b(a,b)  # Perturbs EOS by a*e_0 + b*e_1
            t = gun.T([gun.xi,gun.xf])[-1] - T_0
            T_ab.append(t*scale)
        ax.plot(B/b_scale,T_ab,color='r')
        ax.plot(B/b_scale,(
                B*inner_1_1+a*inner_1_0)*scale,color='g')
    fig.subplots_adjust(hspace=0.25) # Make more space for label
    gun.set_a_b(0,0) # Revert to unperturbed EOS
    return fig
plot_dict['plot_T_study'] = plot_T_study

def plot_q1R(gun, plt):
    return plot_T_study(gun,orient=(1,3),shape=(20,5))
plot_dict['plot_q1R'] = plot_q1R

def plot_allowedET(gun, plt):
    '''Get and plot E,T pairs from inner products and integration.
    '''
    fig = plt.figure(figsize=(9,5))
    gun.ET() # Construct set of functions that explore nonlinear boundary
    n = 30 # Number of extreme functions to plot
    X_0 = axis(data=gun.x_c,label='x',units='cm', ticks=np.arange(0,4.5,1))
    Y_0 = axis(data=gun.ETz[0],label='\delta(x)',units='dyn')
    ax = SubPlot(fig,(1,2,1),X_0,Y_0)
    m = 10**Y_0.magnitude
    ax.plot(gun.x_c,gun.ETz[1]/m)
    M,N = gun.ETz.shape
    for f in [gun.ETz[i,:] for i in range(1,M,int(M/n))]:
        ax.plot(gun.x_c,f/m)
    ax.set_ylabel(ax.get_ylabel(),labelpad=-9)

    # Plot marginal ellipse
    fudge = 1.0e15 # Makes plotting smoother.  Not an error
    a = gun.q_0()
    b = gun.q_1()*fudge
    E,T = gun.ellipse(a,b,M=2000,Marginal=True)
    ME = magnitude(E)
    MT = -6
    mul = 4
    ax = SubPlot(
        fig,(1,2,2),
        axis(data=mul*E,label=r'E_\delta',units='erg',magnitude=ME,
             ticks=np.arange(-20, 21,10)),
        axis(data=mul*T/fudge,label=r'T_\delta',units='s',magnitude=MT),
        label=r'$%d\sigma$Marginal'%mul)
    ax.set_xlim(-22,22)
    # Project functions onto functional derivatives for "Linear" trace.
    ax.plot(gun.inner(gun.ETz,gun.q_0())/10**ME,
            gun.inner(gun.ETz,gun.q_1())/10**MT,color='r',label=r'Linear')
    # Use perturbations in gun.ETz to plot direct calculations of E and T
    e_nom = gun.E(gun.xf)
    t_nom = gun.T([gun.xi,gun.xf])[-1]
    N = len(gun.x_c)
    N_direct = 2*len(magic.samples)
    E = np.empty(N_direct)
    T = np.empty(N_direct)
    for i in range(N_direct):
        j = i%2
        k = int(i/2)
        L = min(len(gun.ETz)-1, int(N*(j+magic.samples[int(i/2)])))
        gun.set_eos(gun.ETz[L])
        E[i] = gun.E(gun.xf) - e_nom
        T[i] = gun.T([gun.xi,gun.xf])[-1] - t_nom
    gun.set_eos() # Restore gun eos to nominal
    ax.plot(E/10**ME,T/10**MT,'.',color='k',label=r'Direct')
    ax.legend()
    fig.subplots_adjust(wspace=0.4, bottom=0.15) # Make more space for label
    return fig
plot_dict['plot_allowedET'] = plot_allowedET

def plot_nominal(gun, plt):
    fig = plt.figure(figsize=(6,9))
    X = axis(data=gun.x_c,label='x', units='cm')
    ax = SubPlot(fig,(4,1,4),X,axis(data=gun.T(gun.x_c),label='T',magnitude=-5,
                                    units='s',ticks=np.arange(0,11,5)))
    ax.set_title('Time')
    E = np.empty(len(gun.x_c))
    v = np.empty(len(gun.x_c))
    for i in range(len(gun.x_c)):
        x = gun.x_c[i]
        E[i] = gun.E(x)
        v[i] = gun.x_dot(x)
    X.label=False
    X.tick_label_flag=False
    ax = SubPlot(fig,(4,1,3),X,axis(data=v,label='\dot x',magnitude=4,
                                    units='cm/s',ticks=np.arange(0,5,2)))
    ax.set_title('Velocity')
    ax = SubPlot(fig,(4,1,2),X,axis(data=E,label='E',magnitude=10,units='erg',
                                    ticks=np.arange(0,9,4)))
    ax.set_title('Energy')
    ax = SubPlot(fig,(4,1,1),X,axis(data=gun.s,label='\\tilde f',magnitude=11,
                                    units='dyn',ticks=np.arange(0,5,2)))
    ax.set_title('Nominal EOS')
    return fig
plot_dict['plot_nominal'] = plot_nominal

def plot_PCA(gun, plt):
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(np.log10(gun.cov_vals))
    ax2 = fig.add_subplot(2,1,2)
    vecs = gun.cov_vecs
    L = int(0.6*len(gun.s))
    for k in range(magic.N_plot):
        v = vecs[k].copy()
        v /= np.sqrt(gun.inner(v,v))
        if v[0] < 0:
            v *= -1
        ax2.plot(gun.x_c[:L], v[:L], label=r'$\lambda_%d=%.2g$'%(
            k,gun.cov_vals[k]))
    ax2.legend(loc='upper right')
    ax1.set_ylabel(r'$\log_{10}(\lambda_i)$')
    ax1.set_xlabel(r'$i$')
    ax2.set_ylabel(r'$e_i(x)$')
    ax2.set_xlabel(r'$x/\rm{cm}$')
    return fig
plot_dict['plot_PCA'] = plot_PCA

def plot_moments(gun, plt):
    ''' FixMe: Check the notation.
    '''
    fig = plt.figure(figsize=(16,4))
    # Plot probability density function
    xticks = np.arange(0.98,1.03,0.02) # Need formatting outside SubPlot
    X = axis(data=gun.f/gun.s[0],magnitude=False,ticks=xticks,
             label='g')
    Y = axis(data=gun.Pf/gun.W,magnitude=False,ticks=np.arange(0,35,10),
             label='P_g')
    axa = SubPlot(fig,(1,3,1),X,Y)
    axa.set_xticklabels(['%4.2f'%x for x in (xticks-1)])
    # Plot ACF
    n = int(magic.frac_acf*len(gun.s))
    X = axis(data=gun.x_c[:n], magnitude=False,label='\chi', units='cm')
    Y = axis(data=gun.ACF[:n], magnitude=19,
             label='\Sigma(f(0.4),f(\chi))',units='dyn^2')
    axb = SubPlot(fig,(1,3,2),X,Y)
    # Plot ACF as function of 2-d
    axc = fig.add_subplot(1,3,3,projection='3d',elev=30,azim=-115)
    d = int(n/magic.N_acf)
    R = tuple(range(0,n,d))
    acf_xy = gun.Cov[:n:d,:n:d]
    N,M = acf_xy.shape
    assert N == M
    X = np.arange(N)
    X, Y = np.meshgrid(X, X)
    surf = axc.plot_surface(X, Y, acf_xy/1e19, rstride=1, cstride=1,
                   cmap=mpl.cm.jet, linewidth=0, antialiased=False)
    xticks = np.array((0,N-1))
    xvals = np.array((gun.x[0], gun.x[R[-1]]))
    axc.set_xticks(xticks)
    axc.set_xticklabels(['%3.1f'%x for x in xvals])
    axc.set_yticks(xticks)
    axc.set_yticklabels(['%3.1f'%x for x in xvals])
    axc.set_xlabel(r'$x$')
    axc.set_ylabel(r'$\chi$')
    axc.set_zlabel(r'$\Sigma/10^{19}$')
    fig.subplots_adjust(bottom=0.15) # Make more space for label
    fig.subplots_adjust(wspace=0.3)  # Make more space for label
    return fig
plot_dict['plot_moments'] = plot_moments

def plot_allowed_tp2(gun, plt):
    '''Plot upper and lower bounds of allowed values of f(x[2]) as
    functions of f(x[0]) and f(x[1]).  Lower and upper bounds given
    f(x[0])=f[i] and f(x[1])=f[j] are gun.envelope[i,j,0]
    gun.envelope[i,j,1] respectively.  The values are np.nan if (i,j)
    is not an allowed pair.

    '''
    #s = int(magic.N_f/magic.N_f_plot)
    X = gun.f/gun.s[0] # Centers of intervals in f at xi
    Y = X
    Zu = gun.envelope[:,:,0]/gun.s[2]
    Zl = gun.envelope[:,:,1]/gun.s[2]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d', azim=-150, elev=5)
    ax.set_xlabel(r'$g(y_0)$')
    ax.set_ylabel(r'$g(y_1)$')
    ax.set_zlabel(r'$g(y_2)$')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Zl.T, rstride=1, cstride=1, cmap=mpl.cm.gray,
                           alpha=0.8, vmax=1.25, vmin=.75, linewidth=0.15)
    surf = ax.plot_surface(X, Y, Zu.T, rstride=1, cstride=1, cmap=mpl.cm.gray,
                           alpha=1, vmax=1.25, vmin=.75, linewidth=0.15)
    ticks = np.arange(0.98, 1.021, 0.02)
    labels = ['%4.2f'%f for f in (ticks-1)] # Good enough approx for log
    ax.set_xticks(ticks, minor=False)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks, minor=False)
    ax.set_yticklabels(labels)
    ax.set_zticks(ticks, minor=False)
    ax.set_zticklabels(labels)
    return fig
plot_dict['plot_allowed_tp2'] = plot_allowed_tp2


def p_stat(gun, plt):
    '''Plot the stationary distribution as a surface.

    '''
    X = gun.f/gun.s[0] # Centers of intervals in f at xi
    n_f = len(gun.f)
    Y = X
    print('n_f=%d, P_stat.shape=%s'%(n_f, gun.P_stat.shape))
    Zu = gun.P_stat.reshape((n_f,n_f))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d', azim=-120, elev=90)
    ax.set_xlabel(r'$r(y_0)$')
    ax.set_ylabel(r'$r(y_1)$')
    ax.set_zlabel(r'$P$')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Zu, rstride=1, cstride=1, cmap=mpl.cm.gray,
                           linewidth=0.15)
    return fig
plot_dict['p_stat'] = p_stat

def plot_mean(gun, plt):
    ''' This is function is an unreliable hack that I wrote to get
    QM_2_12.pdf to build after I changed the way I calculate the
    probabilities.
    '''
    fig = plt.figure(figsize=(7,14))
    r = float(gun.mean)/gun.s[0] # gun.mean is 1x1 if read from
                                 # stash_stationary.mat
    N = len(gun.x)
    yticks = np.arange(0.98,1.03,0.02) # Need formatting outside SubPlot
    X = axis(data=gun.x,magnitude=False,label='x')
    Y = axis(data=r*np.ones(N),magnitude=False,ticks=yticks,
             label='\mu/f')
    axa = SubPlot(fig,(2,1,1),X,Y)
    axa.set_ylim(0.98,1.03)
    axa.set_yticklabels(['%4.2f'%x for x in yticks])
    xc, f = uniform_acf(gun)
    axc = fig.add_subplot(2,1,2,projection='3d',elev=30,azim=-115)
    norm = gun.s**.5
    acf_xy = calc.acf2d(f(xc[:-1]),gun.eos(xc[:-1]))/norm
    acf_xyT = acf_xy.T
    acf_xyT /= norm
    N,M = acf_xy.shape
    X = np.arange(N)
    Y = np.arange(M)
    X, Y = np.meshgrid(X, Y)
    surf = axc.plot_surface(X, Y, acf_xy/1e19, rstride=1, cstride=1,
                   cmap=mpl.cm.jet, linewidth=0, antialiased=False)
    f = scipy.interpolate.interp1d([.4,4],[0,N],kind='linear')
    xticks = np.arange(1,4.5,1)
    axc.set_xticks([f(x) for x in xticks])
    axc.set_xticklabels(['%d'%int(x) for x in xticks])
    axc.set_yticks([f(x) for x in xticks])
    axc.set_yticklabels(['%d'%int(x) for x in xticks])
    axc.set_xlabel(r'$x$')
    axc.set_ylabel(r'$y$')
    return fig
plot_dict['plot_mean'] = plot_mean

def plot_q0_1(gun, plt):
    ''' Plots q_0 and q_1.
    '''
    fig = plt.figure(figsize=(5,7))
    X = axis(data=gun.x_c,label='x', units='cm')
    ax = SubPlot(fig,(2,1,2),X,
           axis(data=gun.q_1(),label='q_1',magnitude=-16,units='s/(dyn\ cm)'))
    ax.set_ylabel(ax.get_ylabel(),labelpad=-5) # Shift by 5 points
    X.label=False
    X.tick_label_flag=False
    ax = SubPlot(fig, (2,1,1), X, axis(data=gun.q_0(),label='q_0',
                      ticks=np.arange(0,2.1,1),units='(erg/(dyn\ cm))'))
    fig.subplots_adjust(left=0.15) # Make more space for label
    return fig
plot_dict['plot_q0_1'] = plot_q0_1

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
