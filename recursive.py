"""recursive.py:
Plots for explaining the recursively augmented bounds
"""

import calc    # Code that does the gun stuff
import sys
import matplotlib as mpl
import numpy as np
params = {'axes.labelsize': 18,     # Plotting parameters for latex
          'text.fontsize': 15,
          'legend.fontsize': 15,
          'text.usetex': True,
          'font.family':'serif',
          'font.serif':'Computer Modern Roman',
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
mpl.rcParams.update(params)
def main(argv=None):
    DEBUG = False
    if DEBUG:
        mpl.rcParams['text.usetex'] = False
    else:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use

    if argv is None:
        argv = sys.argv[1:]

    C = 2.56e10
    xi = 0.9
    xf = 1.1111111111
    m = 10.0
    Nx = 31
    dev = 0.05
    gun = calc.GUN(C, xi, xf, m, Nx, dev)
    
    Nf = 5
    if DEBUG:
        ij, I = gun.ijk(Nf)
        i = 2
        for j in range(Nf):
            if ij[i,j] != None:
                print('ij[%d, %d] allows k in %s'%(
                    i,j,range(ij[i,j][0], ij[i,j][1])))
    # Several lines copied/edited from calc.GUN.stationary
    g_L,g_U = (np.log(r) for r in gun.LU(1.0))
    step = (g_U-g_L)/Nf
    S_g = np.arange(g_L,g_U+step/3,step)
    S_r = np.exp(S_g) # Sampled ratio (f/\tilde f) points
    L = S_r[:-1]      # Factors for lower edges of intervals
    U = S_r[1:]       # Factors for upper edges of intervals
    C = (U+L)/2       # Factors for centers of intervals
    assert len(C) == Nf
    # Next, map from factors to EOS values.  Will use U and L for
    # adjacency and C for moments.
    L = np.outer(gun.s,L)
    C = np.outer(gun.s,C)
    U = np.outer(gun.s,U)
    x_c = gun.x_c
    def grid(ax, i, h=None):
        '''Plot a vertical line at x_c[i] with Nf-1 ticks
        '''
        Y = list(L[i])+[U[i,-1]]
        x = x_c[i]
        ax.plot((Nf+1)*[x], Y ,'k')
        for y in Y:
            ax.plot([x-0.0001,x+0.0001],[y,y],'k')
        if h != None:
            ax.plot(2*[x], Y[h:h+2] ,'w',linewidth=1)
    def tan_a(i, h, F):
        ''' Return y values that give first tangent to U[:,-1] that
        goes through (x_c[i], F[i,h])
        '''
        go = gun.recursive(i, F[i,h])
        return F[i,h] + go.df_a*(x_c - x_c[i])
    def tan_b(i, h, F):
        ''' Return y values that give second tangent to U[:,-1] that
        goes through (x_c[i], F[i,h])
        '''
        go = gun.recursive(i, F[i,h])
        return F[i,h] + go.df_b*(x_c - x_c[i])

    fig4 = plt.figure(figsize=(6,6))
    ax = fig4.add_subplot(1,1,1)
    ax.plot(x_c, L[:,0],'b')
    ax.plot(x_c, U[:,-1],'b')
    i = 15
    j = i+1
    k = i+2
    hi = 2
    ax.plot([x_c[i]], [C[i,hi]],'k.')
    ax.plot(x_c, tan_a(i,hi,C),'r')
    ax.plot(x_c, tan_b(i,hi,C),'g')

    fig5 = plt.figure(figsize=(6,6))
    ax = fig5.add_subplot(1,1,1)
    ax.plot(x_c, L[:,0],'b')
    ax.plot(x_c, U[:,-1],'b')
    grid(ax, i, hi)
    grid(ax, j)
    grid(ax, k)
    ax.plot(x_c[i:], tan_a(i,hi,L)[i:],'r')
    ax.plot(x_c[i:], tan_b(i,hi,U)[i:],'g')
    ax.set_xlim(0.994, 1.015)
    ax.set_ylim(2.35e10, 2.7e10)

    def plotj(hj):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x_c, L[:,0],'b')
        ax.plot(x_c, U[:,-1],'b')
        grid(ax, i, hi)
        grid(ax, j, hj)
        grid(ax, k)
        ax.plot(x_c[j:], tan_a(j,hj,L)[j:],'r')
        ax.plot(x_c[j:], tan_b(j,hj,U)[j:],'g')
        s = (L[j,hj] - U[i,hi])/(x_c[j] - x_c[i])
        ax.plot(x_c[i:], U[i,hi] + s*(x_c[i:] - x_c[i]),'c')
        ax.plot(x_c[i:], tan_a(i,hi,L)[i:],'r')
        ax.plot(x_c[i:], tan_b(i,hi,U)[i:],'g')
        bottom = max(tan_a(i,hi,L)[k], tan_a(j,hj,L)[k],
                     U[i,hi] + s*(x_c[k] - x_c[i]), L[k,0])
        top = min(U[k,-1], tan_b(j,hj,U)[k], tan_b(i,hi,U)[k])
        ax.plot(2*[x_c[k]], [bottom,top] ,'m',linewidth=1)
        ax.set_xlim(0.994, 1.015)
        ax.set_ylim(2.35e10, 2.7e10)
        return fig
    fig1 = plotj(1)
    fig2 = plotj(2)
    fig3 = plotj(3)

    if DEBUG:
        plt.show()
    else:
        for fig, name in (
            (fig1,'recursiveI'), (fig2,'recursiveII'), (fig3,'recursiveIII'),
            (fig4,'recursiveIV'),(fig5,'recursiveV')):
            fig.savefig('%s.pdf'%name, format='pdf')
if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# eval: (python-mode)
# End:
