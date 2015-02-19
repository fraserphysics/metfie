'''converge.py study convergence of numerical estimates of the
conditional distribution given an initial point.  In particular for
studying convergence as dy gets small.  order Markov integral
operator.

'''
import sys
import numpy as np
import matplotlib as mpl
from first_c import LO_step
#from first import LO_step
from first import Archive
def main(argv=None):
    '''For looking at the probability distribution over the image of a
    point under A
    '''
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=float, default=100,
        help='Max g')
    parser.add_argument('--d_g', type=float, default=4,
        help='element size')
    parser.add_argument('--d_h', type=float, default=4,
        help='element size')
    parser.add_argument('--iterations', type=int, default=2,
        help='Apply operator n times and scale d, d_h and d_g')
    parser.add_argument(
        '--point', type=float, nargs=2, default=(0.0, 0.0),
        help='Analyze the image of this point (h and g as fractions of maxima')
    parser.add_argument('--out', type=str, default=None,
        help="Write plot to this file")
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

    archive = Archive(LO_step)
    # Initialize operator
    d = args.d*args.iterations**2
    A,read = archive.create( d, args.d_h, args.d_g)
    if not read:
        A.power(small=1.0e-8)
        print('{0:d} iterations'.format(A.iterations))

    # Create unit vector with one component specified by args.point
    h_,g_ = args.point
    g = A.d * g_
    h_max = A.h_lim(g)
    h = h_max * h_
    # Get integer indices of cell and set corresponding component
    H = A.h2H(h)
    G = A.g2G(g)
    h_0,g_0 = A.H2h(H),A.G2g(G) # Actual point used
    point_index = A.state_dict[(H,G)]
    v = np.zeros(A.n_states)
    v[point_index] = 1.0

    h_1,g_1 = h_0, g_0
    for i in range(args.iterations):
        v = A.matvec(v)
        v /= v.max()
        h_1,g_1 = A.affine(h_1,g_1)
    if not read:
        A.archive({(H,G):v})
    h_3 = A.h_lim(g_1)
    g_3 = g_1
    # z_1: apex of pie slice, z_3: lower right corner

    G_t, H_t = A.g2G(g_3)-2, A.h2H(h_3)-2
    while v[A.state_dict[(H_t,G_t)]] == 0.0 :
        assert G_t < A.n_g
        G_t += 1
    G_3 = G_1 = G_t
    assert abs(A.g2G(g_3) - G_3) < 3
    
    edge = []
    for G in range(G_1, A.n_g):
        g = A.G2g(G)
        H_G = int(np.ceil(A.h_lim(g)/A.h_step) + A.n_h/2 -1)
        h = A.H2h(H_G)
        v_ = v[A.state_dict[H_G,G]]
        if v_ == 0:
            d_g = (g-g_1)/args.iterations
            d_h = h-h_1
            assert abs(d_g-d_h) < 2*(A.g_step + A.h_step)
            break
        slope = (g-g_1)/(h-h_1)
        edge.append( (g, h, slope, v_))
    edge = np.array(edge)

    line = []
    dz = np.array((1.0, args.iterations/2.0))*A.h_step/5.0
    z = np.array((h_1, g_1),np.float64)
    HG = lambda z: (A.h2H(z[0]), A.g2G(z[1]))
    while HG(z) in A.state_dict:
        assert len(line) < A.n_h*5
        i = A.state_dict[HG(z)]
        ev_i = A.eigenvector[i]
        line.append((v[i], ev_i, v[i]*ev_i))
        z += dz
    line = np.array(line,np.float64)
    line /= line.max(axis=0)
    x = np.linspace(0,1,len(line))
    
    fig = plt.figure(figsize=(6,7))
    ax = fig.add_subplot(4,1,1)
    ax.plot(edge[:,2]/args.iterations, np.log10(edge[:,3]))
    ax.set_ylim(-10,0)
    ax = fig.add_subplot(4,1,2)
    ax.plot(x, np.log10(line[:,0]))
    ax = fig.add_subplot(4,1,3)
    ax.plot(x, np.log10(line[:,1]))
    ax = fig.add_subplot(4,1,4)
    ax.plot(x, np.log10(line[:,2]))
    ax.set_ylim(-10,0)
    if args.out == None:
        plt.show()
    else:
        fig.savefig( open(args.out, 'wb'), format='pdf')
    return 0

if __name__ == "__main__":
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
