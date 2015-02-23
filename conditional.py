'''conditional.py fit a 2-d Gaussian to the conditional distribution
given an inital point after a specified number of iterations.

'''
import sys
import numpy as np

def fit(args, archive, iterations):
    from numpy import linalg as LA
    # Initialize operator
    d = args.d*iterations**2
    h_,g_ = args.point
    A,image_dict = archive.get( d, args.d_h, args.d_g,
                                [(h_, g_, iterations)])
    h_max = A.h_lim(-d)
    keys = list(image_dict.keys())
    assert len(keys) == 1
    
    v = image_dict[keys[0]]*A.eigenvector
    indices = np.where(v>0)
    p = v[indices]
    p /= p.sum() # discrete probabilities of cells
    ev = lambda x : (x*p).sum(axis=-1)
    H,G = (A.state_list[indices]).T
    h = np.array([A.H2h(x) for x in H])/h_max
    g = np.array([A.G2g(x) for x in G])/d
    hg = np.array([h,g])
    mu = ev(hg)
    d = (hg.T-mu)
    sigma = np.dot(d.T*p,d)
    vals,vecs = LA.eigh(sigma)
    assert mu.shape == (2,)
    assert sigma.shape == (2,2)
    print('mu={2}\nvals={0}\nvecs={1}\n'.format(vals,vecs,mu))
    theta_mu = np.arctan(mu[1]/mu[0])*(180/np.pi)
    theta_sigma = np.arctan(vecs[1,1]/vecs[1,0])*(180/np.pi)
    return mu, vals, theta_sigma, A
    #*args.d_h*args.d_g # Samples of pdf
def main(argv=None):
    '''For characterizing the probability distribution over the image of a
    point under A
    '''
    import matplotlib as mpl
    from first_c import LO_step
    #from first import LO_step
    from first import Archive
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
    parser.add_argument('--iterations', nargs='*', type=int, default=(2,),
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
    n_iterations = len(args.iterations)
    mu = np.empty((n_iterations,2))
    theta_sigma = np.empty((n_iterations,))
    vals = np.empty((n_iterations,2))
    for i, iters in enumerate(args.iterations):
        mu[i,:], vals[i,:], theta_sigma[i], A = fit(args, archive, iters)

    iterations = np.array(args.iterations)
    fig = plt.figure(figsize=(6,15))
    ax = fig.add_subplot(5,1,1)
    ax.plot(iterations, mu[:,0])
    ax = fig.add_subplot(5,1,2)
    ax.plot(iterations, mu[:,1])
    ax = fig.add_subplot(5,1,3)
    ax.plot(iterations, vals[:,0]/vals[:,1])
    ax = fig.add_subplot(5,1,4)
    ax.plot(iterations, vals[:,1])
    ax = fig.add_subplot(5,1,5)
    ax.plot(iterations, theta_sigma)
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
