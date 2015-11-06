"""stick.py: Class for rate stick experiment.

"""
import numpy as np

from eos import Go

# From description of Shot1 in Pemberton's document
pemberton = Go(
    densities=np.array(  # g/cc, page4
        [1.835, 1.8358, 1.8353, 1.8356, 1.8356, 1.8345]),
    positions=np.array(  # mm, page 8
        [25.9, 50.74, 75.98, 101.8, 125.91, 152.04, 177.61]),
    x_dev=0.02,          # mm, position measurement uncertainty (my guess)
    t_dev=1.0e-9,        # sec, time measurement uncertainty (my guess)
    velocity=8.8132,     # Km/sec, page 9
    )
import fit
class Stick:
    '''Represents an imagined experiment and simulation that creates data,
    t, the "measured" times.

    See the subsection "A Rate Stick" in the section "Experiments" of
    notes.pdf.

    Methods:

    fit_v()       Return detonation velocity

    fit_D()       Calculates derivative of t wrt to eos in
                  terms of spline coefficients
    
    compare(t,c)  Return d_ep/d_c, ep, Sigma_inv

    log_like(...) Calculate the exponent of the likelihood function
    
    '''
    from eos import Spline_eos
    def __init__(
            self,            # Stick instance
            eos,             # Pressure(specific volume)
            x=pemberton.positions,
            vol_0=1/pemberton.densities.mean(),
            x_dev=pemberton.x_dev,
            t_dev=pemberton.t_dev,
            ):
        self.eos = eos
        self.x=x
        self.vol_0=vol_0
        self.x_dev=x_dev
        self.t_dev=t_dev
        return
    def fit_v(         # Stick instance
            self
            ):
        '''Calculate deonation velocity by finding a Rayleigh line
        that touches and is tangent to the eos at one point.  See
        Eqn 2.3 on page 17 of Fickett and Davis.
              '''
        CJ_velocity, CJ_volume, CJ_pressure = self.eos.CJ(self.vol_0)
        return CJ_velocity
    def fit_D(
            self,                   # Stick instance
            fraction=fit.magic.D_frac,  # Finite difference fraction
            ):
        '''Calculate dt/df in terms of spline coefficients and return,
        '''
        # Spline.new_c(c) does not modify Spline.  It returns a copy
        # of Spline with the coefficients c.
        eos_nom = self.eos
        c_f_nom = eos_nom.get_c()    # Nominal coefficients for eos
        v_nom = self.fit_v()         # Nominal deonation velocity
        D = np.empty((len(self.x),len(c_f_nom))) # To be dt/df matrix
        for i in range(len(c_f_nom)):
            c_f = c_f_nom.copy()
            # Next set size of finite difference for derivative approximation
            d_f = float(c_f[i]*fraction)
            c_f[i] += d_f
            # Next calculate detonation velocity for a modified eos
            self.eos = eos_nom.new_c(c_f)
            v_i = self.fit_v()
            for j,x in enumerate(self.x):
                D[j,i] = (x/v_i - x/v_nom)/d_f
        self.eos = eos_nom
        return D
    def compare(
            self,         # Stick instance
            t,            # Vector of experimentally measured times
            c=None
            ):
        ''' Calculate:
        ep = prediction - data
        D = d prediction / d c_eos
        '''
        if type(c) != type(None):
            self.eos = self.eos.new_c(c)
        v = self.fit_v()  # detonation velocity
        ep = t -  self.x/v
        D = self.fit_D()
        t_var = np.ones(len(t))*(self.t_dev**2)
        t_var += (self.x_dev/v)**2
        return (D, ep, np.diag(1.0/t_var))
    def log_like(
            self, # Stick instance
            D,
            ep,
            Sigma_inv,
            ):
        ''' Log likelihood of model m for t_exp with ep = t_exp - t(m)

            L(m) = log(p(t|m)) = \sum_i \frac{-1}{2} ep^T Sigma_inv ep

        '''
        return -np.dot(ep, np.dot(Sigma_inv, ep))/2

def data(eos=None):
    '''Make "experimental" data from stick
    '''
    if eos==None:
        from eos import Experiment
        eos = Experiment()
    vel_CJ, vol_CJ, p_CJ = eos.CJ(1/pemberton.densities.mean())
    stick = Stick(eos)
    # Make simulated measurements
    return stick.x/vel_CJ

# Test functions
close = lambda a,b: a*(1-1e-7) < b < a*(1+1e-7)
import numpy.testing as nt
def make_stick():
    from eos import Nominal, Spline_eos
    return Stick(Spline_eos(Nominal()))
def test_fit_v():
    assert close(make_stick().fit_v(), 2.8593225963e+05)
    return 0
def test_fit_D():
    D = make_stick().fit_D()
    assert D.shape == (7,50)
    assert close(-D[-1,10], 5.32687844706e-15)
    return 0
def test_compare():
    t = data()
    D, ep, SI = make_stick().compare(t)
    assert SI.shape == (7,7)
    assert close(SI[0,0], 2.04351375e+14)
    assert close(ep[-1], 2.569505832e-05),'ep[-1]={0:.9e}'.format(ep[-1])
    return 0
def test_log_like():
    t = data()
    stick = make_stick()
    rv = stick.compare(t)
    ll = stick.log_like(*rv)
    assert close(-ll,1.922445608e+05),'-ll={0:.9e}'.format(-ll)
    return 0
def test_data():
    ref = np.array([9.43278801e-05, 1.84795237e-04, 2.76719395e-04, 3.70755915e-04,
   4.58564609e-04, 5.53730150e-04, 6.46856169e-04])
    t = data()
    nt.assert_allclose(t,ref)
    return 0
def test():
    for name,value in globals().items():
        if not name.startswith('test_'):
            continue
        if value() == 0:
            print('{0} passed'.format(name))
        else:
            print('\nFAILED            {0}\n'.format(name))
    return 0
    
def work():
    ''' This code for debugging stuff will change often
    '''
    import matplotlib.pyplot as plt
    from eos import Nominal, Spline_eos
    from fit import Opt
    vt = data()
    nom = Spline_eos(Nominal(),precondition=True)
    stick = Stick(nom)
    opt = Opt(
        nom,
        {'stick':stick},
        {'stick':vt})
    cs,costs = opt.fit(max_iter=10)
    D, ep, Sigma_inv = stick.compare(vt, cs[-1])
    info = np.dot(D.T, np.dot(Sigma_inv, D))
    _vals,_vecs = np.linalg.eigh(info)
    _vals = np.maximum(_vals, 0)
    i = np.argsort(_vals)[-1::-1]
    vals = _vals[i]
    vecs = _vecs[i]
    n_vals = len(np.where(vals > vals[0]*1e-20)[0])
    n_vecs = len(np.where(vals > vals[0]*1e-2)[0])
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    costs = np.array(costs)
    x = range(len(costs))
    c_min = costs.min()
    c_max = costs.max()
    if c_min > 0:
        y = costs
        ylabel = r'$\log_{e}(p(y|c_i))$'
    else:
        offset = (c_max-c_min)*1e-5 -c_min
        y = costs + offset
        ylabel = r'$\log_{e}(p(y|\hat c_i)) + {0:.3e}$'.format(offset)
    ax.semilogy(x,y)
    ax.semilogy(x,y, 'kx')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'$i$')
    ax.set_xticks(x)
    v = np.linspace(.2,50,500)
    ax = fig.add_subplot(2,2,2)
    ax.loglog(v,nom(v))
    ax = fig.add_subplot(2,2,3)
    ax.semilogy(range(n_vals), vals[:n_vals])
    ax = fig.add_subplot(2,2,4)
    for vec in vecs[:n_vecs]:
        f = nom.new_c(vec)
        ax.semilogx(v,f(v))
    plt.show()
    vecs = _vecs[i]
#    for j,v in enumerate(vals):
#        print('v[{0:2d}] = {1:.3e}'.format(j,v))
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) >1 and sys.argv[1] == 'test':
        rv = test()
        sys.exit(rv)
    if len(sys.argv) >1 and sys.argv[1] == 'work':
        sys.exit(work())
    main()

#---------------
# Local Variables:
# mode: python
# End:
