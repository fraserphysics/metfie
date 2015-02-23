'''test.py Make sure that first.py and first_c.pyx make the same LO

'''
import sys
def main(argv=None):
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser( description=
        'Time calculation of eigenfunction of the integral equation')
    parser.add_argument('--python', action='store_true', help=
                        'build operator without cython')
    parser.add_argument('--no_power', action='store_true', help=
                        'skip cython eigenvalue calculation')
    parser.add_argument('--u', type=float, default=(2.0e-5),
                       help='log fractional deviation')
    parser.add_argument('--dy', type=float, default=3.2e-4,
                       help='y_1-y_0 = log(x_1/x_0)')
    parser.add_argument('--n_g', type=int, default=200,
                       help='number of integration elements in value')
    parser.add_argument('--n_h', type=int, default=200, help=
'number of integration elements in slope.  Require n_h > 192 u/(dy^2).')
    parser.add_argument('--archive', type=str, default='test', help=
'File for storing LO_step instance.  Use "no" to skip.')
    args = parser.parse_args(argv)

    import numpy as np
    import first
    import first_c
    from time import time
    
    d_g = 2*args.u/args.n_g
    h_lim = np.sqrt(48*args.u)
    d_h = 2*h_lim/args.n_h
    
    tol = 5e-6
    maxiter = 1000

    if args.archive != 'no':
        try:
            old_LO = first.read_LO_step(args.archive)
        except:
            print('Failed to read %s'%args.archive)
    print('\n')
    t0 = time()

    LO_cython = first_c.LO_step( args.u, args.dy, d_g, d_h)
    print('cython: n_states=%d, n_pairs=%d'%(
        LO_cython.n_states,LO_cython.n_pairs))
    t1 = time()
    
    if args.no_power:
        t2 = t1
    else:
        LO_cython.power(n_iter=maxiter, small=tol, verbose=True)
        t2 = time()
        if args.archive != 'no':
            LO_cython.archive(args.archive)

    if args.python:
        LO_python = first.LO_step( args.u, args.dy, d_g, d_h)
        print('python: n_states=%d, n_pairs=%d'%(
            LO_python.n_states,LO_python.n_pairs))
        t3 = time()
    print('''
    Time in seconds    Task
    %5.1f              cython build operator
    %5.1f              cython power iterations'''%((t1-t0), (t2-t1)))
    if args.python:
        print('''
    %5.1f              python build operator'''%(t3-t2))

    if 'old_LO' in vars():
        from numpy.linalg import norm
        print('norm(old-new)=%e'%(norm(
            old_LO.eigenvector-LO_cython.eigenvector)))
    return 0

def check_simple(A,B):
    ''' Checks that are done with ==
    '''
    for name in 'd n_g n_h g_step h_step n_states n_pairs'.split():
        a,b = (getattr(x, name) for x in (A,B))
        assert a == b, \
            '{1}=A.{0} != B.{0}={2}'.format(name, a, b)

def check_bounds(A,B):
    '''Assert bounds_a and bounds_b match for A and B
    '''
    import numpy.testing as npt
    for i in range(A.n_states):
        for name in 'bounds_a bounds_b'.split():
            a,b = (getattr(x,name) for x in (A,B))
            npt.assert_allclose(a[i],b[i],
                    err_msg='{0:s}.[{1:d}] fails to match'.format(name,i))

def check_state_list(A,B):
    import numpy.testing as npt
    npt.assert_array_equal( A.state_list, B.state_list)
    npt.assert_array_equal( A.G2state, B.G2state)

def check_images(A,B):
    import numpy as np
    import numpy.testing as npt
    vec = np.zeros(A.n_states)
    for i in range(A.n_states):
        vec[i] = 1.0
        for a, b, hand in ((A.matvec(vec), B.matvec(vec), 'L'),
                           (A.rmatvec(vec), B.rmatvec(vec), 'R')):
            npt.assert_allclose(a, b, err_msg='''
{1:s} multiply fails to match for i={0:d}
'''.format(i,hand))
        vec[i] = 0
    return

def test_first_c():
    ''' See if first.py and first_c.pyx make the same operator
    '''
    import numpy.linalg as LA
    import first_c
    import first
    import numpy.testing as npt

    # Get an operator from first.py
    A = get_A()
    # Make an operator from first_c.pyx with same parameters
    B = first_c.LO_step(A.d, A.g_step, A.h_step)
    
    check_state_list(A,B)
    check_bounds(A,B)
    check_simple(A,B)
    check_images(A,B)
    B.power(small=1.0e-9)
    npt.assert_allclose(B.eigenvector, A.eigenvector)
    
def make_A(d, g_step, h_step, name=None):
    import first
    A = first.LO_step(d, g_step, h_step)
    A.power(small=1e-9)
    if name != None:
        A.archive(name)
    return A

def get_A():
    d, d_g, d_h = 50, 4, 4 # Operator parameters
    name='%d_%d_%d'%(d, d_h, d_g)
    try:
        A = first.read_LO_step(name)
        A.pairs()
    except:
        A = make_A(d, d_h, d_g, name)
    return A

def check_symmetry(A):
    ''' Check to ensure that if
    (x,y)  in A*(g,h) then 
    (x,-y) in (g,-h)*A
    '''
    import numpy as np
    import numpy.testing as npt

    # Check A.conj
    for i in range(A.n_states): # state[i] = (g,h)
        j = A.conj(i)           # state[j] = (g,-h)
        assert A.conj(j) == i

    l_vec = np.zeros(A.n_states)
    r_vec = np.zeros(A.n_states)
    for i in range(A.n_states):
        j = A.conj(i)  # (g,h) in state[i] <=> (g,-h) in state[j]
        
        l_vec[i] = 1.0
        L = A.matvec(l_vec)     # Operate to left
        r_vec[j] = 1.0
        R = A.rmatvec(r_vec)    # Operate to right
        
        # Check A.symmetry()
        npt.assert_array_equal(l_vec, A.symmetry(r_vec), err_msg=
        'conj inconsistent with symmetry.  i={} j={}\n'.format(i,j))

        # Check matvec(x) = S(rmatvec(S(x))), where S is A.symmetry
        npt.assert_allclose(L, A.symmetry(R),
            err_msg='''
matvec(%s)  = %s
rmatvec(%s) = %s'''%(l_vec, L, r_vec, R))
        l_vec[i] = 0
        r_vec[j] = 0
    return

def check_grid(A, d, d_g, d_h):
    ''' Ensure that (g=A.d, h=0) is close to a grid point
    '''
    from scipy.optimize import bisect
    assert A.d == d
    assert A.g_step == d_g
    assert A.h_step == d_h

    # Locate the h boundary that is closest to 0
    h_a = d_h/2
    H_a, H_b = (A.h2H(x) for x in (h_a, -h_a))
    def f(h, H_a, H_b, A):
        return H_a + H_b - 2*A.h2H(h)
    h_0 = bisect(f, -h_a, h_a, args=(H_a, H_b, A))
    assert abs(h_0) < 1e-10

    # Locate the g boundary that is closest to d - d_g
    g_a = A.d - 0.5*d_g
    g_b = A.d - 1.5*d_g
    G_a, G_b = (A.g2G(x) for x in (g_a, g_b))
    def f(g, G_a, G_b, A):
        return G_a + G_b - 2*A.g2G(g)
    g_0 = bisect(f, g_b, g_a, args=(G_a, G_b, A))
    assert abs(g_0 + d_g - A.d) < 1e-10

def test_shape():
    import first
    d, d_g, d_h = 39, 40, 40 # Operator parameters
    A = first.LO_step(d, d_g, d_h)
    check_grid(A, d, d_g, d_h)
    check_symmetry(A)

def test_archive():
    import pickle
    import os
    import glob
    from first import Archive
    from first_c import LO_step
    test_dir = 'test'
    
    for name in glob.glob(test_dir+'/*'): # rm test/*
        os.remove(name)
        
    d, h_step, g_step = 100.0, 4.0, 4.0
    key = (d, h_step, g_step)
    A,x = Archive(LO_step, dir_name=test_dir).get(*key, make_pairs=True)
    B,x = Archive(LO_step, dir_name=test_dir).get(*key, make_pairs=True)
    check_state_list(A,B)
    check_bounds(A,B)
    check_simple(A,B)
                    
if __name__ == "__main__":
    from time import time
    test_archive()
    sys.exit(0)
    test_shape()
    t_1 = time()
    test_first_c()
    t_2 = time()
    print('{0:f} seconds'.format(t_2-t_1))
    sys.exit(0)
    rv = main()
    sys.exit(rv)

#---------------
# Local Variables:
# mode: python
# End:
