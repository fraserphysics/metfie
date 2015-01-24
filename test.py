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
    ''' Checks that == accomplishes
    '''
    for name in 'd n_g n_h g_step h_step n_pairs n_states'.split():
        assert getattr(A, name) == getattr(B, name), \
            'A.{0} != B.{0}'.format(name)

def check_arrays(A,B):
    '''Assert G2state G2h_list match for A and B
    '''
    import numpy.testing as npt
    for a,b,name in ((getattr(A, name), getattr(B, name), name) for name in
                     'G2state G2h_list'.split()):
        npt.assert_allclose(a,b,
            err_msg='{0:s} fails to match'.format(name))

def check_bounds(A,B):
    '''Assert bounds_a and bounds_b match for A and B
    '''
    import numpy.testing as npt
    for i in range(A.n_states):
        for a,b,name in ((getattr(A, name), getattr(B, name), name) for name in
                    'bounds_a bounds_b'.split()):
            npt.assert_allclose(a[i],b[i],
                    err_msg='{0:s}.[{1:d}] fails to match'.format(name,i))

def check_state_list(A,B):
    import numpy.testing as npt
    for i in range(A.n_states):
        assert A.state_list[i] == B.state_list[i] # These are tuples

def check_symmetry(A):
    import numpy as np
    import numpy.testing as npt
    l_vec = np.zeros(A.n_states)
    r_vec = np.zeros(A.n_states)
    for i in range(A.n_states):
        j = A.conj(i)
        assert A.conj(j) == i
        
        l_vec[i] = 1.0
        r_vec[j] = 1.0
        npt.assert_array_equal(l_vec, A.symmetry(r_vec), err_msg='''
conj inconsistent with symmetry.  i={} j={}
            '''.format(i,j))
        L = A.matvec(l_vec)
        R = A.rmatvec(r_vec)
        l_vec[i] = 0
        r_vec[j] = 0
        for k in np.where(L > .1)[0]:
            assert R[A.conj(k)] > .1
        for k in np.where(R > .1)[0]:
            l = A.conj(k)
            if L[l] < .1 :
                strings = ('({0:5.1f}, {1:5.1f})'.format(x[0],x[1]) for x in
                    (A.state_list[m][:2] for m in (i,j,k,l)))
                print('''
                Symmetry failure (states as (g,h))
    {0:15s} * M.T -> {2:s} but not
M * {1:15s}       -> {3:s}'''.format(*strings))
                raise RuntimeError
    return

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
    #  .state_list
    import numpy.linalg as LA
    import first_c
    import first
    import numpy.testing as npt
    d = 50
    d_g = 4
    d_h = 4
    name='%d_%d_%d'%(d, d_h, d_g)
    try:
        A = first.read_LO_step(name)
        A.pairs()
    except:
        A = make_A(d, d_h, d_g, name)
    B = first_c.LO_step(A.d, A.g_step, A.h_step)
    B.power(small=1.0e-9)
    check_simple(A,B)
    check_arrays(A,B)
    check_bounds(A,B)
    check_state_list(A,B)
    #check_symmetry(A)
    #check_symmetry(B)
    check_images(A,B)
    print('iterations:{0:d}, e_value:{1:e}'.format(B.iterations, B.eigenvalue))
    print('norm first_c:{0:f}, first:{1:f}'.format(
        LA.norm(B.eigenvector),LA.norm(A.eigenvector)))
    npt.assert_allclose(B.eigenvector, A.eigenvector, rtol=0.1, atol=1e-3)
    
def make_A(d=48, g_step=4, h_step=4, name='48_4_4'):
    import first
    A = first.LO_step(d, g_step, h_step)
    A.power(small=1e-9)
    print('iterations:{0:d}, e_value={1:e}'.format(A.iterations, A.eigenvalue))
    if name != None:
        A.archive(name)
    return A

if __name__ == "__main__":
    from time import time
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
