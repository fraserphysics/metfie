from __future__ import print_function
#!/usr/local/bin/python
"""

    test_gun.py 

Collection of tests for the gun EOS model 

Revision: 1.0 - $Date: 23/07/2015$


Developers:
-----------
- Andrew Fraiser (AF) 
- Stephen Andrews (SA)

History
-------
    v. 1.0  - 
"""

"""
To Do:
    - 
"""

# =============================================================================
# Standard Python modules
# =============================================================================
import os
import sys
import unittest
import time
# =============================================================================
# Python 3 Compatibility
# =============================================================================
# =============================================================================
# External Python modules
# =============================================================================
import numpy as np
np.set_printoptions(threshold='nan', formatter = {'float':'{: 1.5e}'.format})
# =============================================================================
# Extension modules
# =============================================================================
sys.path.append(os.path.abspath('../'))

from calc import *

import numpy.testing as npt
# =============================================================================
# 
# =============================================================================

class TestBasicSpline(unittest.TestCase):
    """

    Tests the ability of the spline to model a basic function (sin x)

    """

    def setUp(self):
        self.x_s = np.linspace(0,2*np.pi,20)
        self.y = np.sin(self.x_s)
        self.f = Spline(self.x_s,self.y)
    #end

    def tearDown(self):
        pass
    #end

    def test_spline_provenance(self):
        """

        Tests the provenance of the spline 

        """

        self.assertEqual(self.f.provenance.line,  u'f = Spline(x_s,y)', 
                         msg = "Spline has the wrong provenance" )

    def test_final_coeffs(self):
        """

        Ensures that the final four coefficients are zero

        """
        npt.assert_array_equal(self.f.get_c()[-4:], np.zeros(4), 
                        err_msg = "Final four spline coefficients are not zero")        

    def test_spline_fit(self):
        """

        Ensures that the spline matches the analytic solution within 1E-15

        """
        npt.assert_allclose(self.y, self.f(self.x_s), atol=1e-15, 
                  err_msg = "Spline does not match exact solution within 1E-15")
    #end
#end 

class TestGunSpline(unittest.TestCase):
    """

    Tests the spline of the gun class 

    Tests spline approximations of f(x) and v(t).  
    Side effects: Change eos and t2v to splines.

    """

    def setUp(self):
        self.gun = GUN()

        # Instantiate the test with values for C/x^3 eos
        self.x = self.gun.x
        self.y = self.gun.eos(self.x)
        self.v_a = self.gun.x_dot(self.x)
        self.t2v_a = self.gun.set_t2v()

    #end

    def tearDown(self):
        pass
    #end

    def test_v_forall_t_lesthan_zero(self):
        """

        Tests that the velocity is zero for all times less than zero

        """
        
        ts = self.t2v_a.get_t()
        v = self.t2v_a(ts)
        for i in range(len(ts)): # Test v(t) = 0 for t<0
            self.assertTrue(ts[i] > 0 or abs(v[i]) < 1e-13, 
                    msg = "Velocity nonzero for t less than zero")
        #end
    
    def test_gun_spline_update(self):
        """

        Updates the gun spline using the same data used for its instantiation 

        ensures that spline results remains unchanged

        """  
        gun = self.gun

        ts = self.t2v_a.get_t()
        v = self.t2v_a(ts)    

        # Exercise/test GUN.set_eos_spline()
        gun.set_eos_spline(self.x,self.y)

        # Test closeness of spline to C/x^3 for f(x), v(x) and v(t)
        npt.assert_allclose(self.y, gun.eos(self.x))
        npt.assert_allclose(self.v_a, gun.x_dot(self.x))
        npt.assert_allclose(v, gun.set_t2v()(ts), atol=1e-11)
        return 0

    #end
#end 

# @unittest.skip('Computationally expensive D matrix tests skipped')
class TestSetD(unittest.TestCase):
    """

    Tests the 'D' matrix creation and generates plots 

    """

    def setUp(self):
        self.gun = GUN()
        # self.gun.G_matrix()
        # self.gun.eos(self.gun.x)
        # self.gun.set_t2v()
        x = self.gun.x 
        y = self.gun.eos(x)
        self.gun.set_eos_spline(x,y)

    def tearDown(self):
        pass 

    # @unittest.skip('Computationally expensive D matrix tests skipped')
    def test_D_shape(self):
        """

        Ensures the 'D' matrix has the correct shape


        .. note::

            Had to hard code in magic.end number 

        """
        D = self.gun.set_D(fraction=1.0e-2) # 11.6 user seconds
        

        n_f = len(self.gun.eos.get_c()) - 4
        n_v = len(self.gun.t2v.get_c()) - 4
        
        self.assertEqual(D.shape[0], n_v, 
                    msg = "D matrix has the wrong number of rows")
        self.assertEqual(D.shape[1], n_f, 
                    msg = "D matrix has the wrong number of columns")


    # @unittest.skip('Computationally expensive D matrix tests skipped')
    def test_D_matrix_components(self):
        """

        Creates a plot of the D matrix components

        if iTest is True, asserts that the D matrix matches the 'gold' data 
                    False, prints new 'gold' data for D

        """
        D = self.gun.set_D(fraction=1.0e-2) # 11.6 user seconds

        n_f = len(self.gun.eos.get_c()) - 4
        n_v = len(self.gun.t2v.get_c()) - 4

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for j in range(n_f):
            ax.plot(D[:,j])
        #end
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(
            r'$\left( \frac{\partial c_v[k]}{\partial c_f[i]} \right)$')

        if iPlot:
            plt.show()
        #end

        if iWrite:
            fig.savefig(out_dir+'D_test.pdf', format='pdf')
        #end

        # Generated by SA 29-07-15
        D_gold = [[None]] 

        if iTest:
            with open('gold_data{:}D_gold.csv'.format(os.sep),'r') as fid:               
                D_gold = np.loadtxt(fid, delimiter = ',',
                    comments = '#')
            #end
            npt.assert_array_almost_equal(D, D_gold, decimal = 6, 
                err_msg = "D matrix does not match gold data")
        else:
            with open('gold_data{:}D_gold.csv'.format(os.sep),'w') as fid:               
                np.savetxt(fid, D, delimiter = ',',
                    header = "Gold data generated by {:2s} on {:}".format(
                        usr, time.strftime("%d-%m-%Y")))
            #end
        #end
        self.assertTrue(iTest, msg = "D matrix not tested against gold data")


    #end

    def test_BD_matrix_components(self):
        """

        Tests the generation of the BD matrix 

        """

        gun = self.gun 

        v_exp, t_exp = experiment()
        self.gun.set_B_ep((v_exp,t_exp)) 
        D = self.gun.set_D(fraction=1.0e-2) # 11.6 user seconds

        B = gun.B
        D = gun.D
        BD = np.dot(B, D)
        gun.set_BD(BD)

        n_f = len(gun.eos.get_c()) - 4

        fig = plt.figure('BD', figsize=(7,6))
        ax = fig.add_subplot(1,1,1)
        for j in range(n_f):
            ax.plot(t_exp*1e6, BD[:,j])
        ax.set_xlabel(r'$t/(\mu \rm{sec})$')
        ax.set_ylabel(r'$\frac{\partial v(t)}{\partial c_f[i]} /(\rm{cm/s})$')

        if iPlot:
            plt.show() 
        #end 

        if iWrite:
            fig.savefig(out_dir+'BD_test.pdf',format='pdf')
        #end

        if iTest:
            with open('gold_data{:}BD_gold.csv'.format(os.sep),'r') as fid:               
                BD_gold = np.loadtxt(fid, delimiter = ',',
                    comments = '#')
            #end
            npt.assert_array_almost_equal(BD, BD_gold, decimal = 6, 
                err_msg = "BD matrix does not match gold data")
        else:
            with open('gold_data{:}BD_gold.csv'.format(os.sep),'w') as fid:               
                np.savetxt(fid, BD, delimiter = ',',
                    header = "Gold data generated by {:2s} on {:}".format(
                        usr, time.strftime("%d-%m-%Y")))
            #end
        #end
        self.assertTrue(iTest, msg = "BD matrix not tested against gold data")


class test_B_matrix(unittest.TestCase):
    """

    Tests the 'B' matrix creation

    .. note:: 

        The fmt test from the original source was not included 

        fmt = 'B.shape={0} != ({1},{2}) = (len(t_exp), n_f)'.format
        assert gun.B.shape == (len(t_exp), n_v),\
                               fmt(gun.B.shape, len(t_exp), n_v)

    """

    def setUp(self):
        """

        Create a GUN object and instantiate the eos spline with the initial 
        EOS function

        .. math::

            F = 2.56e10 / x^3

        """
        self.gun = GUN()
        x = self.gun.x 
        y = self.gun.eos(x)
        self.gun.set_eos_spline(x,y)
    #end

    def tearDown(self):
        pass

    def test_gun_err_size(self):
        """

        Tests that the gun erro vecotr has the correct shape

        """

        v_exp, t_exp = experiment()
        self.gun.set_B_ep((v_exp,t_exp))
        
        self.assertEqual(len(self.gun.ep), len(t_exp), 
                msg = "Length of error vecotr and t_exp do not match")
    #end

    def test_error_calculation(self):
        """

        Tests the use of the EOS function to map time to velocity 
        Tests the error calculation

        """

        gun = self.gun
        v_exp, t_exp = experiment()
        gun.set_B_ep((v_exp,t_exp))

        t2v = gun.set_t2v()
        ts = t2v.get_t()
        v = t2v(ts)

        fig = plt.figure('v,t')
        ax = fig.add_subplot(1,1,1)
        ax.plot(ts*1e6, v/1e5, label='simulation')
        ax.plot(t_exp*1e6, v_exp/1e5, label='experiment')
        ax.plot(t_exp*1e6, gun.ep/1e5, label=r'error $\epsilon$')
        ax.set_xlabel(r'$t/(\mu \rm{sec})$')
        ax.set_ylabel(r'$v/(\rm{km/s})$')
        ax.legend(loc='upper left')

        if iPlot:
            plt.show()
        #end
        if iWrite:
            fig.savefig(out_dir + 'vt_test.pdf',format='pdf')
        #end

        if iTest:
            with open('gold_data{:}gun_epsilon_gold.csv'.format(os.sep),'r') as fid:               
                gun_epsilon_gold = np.loadtxt(fid, delimiter = ',',
                    comments = '#')
            #end

            with open('gold_data{:}gun_v_gold.csv'.format(os.sep),'r') as fid:               
                gun_v_gold = np.loadtxt(fid, delimiter = ',',
                    comments = '#')
            #end

            npt.assert_array_almost_equal(gun_epsilon_gold, gun.ep, 
                decimal = 6, 
                err_msg = "GUN error vector does not match gold data")
            npt.assert_array_almost_equal(gun_v_gold, v, 
                decimal = 6, 
                err_msg = "GUN velocity vector does not match gold data")
        else:
            with open('gold_data{:}gun_epsilon_gold.csv'.format(os.sep),
                    'w') as fid:               
                np.savetxt(fid, gun.ep, delimiter = ',',
                    header = "Gold data generated by {:2s} on {:}".format(
                        usr, time.strftime("%d-%m-%Y")))
            #end
            with open('gold_data{:}gun_v_gold.csv'.format(os.sep),
                    'w') as fid:               
                np.savetxt(fid, v, delimiter = ',',
                    header = "Gold data generated by {:2s} on {:}".format(
                        usr, time.strftime("%d-%m-%Y")))
            #end
        #end
        self.assertTrue(iTest, 
                msg = "velocity and error vectors not tested against gold data")

#end

        


# def plot_d_hat(d_hat):
#     fig = plt.figure('d_hat', figsize=(7,6))
#     ax = fig.add_subplot(1,1,1)
#     ax.plot(d_hat)
#     ax.set_xlabel(r'$i$')
#     ax.set_ylabel(r'$\hat d[i]$')
#     fig.savefig('d_hat_test.pdf',format='pdf')
#     return 0

# def test_func_etc(gun, eos_0, eos_1, d_hat, t_exp, v_exp, plot_files=None):

#     ep_0 = gun.ep
#     gun.set_eos(eos_0)
#     d = np.zeros(len(d_hat))
#     S_0 = gun.func(d)        # Original cost function
#     G_0 = gun.G_matrix()
#     ll_0 = gun.log_like((v_exp,t_exp))[0] # Original log likelihood
#     f_0 = gun.eos(gun.x)                  # Original eos values
    
#     S_1 = gun.func(d_hat)                # Updated cost function
#     gun.set_eos(eos_1)
#     G_1 = gun.G_matrix() # FixMe: Should devise test of G_0 and G_1
    
#     # Get epsilon for updated EOS
#     gun.set_B_ep((v_exp,t_exp))
#     ll_1 = gun.log_like((v_exp,t_exp))[0] # Updated log likelihood
#     f_1 = gun.eos(gun.x)                  # Updated eos values

#     if plot_files == None:
#         return 0
    
#     print('''lstsq reduced func from {0:.3e} to {1:.3e}
#  and the increase in log likelihood is {2:.3e} to {3:.3e}'''.format(
#         S_0, S_1, ll_0, ll_1))
#     if 'errors' in plot_files:
#         fig = plt.figure('errors')
#         ax = fig.add_subplot(1,1,1)
#         # Plot orignal errors
#         fig = plt.figure('errors')
#         ax = fig.add_subplot(1,1,1)
#         ax.plot(t_exp*1e6, ep_0, label='Original velocity error ep')
#         # Plot reduced errors
#         ax.plot(t_exp*1e6, gun.ep, label='New velocity error')
#         ax.set_xlabel(r'$t/(\mu\rm{sec})$')
#         ax.set_ylabel(r'$v/(\rm{x/s})$')
#         ax.legend(loc='lower right')
#         fig.savefig(plot_files['errors'],format='pdf')
#     return 0
# def test_opt():
#     ''' Exercise GUN.opt()
#     '''
#     # Set up gun with spline eos
#     gun = GUN()
#     gun.set_eos_spline(gun.x,gun.eos(gun.x))
#     old_eos = gun.eos

#     # Make experimental data
#     vt = experiment()
    
#     gun.set_B_ep(vt)
#     error_0 = gun.ep
#     d_hat = gun.opt(vt)
#     gun.set_B_ep(vt)
#     error_1 = gun.ep
#     #d_hat = gun.free_opt((v_exp,t_exp), rcond=1e-6)
#     fig = plt.figure('opt_result', figsize=(7,6))
#     ax = fig.add_subplot(1,1,1)
#     ax.plot(d_hat)
#     ax.set_xlabel(r'$i$')
#     ax.set_ylabel(r'$\hat d[i]$')
#     fig = plt.figure('errors', figsize=(7,6))
#     ax = fig.add_subplot(1,1,1)
#     ax.plot(error_0,label='error_0')
#     ax.plot(error_1,label='error_1')
#     ax.legend()
#     fig = plt.figure('eos', figsize=(7,6))
#     ax = fig.add_subplot(1,1,1)
#     x = gun.x
#     ax.plot(x,old_eos(x),label='f_0')
#     ax.plot(x,gun.eos(x),label='f_1')
#     ax.legend()
#     ax.set_xlabel(r'$x/{\rm cm}$')
#     ax.set_ylabel(r'$f/{\rm dyn}$')
#     #plt.show()
#     fig.savefig('opt_result.pdf',format='pdf')  
#     return 0 
   
# def test():

#     # test_spline()
#     # Test __init__ _set_N, set_eos and E methods of GUN
#     gun = GUN()
#     s = '{0:.1f}'.format(gun.E(4))
#     assert s  == '79200000000.0','gun.E(4)={0}'.format(s)

#     test_gun_splines(gun) # Makes gun.eos and gun.t2v splines
#     G = gun.G_matrix()
#     eos_0 = gun.eos
#     v_exp, t_exp = experiment()
#     test_set_D(gun, 'D_test.pdf')
#     test_set_B_ep(gun, v_exp, t_exp, 'vt_test.pdf')
#     test_set_BD(gun, 'BD_test.pdf', t_exp)
    
#     # Solve BD*d=epsilon for d without constraints
#     d_hat = gun.free_opt((v_exp, t_exp), rcond=1e-5)
#     plot_d_hat(d_hat)
#     eos_1 = gun.eos
    
#     # Exercise func, d_func and constraint and make plots for d_hat, etc
#     plot_files = {
#         'errors':'errors_test.pdf',
#         }
#     test_func_etc(gun, eos_0, eos_1, d_hat, t_exp, v_exp, plot_files)
    
#     #plt.show()
#     # FixMe: What about derivative of constraint?
#     return 0

if __name__ == '__main__':
    """

    **Flags **

    - iPlot -> True: Displays the figure(s) generated by each test immediately 
                     after each test's execution
            -> False: Figure is generated but not displayed 
    - iWrite -> True: Writes a pdf file of each test's figures to 'out_dir'
             -> False: No pdf output
    - iTest -> True: Asserts that the data generated by each test matches the 
                     'gold' data generated and archived by a previous test which
                     a previous user has determined to yield acceptable results
            -> False: Writes a test output to a 'gold' data file. The user must
                      inspect the data and corresponding plots to ensure the 
                      code is behaving properly

    **Variables**

    - out_dir -> STR: Location for figure output. *Default ./figures/*
    - usr -> STR: Two character user identifier, appended as a comment to all 
                  gold data files along with the date. *Default 'SA'*


    **Useage**

    1) With iPlot set to True and iTest false, run the testing suite 
    2) Use the figures and gold data output to ensure the code is behaving 
       properly
    3) Set iPlot to False and iTest to true 
    4) At any time the unit tests can be run using 
        
        $ python test_gun.py 
    
    5) This will ensure that the code returns the same results as it did when a 
       user last checked the quality of its output
    """

    iPlot = False 
    iWrite = False
    iTest = True
    out_dir = '.{:}..{:}figures{:}'.format(os.sep, os.sep, os.sep) 
    usr = 'SA'
    unittest.main(verbosity = 4)