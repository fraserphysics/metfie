"""
This file is used/imported by ideal_qt.py

molar volume of water is 18.016 mL or 18.016e-6 M^3
The CJ pressure for TNT is about 19GPa
The molar volume of TNT products is about 6.25e-6 M^3/mole
  # Fix me: moles or grams?
TNT CJ temp 3712 degrees K
>>> gas = ideal()
>>> CJP = 19e9
>>> CJT = 3712
>>> CJv = gas.PT2v(CJP,CJT)
>>> print('%8.2e M^3/mole'%CJv) # Fix me: moles or grams?
1.62e-06 M^3/mole # Fix me: moles or grams?
>>> print('%6.1f'%gas.Pv2T(CJP,CJv))
3712.0
>>> print('%8.2e'%gas.Tv2P(CJT,CJv))
1.90e+10

"""
import numpy as np, scipy.integrate
class ideal:
    """This class is designed to be subclassed, but it is filled out for
the ideal gas law (and perhaps hydrogen if molar mass necessary).

T Temperature in degrees K
v Specific volume, volume of one mol
P Pressure in Pascals
E in Joules
S in Joules/degree

Assuming cv=5/2 (5 degrees of freedom for a diatomic ideal gas). I got
the following equations from Wikipedia:

Pv = RT
E = cvRT = cvPv
S = R*( cv*log(T/T_0) + log(v/v_0) )

From the above, I find:

S = R*( cv*log(E/E_0) + log(v/v_0) )
e^{S/R} = (E/E_0)^cv * v/v_0
v(E,S) = v_0 * (E_0/E)^cv * (e^{S/R})
E(v,S) = E_0 * ( e^(S/R) * v_0/v )^(1/cv)
v(P,S) = ( e^(S/R) v_0 )^ (1/(1+cv)) * ( E_0/(cv*P) )^(cv/(1+cv))

Below, I list chosen parameters and some consequences

P_min = 1e10 Pascals
P_max = 4e10 Pascals
v_min = 1e-6 Cubic meters/mole # Fix me: moles or grams?
v_max = 4e-6 Cubic meters/mole # Fix me: moles or grams?

E_min = 25e3 Joules/mole # Fix me: moles or grams?
E_max = 400e3 Joules/mole # Fix me: moles or grams?

E_0 = 2.5*R*T_0 = 25e3 Joules/mole # Fix me: moles or grams?
T_0 = 1.20272e3 Degrees K
v_0 = 1e-6 Cubic meters/mole # Fix me: moles or grams?

S_min = 0
S_max = 69.16 Joules/degree

    """
    R = 8.3144621    #Joules/(degree mol) = (Meters^3 Pascal)/(degrees mol)
    cv = 2.5         # 5/2 for diatomic gas
    v_0 = 1.0e-6     # Molar volume in cubic meters that gives zero entropy
    E_0 = 25e3       # Absolute temperature that gives zero entropy
    T_0 = E_0*(cv*R) # Absolute temperature that gives zero entropy
    def __init__(self):
        pass
    def Pv2T(self, P, v):
        """Return Temprature given pressure and specific volume
        (volume of 1 gram)"""
        return P*v/self.R
    def PT2v(self, P,T):
        return self.R*T/P
    def Tv2P(self, T, v):
        return self.R*T/v
    # The next 4 methods work with energy E instead of T via E=c_v*RT.
    # c_v=5/2 for a diatomic gas model.
    def Pv2E(self, P, v):
        return self.cv*P*v
    def PE2v(self, P,E):
        return (E/P)/self.cv
    def Ev2P(self, # ideal instance
             E, v):
        return (E/v)/self.cv
    def Ev2S(self, E, v):
        """Return entropy given energy and specific volume"""
        return self.R*(self.cv*np.log(E/self.E_0) + np.log(v/self.v_0))
    # The next 3 methods calculate 2 state variables from entropy and
    # the third state variable.
    def Sv2PE(self, S, v):
        E = self.E_0 * ( np.exp(S/self.R) * self.v_0/v )**(1/self.cv)
        P = self.Ev2P(E,v)
        return P,E
    def SE2Pv(self, S, E):
        v = self.v_0 * (self.E_0/E)**self.cv * (np.exp(S/self.R))
        P = self.Ev2P(E,v)
        return P,v
    def SP2vE(self, S, P):
        t = np.exp(S/self.R) * self.v_0
        t *= ( self.E_0/(self.cv*P) )**self.cv
        v = t ** (1/(1+self.cv))
        E = self.Pv2E(P,v)
        return v,E
    def isentropic_expansion(self,
                             P_i,      #Initial pressure in Pa
                             v_i,      #Initial volume in M^3
                             v_f,
                             mass=1.e2 # Piston mass in grams
                             ):
        """Integrate work and time for a one dimensional isentropic
        expansion from (P_i,v_i) to v_f.  Assume the gas has no mass,
        but that it is in a cylinder with a cross section of one cm^2
        and it is pushing a piston of the given mass in grams.
        """
        cross_section = 1e-4 # one cm^2 in M^2
        Kg2g = 1.0e3
        N = 1000
        step = (v_f-v_i)/N
        _v = np.arange(v_i,v_f+step/2,step) # Array of N+1 volumes
        def func(P,v):
            """ Returns the derivitive of P wrt v on isentrope"""
            return -(7.0*P)/(5.0*v)
        _P = scipy.integrate.odeint(
            func,                         # Derivative function
            np.array([P_i]),              # Initial condition
            _v)                           # Returns array of pressures in Pa
        _P_bar = (_P[:-1]+_P[1:])/2
        _KE = _P_bar.cumsum()*step        # In Joules
        _vel = np.sqrt(2*_KE*Kg2g/mass)   # In M/s
        _dvdt = _vel * cross_section      # In M^3/s
        _dt = step/_dvdt
        Dt = _dt.sum()                    # In s
        # Now use analytic results
        E_i = self.Pv2E(P_i,v_i)
        E_f = E_i*(v_i/v_f)**.4
        return (_KE[-1],Dt,E_i-E_f)
class shaw(ideal):
    '''
    Equations 4.6 to 4.9 of Hixson et al JAP 2000:

    P(\rho,E) = \rho^3 * (\rho) + g(\rho) * (E - E_CJ(\rho))
    E_CJ(\rho) = E_CJ - \int_{V_CJ}^V P_CJ(\rho) dV

    g(x) = 1.1520 - 1.149*x + 4.2382*x^2 - 2.2234*x^3
    f(x) = 2.4287 + 0.0378*x + 0.1005*x^2 - 0.7166*x^3 + 0.4537*x^4 +
           0.8062*x^5 - 0.8473*x^6

    x(\rho) = \rho - 2.4403

    Valid region 1.9 < \rho 3.2 => 0.3125 < v <0.5263

    From gm.py  17 < P < 79 GPa

    Using E = (P_0+P_1)*(v_0-v_1)/2 = 10.2624 x 10 ^3 Joules
    '''
    v_min = 0.3 # (cm)^3/g
    v_max = 0.6 # (cm)^3/g
    E_min = 5
    E_max = 20
    a = np.array([2.4287, 0.0378, 0.1005, -0.7166, 0.4537, 0.8062, -0.8473, 0])
    b = np.array([1.1520, -1.149, 4.2382, -2.2234])

    E_0 = 2e5      # Fraser's hack to keep E positive
    cv = 2.5       # Joules/degree.  A hack
    S_0 = 1000     # FixMe: Get a plausible value
    ''' 
    dE = cv *dT
    At v_CJ, let T = E/cv
    dS = dE/T      For heating at constant volume
    S = S_0 + cv*ln(T/T_0) = S_0 + cv*ln(E/E_0) at v_CJ
    
    '''

    P_0 = 3.58e10  # CJ value from Hixson
    rhoCJ = 2.4403 # CJ value from Hixson

    n_a = len(a)
    I = np.arange(n_a, dtype=np.int32)

    def __init__(self):
        self.S = None
        def F(v,PE):
            P,E = PE
            #Derivatives at constant S
            dE_dv = P
            #dP/dv = P_v + P_E*P Where
            #P_v = \partial P / \partial v
            #P_E = \partial P / \partial E
            r = 1/v
            
            P_E = self.g(r)
            f = self.f(r)
            f_prime = self.f_prime(r)
            g = self.g(r)
            g_prime = self.g_prime(r)
            E_CJ = self.E_CJ(r)
            P_x = 3*r**2*f + r**3*f_prime + g_prime*(E - E_CJ) + r*f*g
            P_v = -r**2*P_x
            dP_dv = P_v + P_E*P
            return [dP_dv, dE_dv]
        from scipy.integrate import ode
        self.ode = ode(F).set_integrator('dopri5')
    def f(self, # A shaw instance
          r     # Density in g/cm^3
    ):
        ''' Calculate function f from Hixson for density r
        '''
        x = r-self.rhoCJ
        return np.dot(x**self.I, self.a)
    def f_prime(self, # A shaw instance
          r     # Density in g/cm^3
    ):
        ''' Calculate f'
        '''
        x = r-self.rhoCJ
        return np.dot(x**(self.I[1:]), self.a[1:]*self.I[1:])
    def g(self, # A shaw instance
          r     # Density in g/cm^3
    ):
        ''' Calculate function g from Hixson for density r
        '''
        x = r-self.rhoCJ
        return np.dot(x**(self.I[:4]), self.b)
    def g_prime(self, # A shaw instance
          r     # Density in g/cm^3
    ):
        ''' Calculate function g'
        '''
        x = r-self.rhoCJ
        return np.dot(x**(self.I[1:4]), self.b[1:]*self.I[1:4])
    def P_CJ(self, # A shaw instance
             r     # Density in g/cm^3
    ):
        ''' Calculate pressure on the CJ isentrope for density r
        '''
        return r^3 * self.f(r)
    def E_CJ(self, # A shaw instance
             r     # Density in g/cm^3
    ):
        ''' Calculate energy on the CJ isentrope for density r
        '''
        x = r - self.rhoCJ
        E = self.E_0 + self.a[0]*self.rhoCJ*x
        c = (self.a[:-1] - self.rhoCJ*self.a[1:])/self.I[1:]
        return E + np.dot(c, x**(self.I[:-1] + 2))
    # The next 4 methods work with energy E instead of T
    def Ev2P(self, # Shaw instance
             E, v):
        '''Implements Eqn. 4.6 of Hixson
        '''
        r = 1/v
        return r**3 * self.f(r) + self.g(r)*(E-self.E_CJ(r))
    def Ev2S(self, E, v):
        '''Return entropy given energy and specific volume.  A correct
        calculation is:

        1. Integrate isentrope ODE from (E,v) to (E_1, v_CJ)
        2. S = S_CJ + c_v log(E_1/E_CJ)

        This method uses the approximation D = E_CJ(v)-E_0

        '''
        r = 1/v
        D = self.E_CJ(r) - self.E_0
        return self.S_0 + self.cv*np.log((E - D)/self.E_0)
        d_E = E - E_CJ
        return self.R*(self.cv*np.log(E/self.E_0) + np.log(v/self.v_0))
    def Pv2E(self, # shaw instance
             P, v):
        from scipy.optimize import brentq
        def f(E,v):
            return P - self.Ev2P(E,v)
        rv = brentq(f, self.E_min, self.E_max, args=(v,))
        assert rv == float(rv)
        return rv
    def PE2v(self, P, v):
        from scipy.optimize import brentq
        def f(v,E):
            return P - self.Ev2P(E,v)
        rv = brentq(f, self.v_min, self.v_max, args=(E,))
        assert rv == float(rv)
        return rv

    '''The next 3 methods calculate 2 state variables from entropy and the
    third state variable.  Each could be done by using an ODE
    integrator to build an isentropic trajectory in (E,P) with the
    independent varible v running from v_min to v_max.

    1. At v = v_CJ move find E_1 by solving
               S - S_CJ = c_v log(E_1/E_CJ)
                    E_1 = E_CJ*np.exp((S-S_CJ)/cv)

    2. Calculate P_1 from E_1, v_CJ

    3. Integrate ODE from v_CJ to v_max and from v_CJ to v_min to
       obtain isentropic trajectory through (E_1,P_1).

    '''
    def isentrope(self, S):
        if self.S == S:
            return
        self.S = S

        # Make array of v values.  To be independent variable for ode
        # and splines.
        dv = (self.v_max - self.v_min)/1000
        v = np.arange(self.v_min-dv, self.v_max+1.5*dv, dv)
        # Allocate array for dependent variables
        PE = np.empty((2,len(v)))
        
        # Find initial state variables for ode
        v_1 = 1/self.rhoCJ
        E_1 = self.E_0*np.exp((S-self.S_0)/self.cv)
        P_1 = self.Ev2P(E_1, v_1)
        i_1 = np.searchsorted(v,[v_1])[0] # v[i_1-1] < v_1 < v[i_1]

        self.ode.set_initial_value([P_1,E_1],v_1)
        for i in range(i_1,len(v)):
            self.ode.integrate(v[i])
            PE[:,i] = self.ode.y
        self.ode.set_initial_value([P_1,E_1],v_1)
        for i in range(i_1-1,-1,-1):
            self.ode.integrate(v[i])
            PE[:,i] = self.ode.y
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        self.isentrope_v2E = spline(v,PE[1],k=3,s=0)
        self.isentrope_E2v = spline(PE[1,-1::-1],v)
        self.isentrope_P2v = spline(PE[0,-1::-1],v)
        return
    def Sv2PE(self, S, v):
        self.isentrope(S)
        E = self.isentrope_v2E(v) # A spline evaluation
        P = self.Ev2P(E,v)
        return P,E
    def SE2Pv(self, S, E):
        self.isentrope(S)
        v = self.isentrope_E2v(E) # A spline evaluation
        P = self.Ev2P(E,v)
        return P,v
    def SP2vE(self, S, P):
        self.isentrope(S)
        v = self.isentrope_P2v(P) # A spline evaluation
        E = self.Pv2E(P,v)
        return v,E

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

#---------------
# Local Variables:
# eval: (python-mode)
# End:
