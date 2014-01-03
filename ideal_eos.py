"""
This file is used/imported by ideal_gas.py

molar volume of water is 18.016 mL or 18.016e-6 M^3
The CJ pressure for TNT is about 19GPa
The molar volume of TNT products is about 6.25e-6 M^3/mole
TNT CJ temp 3712 degrees K
>>> gas = EOS()
>>> CJP = 19e9
>>> CJT = 3712
>>> CJv = gas.PT2v(CJP,CJT)
>>> print('%8.2e M^3/mole'%CJv)
1.62e-06 M^3/mole
>>> print('%6.1f'%gas.Pv2T(CJP,CJv))
3712.0
>>> print('%8.2e'%gas.Tv2P(CJT,CJv))
1.90e+10

"""
import numpy as np, scipy.integrate
class EOS:
    """This EOS class is designed to be subclassed, but it is filled
out for theideal gas law (and perhaps hydrogen if molar mass
necessary).

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
v_min = 1e-6 Cubic meters/mole
v_max = 4e-6 Cubic meters/mole

E_min = 25e3 Joules/mole
E_max = 400e3 Joules/mole

E_0 = 2.5*R*T_0 = 25e3 Joules/mole
T_0 = 1.20272e3 Degrees K
v_0 = 1e-6 Cubic meters/mole

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
        (volume of 1 mole)"""
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
    def Ev2P(self, E, v):
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

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

#---------------
# Local Variables:
# eval: (python-mode)
# End:
