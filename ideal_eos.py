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
class EOS(object):
    """This EOS class is designed to be subclassed, but it is filled
out for theideal gas law (and perhaps hydrogen if molar mass
necessary).

T Temperature in degrees K
v Specific volume, volume of one mol
P Pressure in Pascals
    """
    R = 8.3144621 #Joules/(degree mol) = (Meters^3 Pascal)/(degrees mol)
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
    # The next 3 methods work with energy E instead of T via E=(5/2)RT
    # or T = (2/5)(E/R) or Pv = (2/5)E
    def Pv2E(self, P, v):
        return 2.5*P*v
    def PE2v(self, P,E):
        return 0.4*E/P
    def Ev2P(self, E, v):
        return 0.4*E/v
    def isentrope_v(self, new_v, old_P, old_E):
        old_v = self.PE2v(old_P,old_E)
        new_P = old_P*(old_v/new_v)**1.2
        new_E = self.Pv2E(new_P,new_v)
        return (new_P,new_E)
    def isentrope_P(self, new_P, old_v, old_E):
        old_P = self.Ev2P(old_E,old_v)
        new_v = old_v*(old_P/new_P)**(5.0/7.0)
        new_E = self.Pv2E(new_P,new_v)
        return (new_v,new_E)
    def isentrope_E(self, new_E, old_P, old_v):
        old_E = self.Pv2E(old_P,old_v)
        new_P = old_P*(new_E/old_E)**(7.0/2.0)
        new_v = self.PE2v(new_P,new_E)
        return (new_P,new_v)
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
