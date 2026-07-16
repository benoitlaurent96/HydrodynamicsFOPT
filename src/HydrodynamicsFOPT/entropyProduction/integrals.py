"""
Functions to compute the ballistic pressure.
See Eqs. (B81) and (C88) of 2411.13641.
"""

import numpy as np
from scipy import integrate, special
        
def reflection(m1: float, m2: float, T: float, v: float, a: int=1) -> float:
    """ 
    Computes the pressure from reflected particles.
    """
    gamma = 1/np.sqrt(1-v**2)
    if v == 1:
        return 0
    if m1 == 0 and a == 0:
        return T**2*(1+v)**3*gamma**2*(2*T**2-np.exp(-gamma*m2*(1-v)/T)*(2*T**2+2*m2*T*gamma*(1-v)+m2**2*gamma**2*(1-v)**2))/(2*np.pi**2)
    if m1 > m2:
        return -reflection(m2, m1, T, -v, a)
    
    def func(k):
        c = gamma*(np.sqrt(k**2+m1**2/T**2)-v*k)
        if a != 0:
            return a*T**4*k**2*np.log(1+a*np.exp(-c))/(2*np.pi**2*gamma)
        return T**4*k**2*np.exp(-c)/(2*np.pi**2*gamma)
    
    P = integrate.quad(func,0,np.sqrt(m2**2-m1**2)/T)
    return P[0]

def transmission_p(m1: float, m2: float, T: float, v: float, a: int=1) -> float:
    """ 
    Computes the pressure from the particles transmitted from the symmetric
    to the broken phase.
    """
    gamma = 1/np.sqrt(1-v**2)
    if m1 == 0 and a == 0:
        if v == 1:
            return m2**2*T**2/(4*np.pi**2)
        return T**2*(np.exp(-gamma*m2*(1-v)/T)*(2*T**2+2*m2*T*gamma*(1-v)+m2**2*gamma**2*(1-v)**2) - m2**2*gamma**2*(1-v)**2*special.kn(2, m2*gamma*(1-v)/T))/(4*np.pi**2*(1-v)**3*gamma**4)
    if m1 > m2:
        return -transmission_m(m2, m1, T, -v, a)
    
    def func(k):
        c = gamma*(np.sqrt(k**2+m1**2/T**2)-v*k)
        if a != 0:
            return a*T**4*k*(k-np.sqrt(k**2-(m2**2-m1**2)/T**2))*np.log(1+a*np.exp(-c))/(4*np.pi**2*gamma)
        return T**4*k*(k-np.sqrt(k**2-(m2**2-m1**2)/T**2))*np.exp(-c)/(4*np.pi**2*gamma)
    
    P = integrate.quad(func,np.sqrt(m2**2-m1**2)/T,np.inf)
    return P[0]

def transmission_m(m1: float, m2: float, T: float, v: float, a: int=1) -> float:
    """ 
    Computes the pressure from the particles transmitted from the broken
    to the symmetric phase.
    """
    if m1 > m2:
        return -transmission_p(m2, m1, T, -v, a)
    gamma = 1/np.sqrt(1-v**2)
    if v == 1:
        return 0
    def func(k):
        c = gamma*(np.sqrt(k**2+m2**2/T**2)+v*k)
        if a != 0:
            return a*T**4*k*(np.sqrt(k**2+(m2**2-m1**2)/T**2)-k)*np.log(1+a*np.exp(-c))/(4*np.pi**2*gamma)
        return T**4*k*(np.sqrt(k**2+(m2**2-m1**2)/T**2)-k)*np.exp(-c)/(4*np.pi**2*gamma)
    
    P = integrate.quad(func,0,np.inf)
    return P[0]

def totalPressure(m1: float, m2: float, a: int, vp: float, vm: float, Tp: float, Tm: float) -> float:
    """
    Computes the total pressure for a single degree of freedom.
    """
    if m1 == m2:
        return 0
    Pr = reflection(m1,m2,Tp,vp,a)
    Ptp = transmission_p(m1,m2,Tp,vp,a)
    Ptm = transmission_m(m1,m2,Tm,vm,a)
    return Pr+Ptp+Ptm

def pT(m: float, T: float, a: int=1) -> float:
    """
    Computes the thermal pressure of a single dof.
    """
    func = lambda p: a*p**2*np.log(1+a*np.exp(-np.sqrt(p**2+(m/T)**2)))
    return T**4*integrate.quad(func, 0, np.inf)[0]/(2*np.pi**2)

def nlteIntegral(m1: float, m2: float, Tp: float, Tm: float, vp: float, vm: float, statistic: int, L: float) -> float:
    """
    Computes the NLTE integral (Eq. ?? in 26xx.xxxxx). 
    Assumes that the mass, temperature and plasma velocity follow
    the same tanh profile.

    Parameters
    ----------
    m1 : float
        Mass in the symmetric phase
    m2 : float
        Mass in the broken phase
    Tp : float
        Temperature in front of the wall
    Tm : float
        Temperature behind the wall
    vp : float
        Plasma velocity in front of the wall (in the wall frame)
    vm : float
        Plasma velocity in behind the wall (in the wall frame)
    statistic : float
        Particle statistic (1 for fermions, -1 for bosons)
    L : float
        Wall width
    N : int, optional
        Number of grid points in the z and p directions. Default is 100.

    """
    dmsq = lambda m: -4*m*(m-m1)*(m-m2)/(L*(m1-m2))
    T = lambda m: Tp + (m-m1)*(Tm-Tp)/(m2-m1)
    v = lambda m: vp + (m-m1)*(vm-vp)/(m2-m1)
    gamma = lambda m: 1/np.sqrt(1-v(m)**2)
    Tavg = (Tp+Tm)/2
    p = lambda rho: -Tavg*np.log(rho)
    drhodpXdfeq = lambda m, rho: -Tavg*np.exp(-np.sqrt(p(rho)**2+m**2)/T(m)+p(rho)/Tavg)/(1+statistic*np.exp(-np.sqrt(p(rho)**2+m**2)/T(m)))**2

    integrand = lambda m, rho: 2*m*(v(m)*gamma(m)/T(m))**2*dmsq(m)*p(rho)**2*drhodpXdfeq(m, rho)/(8*np.pi**2*T(m)*(p(rho)**2+m**2))

    integral = integrate.dblquad(integrand, 1e-10, 1, m1, m2)[0]

    return integral