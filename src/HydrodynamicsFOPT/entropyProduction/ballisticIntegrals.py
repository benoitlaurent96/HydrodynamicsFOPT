"""
Functions to compute the ballistic pressure.
See Eqs. (B81) and (C88) of 2411.13641.
"""

import numpy as np
from scipy import integrate, special
        
def reflection(m1, m2, T, v, a=1):
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

def transmission_p(m1, m2, T, v, a=1):
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

def transmission_m(m1, m2, T, v, a=1):
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

def totalPressure(m1, m2, a, vp, vm, Tp, Tm):
    """
    Computes the total pressure for a single degree of freedom.
    """
    if m1 == m2:
        return 0
    Pr = reflection(m1,m2,Tp,vp,a)
    Ptp = transmission_p(m1,m2,Tp,vp,a)
    Ptm = transmission_m(m1,m2,Tm,vm,a)
    return Pr+Ptp+Ptm