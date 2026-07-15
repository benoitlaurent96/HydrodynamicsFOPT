import numpy as np
from scipy import integrate

from ..matchingResult import MatchingResult
from ..hydrodynamics import Hydrodynamics
from .ballisticIntegrals import totalPressure

class EntropyBallistic:
    def __init__(self,
                 massesSymmetricPhase: list[float],
                 massesBrokenPhase: list[float],
                 dofs: list[int],
                 statistics: list[int],
                 Tn: float,
                 gstar: float = 106.75,
                 cs2: float = 1/3) -> None:
        """
        Initialize the EntropyProduction class

        Parameters
        ----------
        massesSymmetricPhase : list[float]
            List containing the masses of each particle in the symmetric phase.
        massesBrokenPhase : list[float]
            List containing the masses of each particle in the broken phase.
        dofs : list[int]
            List of number of degrees of freedom of each particle.
        statistics : list[int]
            List of the particles' statistic. Must be 1 for fermions and -1 for bosons.
        Tn : float
            Nucleation temperature.
        gstar : float
            Total number of effective degrees of freedom in the symmetric phase.
            Default is 106.75.
        cs2 : float, optional
            Speed of sound squared in front of the wall. Default is 1/3.

        """
        
        assert len(massesSymmetricPhase) == len(massesBrokenPhase) == len(dofs) == len(statistics), """Error: The lists 'massesSymmetricPhase' 'massesBrokenPhase','dofs' and 'statistics' must all have the same length."""
        
        self.mSym = np.array(massesSymmetricPhase)
        self.mBrok = np.array(massesBrokenPhase)
        self.dofs = np.array(dofs)
        self.statistics = np.array(statistics)
        self.Tn = Tn
        self.wn = gstar*(1+1/cs2)*np.pi**2*Tn**4/90
    
    def __call__(self, matching: MatchingResult):
        return self.sigma(matching)

    def sigma(self, matching: MatchingResult):
        Tp = matching.Tp
        Tm = matching.Tm
        vp = matching.vp
        vm = matching.vm
        sp = self.wn*matching.wp/Tp

        DS = sum([self.dofs[i]*self.tau[i]
                  *EntropyNLTE.nlteIntegral(self.mSym[i], self.mBrok[i], Tp, Tm,
                                            vp, vm, self.statistics[i], self.L) for i in range(len(self.tau))])
        return DS*np.sqrt(1-vp**2)/(sp*vp)
        
    def nlteIntegral(m1, m2, Tp, Tm, vp, vm, statistic, L):
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