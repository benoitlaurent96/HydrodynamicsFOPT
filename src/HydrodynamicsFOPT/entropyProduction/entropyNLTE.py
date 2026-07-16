import numpy as np
from collections.abc import Callable

from ..matchingResult import MatchingResult
from .entropyBase import EntropyBase
from .integrals import nlteIntegral

class EntropyNLTE(EntropyBase):
    def __init__(self,
                 massesSymmetricPhase: list[float|Callable[[float],float]],
                 massesBrokenPhase: list[float|Callable[[float],float]],
                 dofs: list[int],
                 statistics: list[int],
                 tau: list[float],
                 L: float,
                 Tn: float,
                 gstar: float = 106.75,
                 cs2: float = 1/3) -> None:
        """
        Initialize the EntropyNLTE class.

        Parameters
        ----------
        massesSymmetricPhase : list[float|Callable[[float],float]]
            List containing the masses of each particle in the symmetric phase.
            The elements of the list can either be a float or a function of the temperature.
        massesBrokenPhase : list[float|Callable[[float],float]]
            List containing the masses of each particle in the broken phase.
            The elements of the list can either be a float or a function of the temperature.
        dofs : list[int]
            List of number of degrees of freedom of each particle.
        statistics : list[int]
            List of the particles' statistic. Must be 1 for fermions and -1 for bosons.
        tau : list[float]
            List of effective relaxation times (in units of 1/T) for each particle.
        L : float
            Wall thickness.
        Tn : float
            Nucleation temperature.
        gstar : float
            Total number of effective degrees of freedom in the symmetric phase.
            Default is 106.75.
        cs2 : float, optional
            Speed of sound squared in front of the wall. Default is 1/3.

        """
        
        super().__init__(massesSymmetricPhase, massesBrokenPhase, dofs, statistics, Tn, gstar, cs2)
        self.tau = np.array(tau)
        self.L = L

    def sigma(self, matching: MatchingResult) -> float:
        """
        Computes the entropy fraction sigma in the NLTE limit.
        See Eqs. (???) and (???) of 26xx.xxxxx.

        Parameters
        ----------
        matching : MatchingResult
            MatchingResult object containing v_\pm and T_\pm.
        """
        Tp = matching.Tp
        Tm = matching.Tm
        vp = matching.vp
        vm = matching.vm
        sp = self.wn*matching.wp/Tp

        DS = sum([self.dofs[i]*self.tau[i]
                  *nlteIntegral(self.mSym[i](Tp), self.mBrok[i](Tm), Tp, Tm,
                                            vp, vm, self.statistics[i], self.L) for i in range(len(self.tau))])
        return DS*np.sqrt(1-vp**2)/(sp*vp)