import numpy as np
from collections.abc import Callable

from ..matchingResult import MatchingResult
from .integrals import totalPressure, pT, enthalpy
from .entropyBase import EntropyBase

class EntropyBallistic(EntropyBase):
    def __init__(self,
                 massesSymmetricPhase: list[float|Callable[[float],float]],
                 massesBrokenPhase: list[float|Callable[[float],float]],
                 dofs: list[int],
                 statistics: list[int],
                 Tn: float,
                 gstar: float = 106.75,
                 cs2: float = 1/3) -> None:
        """
        Initialize the EntropyBallistic class.

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
        Tn : float
            Nucleation temperature.
        gstar : float
            Total number of effective degrees of freedom in the symmetric phase.
            Default is 106.75.
        cs2 : float, optional
            Speed of sound squared in front of the wall. Default is 1/3.

        """
        
        super().__init__(massesSymmetricPhase, massesBrokenPhase, dofs, statistics, Tn, gstar, cs2)

    def sigmaSingleDOF(self, i: int, matching: MatchingResult) -> float:
        """
        Computes the entropy fraction sigma in the ballistic limit.
        See Eqs. (???) and (???) of 26xx.xxxxx.

        Parameters
        ----------
        i : int
            Index of the DOF.
        matching : MatchingResult
            MatchingResult object containing v_\pm and T_\pm.
        """
        Tp = matching.Tp
        Tm = matching.Tm
        vp = matching.vp
        vm = matching.vm
        gp = 1/np.sqrt(1-vp**2)
        gm = 1/np.sqrt(1-vm**2)
        wp = self.wn*matching.wp

        DS = 0.5*(gp*vp/Tp+gm*vm/Tm)*(totalPressure(self.mSym[i](Tp), self.mBrok[i](Tm), self.statistics[i], vp, vm, Tp, Tm)
                                      - pT(self.mSym[i](Tp), Tp, self.statistics[i])
                                      + pT(self.mBrok[i](Tm), Tm, self.statistics[i]))
        DS += 0.5*((gp*vp/Tp-gm*vm/Tm)*gm**2*vm**2-(gp/Tp-gm/Tm)*gm**2*vm)*enthalpy(self.mBrok[i](Tm), Tm, self.statistics[i])
        DS += 0.5*((gp*vp/Tp-gm*vm/Tm)*gp**2*vp**2-(gp/Tp-gm/Tm)*gp**2*vp)*enthalpy(self.mSym[i](Tp), Tp, self.statistics[i])
        
        return Tp*DS/(wp*vp*gp)