import numpy as np
from collections.abc import Callable

from ..matchingResult import MatchingResult
from .entropyBase import EntropyBase
from .entropyBallistic import EntropyBallistic
from .entropyNLTE import EntropyNLTE

class EntropyInterpolated(EntropyBase):
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
        Initialize the EntropyInterpolated class.

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

        self.entropyNLTE = EntropyNLTE(massesSymmetricPhase, massesBrokenPhase, dofs, statistics, tau, L, Tn, gstar, cs2)
        self.entropyBallistic = EntropyBallistic(massesSymmetricPhase, massesBrokenPhase, dofs, statistics, Tn, gstar, cs2)

    def sigma(self, matching: MatchingResult) -> float:
        """
        Computes the interpolation of the entropy fraction sigma
        between the NLTE and ballistic limit.
        See Eqs. (???) and (???) of 26xx.xxxxx.

        Parameters
        ----------
        matching : MatchingResult
            MatchingResult object containing v_\pm and T_\pm.
        """
        sigmaNLTE = self.entropyNLTE(matching)
        sigmaBallistic = self.entropyBallistic(matching)
        return sigmaBallistic*sigmaNLTE/(sigmaBallistic+sigmaNLTE)