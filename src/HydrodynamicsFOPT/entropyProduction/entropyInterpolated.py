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
                 cs2: float = 1/3,
                 interpolateTotal: bool = False,
                 cancelQuadratic: bool = False,
                 vev: float|None = None) -> None:
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

        self.interpolateTotal = interpolateTotal
        self.cancelQuadratic = cancelQuadratic
        self.vev = vev

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

        if not self.interpolateTotal:
            return super().sigma(matching)

        sigmaNLTE = self.entropyNLTE(matching)
        sigmaBallistic = self.entropyBallistic(matching)
        sigma = sigmaBallistic*sigmaNLTE/(sigmaBallistic+sigmaNLTE)

        if self.vev is not None:
            b = 3*self.L**2*matching.wp*self.wn/self.vev**2
            roots = np.roots([-b*sigmaNLTE**2, sigmaNLTE**2-sigmaBallistic**2+2*b*sigmaBallistic*sigmaNLTE**2,
                              -2*sigmaBallistic*sigmaNLTE**2-b*sigmaNLTE**2*sigmaBallistic**2, sigmaNLTE**2*sigmaBallistic**2])
            return roots[np.argmin(np.abs(roots.imag))].real
            # print(roots)
            # print(np.sqrt(1-b*roots)*sigmaNLTE*sigmaBallistic/(sigmaBallistic+np.sqrt(1-b*roots)*sigmaNLTE))
            # print(sigmaNLTE, sigmaBallistic, self.L*self.Tn/np.sqrt(1-b*roots))
        if self.cancelQuadratic:
            sigma -= (sigmaBallistic*sigmaNLTE)**2/(sigmaBallistic+sigmaNLTE)**3
        return sigma

    def sigmaSingleDOF(self, i: int, matching: MatchingResult) -> float:
        """
        Computes the entropy fraction sigma for a single DOF corresponding to 
        the index i. See Eqs. (???) of 26xx.xxxxx.
        Must be redefined by user.

        Parameters
        ----------
        i : int
            Index of the DOF.
        matching : MatchingResult
            MatchingResult object containing v_\pm and T_\pm.
        """
        sigmaNLTE = self.entropyNLTE.sigmaSingleDOF(i, matching)
        sigmaBallistic = self.entropyBallistic.sigmaSingleDOF(i, matching)
        return sigmaNLTE*sigmaBallistic/(sigmaNLTE+sigmaBallistic)