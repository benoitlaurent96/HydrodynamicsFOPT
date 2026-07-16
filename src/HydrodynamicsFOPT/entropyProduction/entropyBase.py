import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..matchingResult import MatchingResult
from .integrals import enthalpy

class EntropyBase(ABC):
    def __init__(self,
                 massesSymmetricPhase: list[float|Callable[[float],float]],
                 massesBrokenPhase: list[float|Callable[[float],float]],
                 dofs: list[int],
                 statistics: list[int],
                 Tn: float,
                 gstar: float = 106.75,
                 cs2: float = 1/3) -> None:
        """
        Abstract class to compute entropy production.

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
        
        assert len(massesSymmetricPhase) == len(massesBrokenPhase) == len(dofs) == len(statistics), """Error: The lists 'massesSymmetricPhase' 'massesBrokenPhase','dofs' and 'statistics' must all have the same length."""
        
        self.mSym = []
        self.mBrok = []
        for mSym, mBrok in zip(massesSymmetricPhase, massesBrokenPhase):
            if callable(mSym):
                self.mSym.append(mSym)
            elif isinstance(mSym, float) or isinstance(mSym, int):
                self.mSym.append(lambda T: mSym)
            else:
                raise ValueError('massesSymmetricPhase must only contain floats or callables.')
            if callable(mBrok):
                self.mBrok.append(mBrok)
            elif isinstance(mBrok, float) or isinstance(mBrok, int):
                self.mBrok.append(lambda T: mBrok)
            else:
                raise ValueError('massesBrokenPhase must only contain floats or callables.')

        self.dofs = np.array(dofs)
        self.statistics = np.array(statistics)
        self.Tn = Tn
        self.wn = gstar*(1+1/cs2)*np.pi**2*Tn**4/90
        self.dofMassless = gstar
        for i, dof in enumerate(dofs):
            if statistics[i] == 1:
                self.dofMassless -= 7*dof/8
            elif statistics[i] == -1:
                self.dofMassless -= dof
            else:
                raise ValueError('statistic must only contain 1 or -1.')
    
    def __call__(self, matching: MatchingResult) -> float:
        """
        Computes the entropy fraction sigma by calling the function sigma().

        Parameters
        ----------
        matching : MatchingResult
            MatchingResult object containing v_\pm and T_\pm.
        """
        return self.sigma(matching)

    @abstractmethod
    def sigma(self, matching: MatchingResult) -> float:
        """
        Computes the entropy fraction sigma. See Eqs. (???) of 26xx.xxxxx.
        Must be redefined by user.

        Parameters
        ----------
        matching : MatchingResult
            MatchingResult object containing v_\pm and T_\pm.
        """
        pass

    def computePsi(self) -> float:
        """
        Computes the enthalpy fraction psi corresponding to the given masses.
        """
        wMasslessDOFs = 2*np.pi**2*self.dofMassless*self.Tn**4/45
        wBrok = sum([self.dofs[i]*enthalpy(self.mBrok[i](self.Tn), self.Tn, self.statistics[i]) for i in range(len(self.dofs))])
        wSym = sum([self.dofs[i]*enthalpy(self.mSym[i](self.Tn), self.Tn, self.statistics[i]) for i in range(len(self.dofs))])
        return (wBrok+wMasslessDOFs)/(wSym+wMasslessDOFs)