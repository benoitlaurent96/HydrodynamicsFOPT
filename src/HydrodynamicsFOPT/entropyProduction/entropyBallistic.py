import numpy as np
from scipy import integrate

from ..matchingResult import MatchingResult
from .ballisticIntegrals import totalPressure, pT

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
        self.dofMassless = gstar
        for i, dof in enumerate(dofs):
            if statistics[i] == 1:
                self.dofMassless -= 7*dof/8
            elif statistics[i] == -1:
                self.dofMassless -= dof
            else:
                raise ValueError('statistic must only contain 1 or -1.')
    
    def __call__(self, matching: MatchingResult):
        return self.sigma(matching)

    def sigma(self, matching: MatchingResult):
        Tp = matching.Tp
        Tm = matching.Tm
        vp = matching.vp
        vm = matching.vm
        gp = 1/np.sqrt(1-vp**2)
        gm = 1/np.sqrt(1-vm**2)
        wm = self.wn*matching.wm
        wp = self.wn*matching.wp

        DS = gp*vp*sum([self.dofs[i]*(totalPressure(self.mSym[i], self.mBrok[i], self.statistics[i],
                                                    vp, vm, Tp, Tm)
                                      -pT(self.mSym[i], Tp, self.statistics[i])
                                      +pT(self.mBrok[i], Tm, self.statistics[i])) for i in range(len(self.mSym))])/Tp
        
        DS -= gp*vp*self.dofMassless*np.pi**2*(Tp**4-Tm**4)/(90*Tp)
        DS -= (gp/Tp-gm/Tm)*wm*gm**2*vm
        DS += (gp*vp/Tp-gm*vm/Tm)*wm*gm**2*vm**2
        
        return Tp*DS/(wp*vp*gp)