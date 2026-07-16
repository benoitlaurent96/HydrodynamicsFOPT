import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
from dataclasses import dataclass

@dataclass
class MatchingResult():
    """
    Data class that contains the solution to the hydrodynamics equations.
    """

    vw: float | None
    """Wall velocity"""

    vp: float | None
    """Plasma velocity in front of the wall (wall frame)"""

    vm: float | None
    """Plasma velocity behind the wall (wall frame)"""

    wp: float | None
    """Enthalpy density in front of the wall"""

    wm: float | None
    """Enthalpy density behind the wall"""
    
    foundSolution: bool
    """Solution found successfully"""
    
    frontWaveProfile: integrate._ivp.ivp.OdeResult | None = None
    """Fluid profile in the front wave returned by integrate.solve_ivp"""

    backWaveProfile: integrate._ivp.ivp.OdeResult | None = None
    """Fluid profile in the back wave returned by integrate.solve_ivp"""
    
    Tp: float | None = None
    """Temperature in front of the wall"""

    Tm: float | None = None
    """Temperature behind the wall"""

    frontWaveRange: list | None = None
    backWaveRange: list | None = None
    """xi coordinates of the waves' boundaries"""

    _vFrontSpl: list[interpolate.BSpline] | None = None
    _vBackSpl: list[interpolate.BSpline] | None = None
    _wFrontSpl: list[interpolate.BSpline] | None = None
    _wBackSpl: list[interpolate.BSpline] | None = None
    _nFrontSpl: list[interpolate.BSpline] | None = None
    _nBackSpl: list[interpolate.BSpline] | None = None
    _isSpl: bool = False
    
    def normalizeDensity(self, densityNucl=1):
        self.Np = densityNucl
        if self.frontWaveProfile is not None:
            vpShock = self.frontWaveProfile.y[0,-1]
            vmShock = (vpShock-self.frontWaveProfile.t[-1])/(1-vpShock*self.frontWaveProfile.t[-1])
            nmShock = densityNucl*(vpShock/vmShock)*np.sqrt((1-vmShock**2)/(1-vpShock**2))
            self.frontWaveProfile.y[2] *= nmShock/self.frontWaveProfile.y[2,-1]
            self.Np = self.frontWaveProfile.y[2,0]
        
        self.Nm = self.Np*(self.vp/self.vm)*np.sqrt((1-self.vm**2)/(1-self.vp**2))
        
        if self.backWaveProfile is not None:
            self.backWaveProfile.y[2] *= self.Nm/self.backWaveProfile.y[2,0]
    
    def plotVelocity(self, derivative=0):
        xi = np.linspace(0, 1, 1000)
        plt.plot(xi, self.velocityProfile(xi, derivative))
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$v$')
        plt.xlim((0,1))
        plt.grid(True)
        plt.show()
            
    def plotEnthalpy(self, derivative=0):
        xi = np.linspace(0, 1, 1000)
        plt.plot(xi, self.enthalpyProfile(xi, derivative))
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$w/w_n$')
        plt.xlim((0,1))
        plt.grid(True)
        plt.show()
    
    def plotDensity(self, derivative=0):
        xi = np.linspace(0, 1, 1000)
        plt.plot(xi, self.densityProfile(xi, derivative))
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$N/N_n$')
        plt.xlim((0,1))
        plt.grid(True)
        plt.show()
    
    def setSplines(self, densityNucl=1):
        self.normalizeDensity(densityNucl)
        if self.frontWaveProfile is None and self.backWaveProfile is None:
            print('Error: The plasma profile has not been initialized.')
            raise 
        if self.frontWaveProfile is not None:
            mask = np.append([True], np.abs(self.frontWaveProfile.y[0,1:]-self.frontWaveProfile.y[0,:-1]) > 0)
            k = 1
            self.frontWaveRange = [self.frontWaveProfile.y[0,0], self.frontWaveProfile.y[0,-1]]
            self._vFrontSpl = [interpolate.make_interp_spline(self.frontWaveProfile.y[0,mask], self.frontWaveProfile.t[mask], k)]
            self._wFrontSpl = [interpolate.make_interp_spline(self.frontWaveProfile.y[0,mask], self.frontWaveProfile.y[1,mask], k)]
            self._nFrontSpl = [interpolate.make_interp_spline(self.frontWaveProfile.y[0,mask], self.frontWaveProfile.y[2,mask], k)]
            self._vFrontSpl.append(self._vFrontSpl[0].derivative(1))
            self._wFrontSpl.append(self._wFrontSpl[0].derivative(1))
            self._nFrontSpl.append(self._nFrontSpl[0].derivative(1))
        else:
            self.frontWaveRange = [self.backWaveProfile.y[0,0], self.backWaveProfile.y[0,0]]
            self._vFrontSpl = [lambda x: np.zeros_like(x), lambda x: np.zeros_like(x)]
            self._wFrontSpl = [lambda x: np.zeros_like(x), lambda x: np.zeros_like(x)]
            self._nFrontSpl = [lambda x: np.zeros_like(x), lambda x: np.zeros_like(x)]
        if self.backWaveProfile is not None:
            mask = np.append([True], np.abs(self.backWaveProfile.y[0,1:]-self.backWaveProfile.y[0,:-1]) > 0)
            k = 1
            self.backWaveRange = [self.backWaveProfile.y[0,-1], self.backWaveProfile.y[0,0]]
            self._vBackSpl = [interpolate.make_interp_spline(np.flip(self.backWaveProfile.y[0,mask]), np.flip(self.backWaveProfile.t[mask]), k)]
            self._wBackSpl = [interpolate.make_interp_spline(np.flip(self.backWaveProfile.y[0,mask]), np.flip(self.backWaveProfile.y[1,mask]), k)]
            self._nBackSpl = [interpolate.make_interp_spline(np.flip(self.backWaveProfile.y[0,mask]), np.flip(self.backWaveProfile.y[2,mask]), k)]
            self._vBackSpl.append(self._vBackSpl[0].derivative(1))
            self._wBackSpl.append(self._wBackSpl[0].derivative(1))
            self._nBackSpl.append(self._nBackSpl[0].derivative(1))
        else:
            self.backWaveRange = [self.vw-1e-10, self.vw-1e-10]
            self._vBackSpl = [lambda x: np.zeros_like(x), lambda x: np.zeros_like(x)]
            self._wBackSpl = [lambda x: np.zeros_like(x), lambda x: np.zeros_like(x)]
            self._nBackSpl = [lambda x: np.zeros_like(x), lambda x: np.zeros_like(x)]
        self._isSpl = True

    
    def velocityProfile(self, xi, derivative=0):
        if not self._isSpl:
            self.setSplines()
        return np.where(xi < self.backWaveRange[0], 0,
                        np.where(xi <= self.backWaveRange[1], self._vBackSpl[derivative](xi),
                                 np.where(np.all([self.frontWaveRange[0] <= xi, xi <= self.frontWaveRange[-1]], axis=0),
                                          self._vFrontSpl[derivative](xi), 0)))
    
    def enthalpyProfile(self, xi, derivative=0):
        if not self._isSpl:
            self.setSplines()
        if derivative == 0:
            w1 = 1
            if self.backWaveProfile is None:
                w0 = self.wm
            else:
                w0 = self.backWaveProfile.y[1,-1]
        else:
            w0 = 0
            w1 = 0
        return np.where(xi < self.backWaveRange[0], w0,
                        np.where(xi <= self.backWaveRange[1], self._wBackSpl[derivative](xi),
                                 np.where(np.all([self.frontWaveRange[0] <= xi, xi < self.frontWaveRange[-1]], axis=0),
                                          self._wFrontSpl[derivative](xi), w1)))
    
    def densityProfile(self, xi, derivative=0):
        if not self._isSpl:
            self.setSplines()
        if derivative == 0:
            n1 = 1
            if self.backWaveProfile is None:
                n0 = self.Nm
            else:
                n0 = self.backWaveProfile.y[2,-1]
        else:
            n0 = 0
            n1 = 0
        return np.where(xi < self.backWaveRange[0], n0,
                        np.where(xi <= self.backWaveRange[1], self._nBackSpl[derivative](xi),
                                 np.where(np.all([self.frontWaveRange[0] <= xi, xi < self.frontWaveRange[-1]], axis=0),
                                          self._nFrontSpl[derivative](xi), n1)))