import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
from time import time
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
    
    def plotVelocity(self, onlyFront=False):
        if self.frontWaveProfile is not None:
            plt.plot(self.frontWaveProfile.y[0], self.frontWaveProfile.t)
        
        if self.backWaveProfile is not None and not onlyFront:
            plt.plot(self.backWaveProfile.y[0], self.backWaveProfile.t)
        
        if self.frontWaveProfile is not None or self.backWaveProfile is not None:
            plt.xlabel(r'$\xi$')
            plt.ylabel(r'$v$')
            plt.grid(True)
            plt.show()
            
    def plotEnthalpy(self, onlyFront=False):
        if self.frontWaveProfile is not None:
            plt.plot(self.frontWaveProfile.y[0], self.frontWaveProfile.y[1])
        
        if self.backWaveProfile is not None and not onlyFront:
            plt.plot(self.backWaveProfile.y[0], self.backWaveProfile.y[1])
        
        if self.frontWaveProfile is not None or self.backWaveProfile is not None:
            plt.xlabel(r'$\xi$')
            plt.ylabel(r'$w/w_n$')
            plt.grid(True)
            plt.show()
    
    def plotDensity(self):
        if self.frontWaveProfile is not None:
            plt.plot(self.frontWaveProfile.y[0], self.frontWaveProfile.y[2])
        
        if self.backWaveProfile is not None:
            plt.plot(self.backWaveProfile.y[0], self.backWaveProfile.y[2])
        
        if self.frontWaveProfile is not None or self.backWaveProfile is not None:
            plt.xlabel(r'$\xi$')
            plt.ylabel(r'$n$')
            plt.grid(True)
            plt.show()


class Hydrodynamics:
    def __init__(self,
                 alN: float,
                 cb2: float=1/3,
                 cs2: float=1/3,
                 psiN: float=1,
                 Tn: float=1,
                 rtol: float=1e-6,
                 atol: float=1e-10):
        """
        Initialize the Hydrodynamics class which is used to solve hydrodynamics equations during a FOPT.
        Uses the template model to approximate the EOS. Can be use for direct or inverse PTs. For direct PTs
        with equal sound speeds on both sides of the wall, will use a much faster algorithm.

        Parameters
        ----------
        alN : float
            Strength of PT alpha_theta. It is defined in Eq. (19) of 2303.10171.
        cb2 : float, optional
            Squared speed of sound behind the wall. Default is 1/3.
        cs2 : float, optional
            Squared speed of sound in front of the wall. Default is 1/3.
        psiN : float, optional
            Enthalpy ratio. It is defined in Eq. (19) of 2303.10171. Default is 1.
        Tn : float, optional
            Nucleation temperature. Default is 1.
        rtol : float, optional
            Relative tolerance. Default is 1e-6.
        atol : float, optional
            Absolute tolerance. Default is 1e-10
        """
        
        assert 0 < cb2 < 1, "Error: cb2 must be between 0 and 1."
        assert 0 < cs2 < 1, "Error: cs2 must be between 0 and 1."
        assert psiN > 0, "Error: psiN must be positive."
        assert Tn > 0, "Error: Tn must be positive."
        assert rtol > 0, "Error: rtol must be positive."
        assert atol > 0, "Error: atol must be positive."
        
        self.alN = alN
        self.cb2 = cb2
        self.cs2 = cs2
        self.psiN = psiN
        self.Tn = Tn
        
        self.rtol = rtol
        self.atol = atol
        
        self.nu = 1+1/cb2
        self.mu = 1+1/cs2
        self.cb = np.sqrt(cb2)
        self.cs = np.sqrt(cs2)
        self.epsilon = (1/self.mu-(1-3*self.alN)/self.nu)
        self.epsilonSign = np.sign(self.epsilon)

        # Use a faster algorithm for direct PTs with the same sound speeds.
        self.fastCompute = False
        if cb2 == cs2 and alN > 0:
            self.fastCompute = True
        
        # Compute the all the limiting velocities
        self.findLimitingVelocities()
        
    def findLimitingVelocities(self):
        """
        Function that computes all the limiting velocities described below.
        If self.fastCompute=True, only vJDirect is computed. 
        """
        # Direct Jouguet (solution with vp=vw and vm=cb)
        if self.alN >= 0:
            self.vJDirect = self.cb*(1+np.sqrt(3*self.alN*(1-self.cb2+3*self.cb2*self.alN)))/(1+3*self.cb2*self.alN)
        else:
            self.vJDirect = None
        
        # Lowest direct deflagration (solution with vp=0 and vm=min(cb,vw))
        self.vLowFrontWave = 0
        if self.alN > 1/3:
            func = lambda vw: self.eqFrontWave(min(vw, self.cb), 0, vw)
            if func(1e-10)*func(self.vJDirect) <= 0:
                self.vLowFrontWave = optimize.root_scalar(func, bracket=(1e-10, self.vJDirect)).root
        
        if not self.fastCompute:
            # Lowest no front wave solution
            # (slowest solution between cs, direct Jouguet or one with vp=vw and vm=1)
            self.vLowNoFrontWave = self.cs
            if self.vJDirect is not None and self.vJDirect > self.cs:
                self.vLowNoFrontWave = self.vJDirect
            self.vLowNoFrontWave = min(1, max(self.vLowNoFrontWave, (1+self.cb2)/(1+3*self.cb2*self.alN)-1))
            if self.alN < (self.cb2-1)/(6*self.cb2):
                self.vLowNoFrontWave = 1
            
            # Inverse Jouguet (solution with vp=cs and vm=vw)
            sign = np.sign(self.eqFrontWave(1e-10, self.cs, 1e-10))
            func = lambda vw: sign*self.eqFrontWave(vw, self.cs, vw)
            
            minimum = optimize.minimize_scalar(func, bounds=(1e-10, self.cs), method='Bounded')
            if minimum.fun > 0:
                self.vJInverse = None
            else:
                self.vJInverse = optimize.root_scalar(func, bracket=(1e-10, minimum.x)).root
                
            # Lowest Inverse Hybrid (solution with vp=cs and vm=1)
            func = lambda vw: self.eqFrontWave(1, self.cs, vw)
            lowerBound = 1e-10 if self.vJInverse is None else self.vJInverse
            minimum = optimize.minimize_scalar(func, bounds=(lowerBound, self.cs), method='Bounded')
            maximum = optimize.minimize_scalar(lambda vw: -func(vw), bounds=(lowerBound, self.cs), method='Bounded')
            maximum.fun *= -1
            
            if minimum.fun * maximum.fun > 0:
                self.vLowInvHyb = None
            else:
                self.vLowInvHyb = optimize.root_scalar(func, bracket=(minimum.x, maximum.x)).root
                _,_,bounds1, bounds2 = self.velocityBounds(self.vLowInvHyb)
                if not (bounds1[0] <= self.vLowInvHyb <= bounds1[1] or bounds2[0] <= self.vLowInvHyb <= bounds2[1]):
                    self.vLowInvHyb = None
            if self.vLowInvHyb is None and self.vLowNoFrontWave == self.cs:
                self.vLowInvHyb = self.vJInverse
            
            # Solution with vp=cs and vm=cb
            self.vSoundSpeed = None
            if self.cb < self.cs:
                func = lambda vw: self.eqFrontWave(self.cb, self.cs, vw)
                if func(self.cb)*func(self.cs) <= 0:
                    self.vSoundSpeed = optimize.root_scalar(func, bracket=(self.cb, self.cs)).root
            
            # Fastest wall which has the BC vm=min(vw, cb)
            if self.vJDirect is not None:
                self.vHighVmBC = self.vJDirect
            elif self.vJInverse is not None:
                self.vHighVmBC = self.vJInverse
            if self.vSoundSpeed is not None:
                self.vHighVmBC = self.vSoundSpeed
            
            # Slowest wall which has the BC vp=max(cs, vw)
            self.vLowVpBC = self.vLowNoFrontWave
            if self.vLowInvHyb is not None:
                self.vLowVpBC = min(self.vLowVpBC, self.vLowInvHyb)
            if self.vSoundSpeed is not None:
                self.vLowVpBC = min(self.vLowVpBC, self.vSoundSpeed)
        
        else:
            # Only for fast calculations
            self.vLowNoFrontWave = self.vJDirect
            self.vHighVmBC = self.vJDirect
            self.vLowVpBC = self.vJDirect
            self.vJInverse = None
            self.vLowInvHyb = None
            self.vSoundSpeed = None
    
    def gammaSq(self, v: float) -> float:
        return 1/(1-v**2)
    def boostVelocity(self, v1: float, v2: float) -> float:
        return (v1-v2)/(1-v1*v2)
    
    def alpha(self, wp: float) -> float:
        """
        Computes alpha(Tp) from the enthalpy.

        Parameters
        ----------
        wp : float
            Enthalpy in front of the wall.

        Returns
        -------
        float

        """
        return (self.mu-self.nu)/(3*self.mu)+(self.alN-(self.mu-self.nu)/(3*self.mu))/wp
    
    def alphaFromV(self, vm: float, vp: float) -> float:
        """
        Computes alpha(Tp) from the plasma velocities behind and in front of the wall.

        Parameters
        ----------
        vm : float
            Plasma velocity behind the wall.
        vp : float
            Plasma velocity in front of the wall.

        Returns
        -------
        float
        
        """
        return (vm-vp)*(self.cb2-vm*vp)/(3*self.cb2*vm*(1-vp**2))
    
    def wFromAlpha(self, alpha: float) -> float:
        """
        Computes w(Tp) from alpha(Tp).

        Parameters
        ----------
        alpha : float
            alpha(Tp)

        Returns
        -------
        float
        
        """
        if self.nu-self.mu+3*self.mu*alpha == 0:
            return np.inf
        return (self.nu-self.mu+3*self.mu*self.alN)/(self.nu-self.mu+3*self.mu*alpha)
    
    def shockDE(self, v: np.ndarray, xiAndW: np.ndarray, shockWave: bool=True) -> list:
        r"""
        Hydrodynamic equations for the self-similar coordinate :math:`\xi = r/t` and
        the fluid temperature :math:`T` in terms of the fluid velocity :math:`v`
        See e.g. eq. (B.10, B.11) of 1909.10040

        Parameters
        ----------
        v : array
            Fluid velocities.
        xiAndW : array
            Values of the self-similar coordinate :math:`\xi = r/t` and
            the enthalpy :math:`w`
        shockWave : bool, optional
            If True, the integration happens in the shock wave. If False, it happens in
            the rarefaction wave. Default is True.

        Returns
        -------
        eq1, eq2 : array
            The expressions for :math:`\frac{\partial \xi}{\partial v}`
            and :math:`\frac{\partial w}{\partial v}`
        """
        xi, w = xiAndW[:2]

        if shockWave:
            csq = self.cs2
        else:
            csq = self.cb2
        eq1 = (
            self.gammaSq(v)
            * (1.0 - v * xi)
            * (self.boostVelocity(xi, v) ** 2 / csq - 1.0)
            * xi
            / 2.0
            / v
        )
        eq2 = w * (1 + 1/csq) * self.gammaSq(v) * self.boostVelocity(xi, v)
        
        if xiAndW.shape[0] == 2:
            return [eq1, eq2]
        
        n = xiAndW[2]
        eq3 = n*self.gammaSq(v)*self.boostVelocity(xi, v)/csq
        return [eq1, eq2, eq3]
    
    def integratePlasma(self,
                        vw: float,
                        v0: float,
                        w0: float,
                        shockWave: bool=True,
                        n0: float|None=None
                        ) -> integrate._ivp.ivp.OdeResult:
        """
        Integrate the fluid equations through the plasma around the wall.

        Parameters
        ----------
        vw : float
            Wall velocity.
        v0 : float
            Initial plasma velocity (in the wall frame).
        w0 : float
            Initial enthalpy.
        shockWave : bool, optional
            If True, the integration happens in the shock wave. If False, it happens in
            the rarefaction wave. Default is True.
        n0 : float or None, optional
            If a value is provided, also integrates the number density. Default is None.

        Returns
        -------
        integrate._ivp.ivp.OdeResult
        
        """
        v0BubbleFrame = self.boostVelocity(vw, v0)
        if shockWave:
            csq = self.cs2
        else:
            csq = self.cb2
        
        def shockEvent(v, xiAndW, shockWave=True):
            xi, w = xiAndW[:2]
            return xi*self.boostVelocity(xi, v)-csq
        shockEvent.terminal = True
        
        xIni = [vw, w0] if n0 is None else [vw, w0, n0]
        if abs(v0*vw-csq) < 1e-15:
            fluidProfile = optimize.OptimizeResult(success=True,
                                                   t=np.array([v0BubbleFrame]),
                                                   y=np.transpose([xIni]))
        else:
            fluidProfile = integrate.solve_ivp(self.shockDE,
                                               (v0BubbleFrame, np.sign(v0BubbleFrame)*1e-100),
                                               xIni,
                                               args=(shockWave,),
                                               events=shockEvent,
                                               rtol=self.rtol,
                                               atol=self.atol)
        return fluidProfile
    
    def eqFrontWave(self, vm: float, vp: float, vw: float) -> float:
        """
        Returns the residual of the matching equation at the shock front.

        Parameters
        ----------
        vm : float
            Plasma velocity behind the wall.
        vp : float
            Plasma velocity in front of the wall.
        vw : float
            Wall velocity.

        Returns
        -------
        float
    
        """
        alpha = self.alphaFromV(vm, vp)
        w0 = self.wFromAlpha(alpha)
        
        fluidProfile = self.integratePlasma(vw, vp, 1)
        shockVelocity = fluidProfile.y[0,-1]
        vmShockBubbleFrame = fluidProfile.t[-1]
        wmShock = fluidProfile.y[1,-1]
        vmShock = self.boostVelocity(shockVelocity, vmShockBubbleFrame)
        
        return shockVelocity*self.gammaSq(shockVelocity)/w0 - wmShock*vmShock*self.gammaSq(vmShock)
    
    def matchingNoFrontWave(self, vw: float) -> MatchingResult:
        """
        Solves the matching and fluid equations for a solution without front wave.

        Parameters
        ----------
        vw : float
            Wall velocity.

        Returns
        -------
        MatchingResult
        
        """
        
        X = self.cb2+vw**2-3*self.cb2*self.alN*(1-vw**2)
        if X**2-4*self.cb2*vw**2 < 0:
            return MatchingResult(None, None, None, None, None, False)
        
        vm1 = 0.5*(X+np.sqrt(X**2-4*self.cb2*vw**2))/vw
        vm2 = 0.5*(X-np.sqrt(X**2-4*self.cb2*vw**2))/vw
        
        vp = vw
        wp = 1
        vm: float
        wm: float
        
        if self.cb < vm1 < 1:
            vm = vm1
        elif self.cb < vm2 < 1:
            vm = vm2
        else:
            return MatchingResult(None, None, None, None, None, False)
        
        wm = self.gammaSq(vp)*vp*wp/(vm*self.gammaSq(vm))
        backWaveProfile = self.integratePlasma(vw, vm, wm, False, 1)
        
        return MatchingResult(vw, vp, vm, wp, wm, True, backWaveProfile=backWaveProfile)
    
    def matchingWithFrontWave(self, vw: float) -> MatchingResult:
        """
        Solves the matching and fluid equations for a solution with a front wave.

        Parameters
        ----------
        vw : float
            Wall velocity.

        Returns
        -------
        MatchingResult
        
        """
        
        # Compute the bounds on vm or vp
        vpBounds1, vpBounds2, vmBounds1, vmBounds2 = self.velocityBounds(vw)
        
        ##############################################################################
        ## First consider the case where vm is fixed to min(cb,vw) and vp is varied ##
        ##############################################################################
        vm = min(self.cb, vw)
        vp = None
        
        if vpBounds1[0] != vpBounds1[1] and self.eqFrontWave(vm, vpBounds1[0], vw)*self.eqFrontWave(vm, vpBounds1[1], vw) <= 0:
            vp = optimize.root_scalar(lambda x: self.eqFrontWave(vm, x, vw), bracket=vpBounds1.tolist(), xtol=self.atol, rtol=self.rtol).root
        
        # Problem with alpha_n=-0.05 and vw=0.4
        elif vpBounds2[0] != vpBounds2[1] and self.eqFrontWave(vm, vpBounds2[0], vw)*self.eqFrontWave(vm, vpBounds2[1], vw) <= 0:
            vp = optimize.root_scalar(lambda x: self.eqFrontWave(vm, x, vw), bracket=vpBounds2.tolist(), xtol=self.atol, rtol=self.rtol).root
        
        if vp is not None:
            alpha = self.alphaFromV(vm, vp)
            wp = self.wFromAlpha(alpha)
            wm = self.gammaSq(vp)*vp*wp/(vm*self.gammaSq(vm))
            frontWaveProfile = self.integratePlasma(vw, vp, wp, True, 1)
            backWaveProfile = self.integratePlasma(vw, vm, wm, False, 1) if vm != vw else None
            return MatchingResult(vw, vp, vm, wp, wm, True, frontWaveProfile=frontWaveProfile, backWaveProfile=backWaveProfile)
        
        ###########################################################################################
        ## If no solution were found, consider the case where vp is fixed to cs and vm is varied ##
        ###########################################################################################
        
        if vw < self.cs and not self.fastCompute:
            # The wall must be subsonic for this solution to be valid
            vm = None
            vp = self.cs
            
            if vmBounds1[0] != vmBounds1[1] and self.eqFrontWave(vmBounds1[0], vp, vw)*self.eqFrontWave(vmBounds1[1], vp, vw) <= 0:
                vm = optimize.root_scalar(lambda x: self.eqFrontWave(x, vp, vw), bracket=vmBounds1.tolist(), xtol=self.atol, rtol=self.rtol).root
                
            elif vmBounds2[0] != vmBounds2[1] and self.eqFrontWave(vmBounds2[0], vp, vw)*self.eqFrontWave(vmBounds2[1], vp, vw) <= 0:
                vm = optimize.root_scalar(lambda x: self.eqFrontWave(x, vp, vw), bracket=vmBounds2.tolist(), xtol=self.atol, rtol=self.rtol).root
            
            if vm is not None:
                alpha = self.alphaFromV(vm, vp)
                wp = self.wFromAlpha(alpha)
                wm = self.gammaSq(vp)*vp*wp/(vm*self.gammaSq(vm))
                frontWaveProfile = self.integratePlasma(vw, vp, wp, True, 1)
                backWaveProfile = self.integratePlasma(vw, vm, wm, False, 1) if vm != vw else None
                return MatchingResult(vw, vp, vm, wp, wm, True, frontWaveProfile=frontWaveProfile, backWaveProfile=backWaveProfile)
        
        # If no solution were found, return an empty MatchingResult object.
        return MatchingResult(None, None, None, None, None, False)
    
    def findMatching(self, vw: float) -> MatchingResult:
        """
        Solves the matching and fluid equations for a general solution.

        Parameters
        ----------
        vw : float
            Wall velocity.

        Returns
        -------
        MatchingResult
        
        """
        if vw < self.cs or (self.vJDirect is not None and vw < self.vJDirect):
            matching = self.matchingWithFrontWave(vw)
        else:
            matching = self.matchingNoFrontWave(vw)
        self.computeTemperatures(matching)
        return matching
    
    def velocityBounds(self, vw: float):
        """
        Bounds on vp and vm at a given wall velocity. They are constrained by the fact
        that alpha+ can only take values between (mu-nu)/(3*mu) and +/-inf. The matching
        equation alpha+ = (vm-vp)*(cb2-vm*vp)/(3*cb2*(1-vp**2)) then gives constraints
        on vp and vm.
        
        There are in general 4 regions where solutions are possible, which are returned
        in vpBounds1 and vpBounds2 (assuming vm = min(cb,vw)) and in vmBounds1 and
        vmBounds2 (assuming vp=min(cs,cs2/vw)).
        

        Parameters
        ----------
        vw : float
            Wall velocity.

        Returns
        -------
        vpBounds1, vpBounds2, vmBounds1, vmBounds2

        """
        alM = (self.mu-self.nu)/(3*self.mu)
        
        ##################################
        ######### Bounds on vp ###########
        ##################################
        vm = min(self.cb, vw)
        discr = (self.cb2+vm**2)**2-4*self.cb2*vm*(1-3*alM)*(vm+3*self.cb2*vm*alM)
        
        # In general, vp can have 2 allowed regions, constrained by vpBounds1 and vpBounds2
        vpBounds1 = np.array([0,0])
        vpBounds2 = np.array([0,0])
        if discr < 0 and self.epsilonSign >= 0:
            vpBounds2 = np.array([0,1])
        elif discr >= 0:
            vpBounds = np.sort([(self.cb2+vm**2-np.sqrt(discr))/(2*vm*(1+3*self.cb2*alM)),
                                (self.cb2+vm**2+np.sqrt(discr))/(2*vm*(1+3*self.cb2*alM))])
            if self.epsilonSign < 0:
                # The allowed region is inside vpBounds
                vpBounds2 = vpBounds
            else:
                # The allowed region is outside vpBounds
                vpBounds1 = np.array([-np.inf,vpBounds[0]])
                vpBounds2 = np.array([vpBounds[1],np.inf])
        vpBounds1 = np.maximum(0, np.minimum(min(self.cs,self.cs2/vw), vpBounds1))
        vpBounds2 = np.maximum(0, np.minimum(min(self.cs,self.cs2/vw), vpBounds2))
        
        ##################################
        ######### Bounds on vm ###########
        ##################################
        vp = self.cs
        discr = (self.cb2+vp**2-3*self.cb2*(1-vp**2)*alM)**2-4*self.cb2*vp**2
        
        # In general, vm can have 2 allowed regions, constrained by vmBounds1 and vmBounds2
        vmBounds1 = np.array([1,1])
        vmBounds2 = np.array([1,1])
        if discr < 0 and self.epsilonSign < 0:
            vmBounds1 = np.array([0,1])
        elif discr >= 0:
            vmBounds = np.sort([(self.cb2+vp**2-3*self.cb2*alM*(1-vp**2)-np.sqrt(discr))/(2*vp),
                                (self.cb2+vp**2-3*self.cb2*alM*(1-vp**2)+np.sqrt(discr))/(2*vp)])
            if self.epsilonSign >= 0:
                # The allowed region is inside vmBounds
                vmBounds1 = vmBounds
            else:
                # The allowed region is outside vmBounds
                vmBounds1 = np.array([-np.inf,vmBounds[0]])
                vmBounds2 = np.array([vmBounds[1],np.inf])
        vmBounds1 = np.maximum(max(self.cb, min(1, self.cb2/vw)), np.minimum(1, vmBounds1))
        vmBounds2 = np.maximum(max(self.cb, min(1, self.cb2/vw)), np.minimum(1, vmBounds2))
        
        return vpBounds1, vpBounds2, vmBounds1, vmBounds2
    
    def alphaN(self, epsilon: float) -> float:
        """
        Computes alpha(Tn) as a function of epsilon (the vacuum energy).

        Parameters
        ----------
        epsilon : float
            Vacuum energy

        Returns
        -------
        float
        
        """
        return (self.mu-self.nu)/(3*self.mu)+self.nu*epsilon/3
    
    def computeTemperatures(self, matching: MatchingResult):
        """
        Computes the temperatures from the solution stored in matching.

        Parameters
        ----------
        matching : MatchingResult
            MatchingResult object containing the solution of the fluid equations.
        
        """
        if matching.foundSolution:
            matching.Tp = self.Tn*matching.wp**(1/self.mu)
            matching.Tm = self.Tn*(matching.wm/self.psiN)**(1/self.nu)
        else:
            matching.Tp = None
            matching.Tm = None
    
    """
    The following functions are for finding the wall velocity in LTE.
    """
    
    def entropy(self, vw: float, sigma: callable|None=None) -> float:
        """
        Computes the entropy ratio generated at the wall.

        Parameters
        ----------
        vw : float
            Wall Velocity
        sigma : callable | None, optional
            Desired entropy ratio which is substracted to the result. Must be a
            function taking a MatchingResult and returning a float. Can also be
            None, in which case it is set to 0. Default is None.

        Returns
        -------
        float
        """
        if sigma is None:
            sigma = lambda x: 0
        matching = self.findMatching(vw)

        if not matching.foundSolution:
            return None
        
        sp = matching.wp/matching.Tp
        sm = matching.wm/matching.Tm
        
        return ((matching.vm*sm/(matching.vp*sp))*np.sqrt((1-matching.vp**2)/(1-matching.vm**2)) 
                - 1 - sigma(matching))
    
    def findVwLTE(self, sigma: callable|None=None) -> list[float]:
        """
        Computes the wall velocity in LTE. If sigma is specified, finds the solutions
        which produce an entropy ratio sigma. Returns a list containing all the solutions found.

        Parameters
        ----------
        sigma : callable | None, optional
            Desired entropy ratio. Must be a
            function taking a MatchingResult and returning a float. Can also be
            None, in which case it is set to 0. Default is None.

        Returns
        -------
        list[float]
            List containing all the solutions found.
        """
        solutions = []
        if sigma is None:
            sigma = lambda match: 0
        eps = 1e-6
        if self.vLowFrontWave+eps < self.vHighVmBC-eps:
            if self.fastCompute:
                vwDefl = self.fastFindDeflagVwLTE(sigma)
                if vwDefl is not None:
                    solutions.append(vwDefl)
            elif self.entropy(self.vLowFrontWave+eps, sigma) * self.entropy(self.vHighVmBC-eps, sigma) <= 0:
                solutions.append(optimize.root_scalar(self.entropy,
                                                      bracket=(self.vLowFrontWave+eps, self.vHighVmBC-eps),
                                                      args=(sigma,)).root)
        
        if self.vLowVpBC+eps < self.vLowNoFrontWave-eps:
            if self.entropy(self.vLowVpBC+eps, sigma) * self.entropy(self.vLowNoFrontWave-eps, sigma) <= 0:
                solutions.append(optimize.root_scalar(self.entropy,
                                                      bracket=(self.vLowVpBC+eps, self.vLowNoFrontWave-eps),
                                                      args=(sigma,)).root)
                
        if self.vLowNoFrontWave+eps < 1-eps:
            if self.entropy(self.vLowNoFrontWave+eps, sigma) * self.entropy(1-eps, sigma) <= 0:
                solutions.append(optimize.root_scalar(self.entropy,
                                                      bracket=(self.vLowNoFrontWave+eps, 1-eps),
                                                      args=(sigma,)).root)
            
        return solutions
    
    ########################################################
    ### Faster functions used when self.fastCompute=True ###
    ########################################################

    def fastVpFromVw(self, vw: float, sigma: callable|None=None) -> float:
        """
        Computes vp as a function of vw assuming conservation of
        entropy, or entropy ratio sigma if it is provided. Used for
        deflagrations or hybrid solutions.

        Parameters
        ----------
        vw : float
            Wall velocity
        sigma : callable | None, optional
            Desired entropy ratio. Must be a
            function taking a MatchingResult and returning a float. Can also be
            None, in which case it is set to 0. Default is None.

        Returns
        -------
        float

        """
        assert self.fastCompute, "Error: fastVpFromVw can only be used when self.fastCompute==True."

        vm = min(vw, self.cb)

        if abs(self.cb2-1/3) < 1e-10 and sigma is None:
            return 2*np.sin(np.arcsin(0.5*3**1.5*vm*(1-vm**2)*self.psiN)/3)/np.sqrt(3)
        
        gmsq = self.gammaSq(vm)
        
        def psiEff(vp):
            # Computes psi_eff in case sigma is provided
            if sigma is None:
                return self.psiN
            wp = self.wFromAlpha(self.alphaFromV(vm, vp))
            wm = vp*wp/(vm*gmsq*(1-vp**2))
            matching = MatchingResult(vw, vp, vm, wp, wm, True)
            self.computeTemperatures(matching)
            sig = sigma(matching)
            return self.psiN/(1+sig)**self.nu
        
        def func(vp):
            gpsq = self.gammaSq(vp)
            return vp - vm*(gpsq/gmsq)**(0.5*(1/self.cb2-1))*psiEff(vp)
        
        return optimize.root_scalar(func, bracket=[0, vm], xtol=self.atol, rtol=self.rtol).root
    
    def fastEqFrontWave(self, vw: float, sigma: callable|None=None) -> float:
        """
        Returns the residual of the matching equation at the shock front for the fast algorithm.

        Parameters
        ----------
        vw : float
            Wall velocity
        sigma : callable | None, optional
            Desired entropy ratio. Must be a
            function taking a MatchingResult and returning a float. Can also be
            None, in which case it is set to 0. Default is None.

        Returns
        -------
        float
    
        """
        vm = min(vw, self.cb)
        vp = self.fastVpFromVw(vw, sigma)
        return self.eqFrontWave(vm, vp, vw)
    
    def fastFindDeflagVwLTE(self, sigma: callable|None=None) -> float:
        """
        Computes the wall velocity for deflagration in LTE using the fast algorithm. 
        If sigma is specified, finds the solutions which produce an entropy ratio sigma.

        Parameters
        ----------
        sigma : callable | None, optional
            Desired entropy ratio. Must be a
            function taking a MatchingResult and returning a float. Can also be
            None, in which case it is set to 0. Default is None.

        Returns
        -------
        float
        
        """
        eps = 1e-6

        if self.fastEqFrontWave(self.vLowFrontWave+eps)*self.fastEqFrontWave(self.vHighVmBC-eps):
            return optimize.root_scalar(self.fastEqFrontWave,
                                        bracket=[self.vLowFrontWave+eps, self.vHighVmBC-eps],
                                        xtol=self.atol,
                                        rtol=self.rtol).root
        return None