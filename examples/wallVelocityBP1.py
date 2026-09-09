"""
Example where we compute the wall velocity using various approximations.
The model considered here corresponds to BP1 in 2609.xxxxx.
"""

from HydrodynamicsFOPT import Hydrodynamics, EntropyNLTE, EntropyBallistic, EntropyInterpolated

# WallGo result
vwWG = 0.347

# Hydrodynamics parameters corresponding to BP1
Tn = 121.48
vn = 176.0041332244873
alN = 0.004777442806825361
psi = 0.9877384069928336
cb2 = 0.3307916625223334
cs2 = 0.33334191365999344
gstar = 107.75
wallWidth = 0.037086114257975665 #TODO: Update this!!!

# Creating the Hydrodynamics object
hydro = Hydrodynamics(alN, cb2, cs2, psi, Tn)

# Properties of the out-of-equilibrium species (topL, topR and W)
# Masses
mTopL = 173*vn/246
mTopR = 173*vn/246
mW = 80.34*vn/246

# Degrees of freedom
dofTopL = 6
dofTopR = 6
dofW = 9

# Statistics (1 for fermion, -1 for boson)
statisticTopL = 1
statisticTopR = 1
statisticW = -1

# Relaxation time tau (in units of 1/Tn)
tauTopL = 20.79132841
tauTopR = 58.98017835
tauW = 80.25163104

# Creating the Entropy objects
entropyNLTE = EntropyNLTE(massesSymmetricPhase=[0, 0, 0],
                          massesBrokenPhase=[mTopL, mTopR, mW],
                          dofs=[dofTopL, dofTopR, dofW],
                          statistics=[statisticTopL, statisticTopR, statisticW],
                          tau=[tauTopL, tauTopR, tauW],
                          L=wallWidth,
                          Tn=Tn,
                          gstar=gstar,
                          cs2=cs2)
entropyBall = EntropyBallistic(massesSymmetricPhase=[0, 0, 0],
                               massesBrokenPhase=[mTopL, mTopR, mW],
                               dofs=[dofTopL, dofTopR, dofW],
                               statistics=[statisticTopL, statisticTopR, statisticW],
                               Tn=Tn,
                               gstar=gstar,
                               cs2=cs2)
entropyInterp = EntropyInterpolated(massesSymmetricPhase=[0, 0, 0],
                                    massesBrokenPhase=[mTopL, mTopR, mW],
                                    dofs=[dofTopL, dofTopR, dofW],
                                    statistics=[statisticTopL, statisticTopR, statisticW],
                                    tau=[tauTopL, tauTopR, tauW],
                                    L=wallWidth,
                                    Tn=Tn,
                                    gstar=gstar,
                                    cs2=cs2,
                                    interpolateTotal=False)

# Compute all the wall velocities
print('Wall velocities:')
print('LTE:', hydro.findVwLTE())
print('NLTE:', hydro.findVwLTE(entropyNLTE))
print('Ballistic:', hydro.findVwLTE(entropyBall))
print('Interpolated:', hydro.findVwLTE(entropyInterp))
print('WallGo:', vwWG)