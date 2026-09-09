"""
Example to compute the relaxation time for the left and right handed
top quarks and the SU(2) gauge bosons, including QCD and weak interactions.

We work in units where Tn = 1.

This file uses the package WallGo. Please cite 2411.04970 and 2510.27691
if you use it.
"""

import numpy as np
from pathlib import Path
from WallGo import Fields, Grid, Particle, BoltzmannBackground, BoltzmannSolver

# Path to the directory containing the collision integrals
# These can be computed with WallGoCollision
collisionPath = Path('/Users/benoitlaurent/Documents/git/boltzmann/EntropyProduction/collisions_tests_new/CollisionOutput_N11_massive+2')

# Create the Particle objects

# Left and right handed top quark
def topMsqVacuum(fields: Fields) -> Fields:
    return (173*fields.getField(0)/246)**2
def topMsqDerivative(fields: Fields) -> Fields:
    return np.transpose(
        [2*(173/246)**2*fields.getField(0), 0 * fields.getField(1)]
    )

topQuarkL = Particle(
    name="TopL",
    index=0,
    msqVacuum=topMsqVacuum,
    msqDerivative=topMsqDerivative,
    statistics="Fermion",
    totalDOFs=6,
)

topQuarkR = Particle(
    name="TopR",
    index=1,
    msqVacuum=topMsqVacuum,
    msqDerivative=topMsqDerivative,
    statistics="Fermion",
    totalDOFs=6,
)

# SU(2) gauge boson
def WMsqVacuum(fields: Fields) -> Fields:  # pylint: disable=invalid-name
    return (80.37/246)**2 * fields.getField(0) ** 2

def WMsqDerivative(fields: Fields) -> Fields:  # pylint: disable=invalid-name
    return np.transpose(
        [2*(80.37/246)**2*fields.getField(0), 0 * fields.getField(1)]
    )

wBoson = Particle(
    name="W",
    index=6,
    msqVacuum=WMsqVacuum,
    msqDerivative=WMsqDerivative,
    statistics="Boson",
    totalDOFs=9,
)

particleList = [topQuarkL, topQuarkR, wBoson]

# Thermal masses used to regularize the integrals. Only used for the gauge bosons.
thermalMassesSq = [0, 0, 0.3914591185306366]

# Yukawa couplings
yukawa = np.sqrt(2)*np.array([173/246, 173/246, 80.37/246])

# Define WallGo objects
N = 11 # This must correspond to the N used to compute the collision operators
grid = Grid(20, N, 10, 1)
boltzmann = BoltzmannSolver(grid, basisN='Cardinal')
background = BoltzmannBackground(0, np.zeros(grid.M+1), np.zeros((grid.M+1,1)), grid.momentumFalloffT*np.ones(grid.M+1))
boltzmann.setBackground(background)
boltzmann.updateParticleList(particleList)
boltzmann.loadCollisions(collisionPath)

# Load the collision operator and compute its inverse
_,_,_,collision = boltzmann.buildLinearEquations()
size = len(particleList)*(grid.M-1)*(grid.N-1)**2
collisionFlat = collision.reshape(
    (len(particleList), grid.M-1, grid.N-1, grid.N-1, size)
    ).reshape((size, size))
collisionFlatInv = np.linalg.inv(collisionFlat)
collisionInv = collisionFlatInv.reshape(
    (len(particleList), grid.M-1, grid.N-1, grid.N-1, size)
    ).reshape((len(particleList), grid.M-1, grid.N-1, grid.N-1, len(particleList), grid.M-1, grid.N-1, grid.N-1))

# Create a few momentum-dependent functions
temperature = background.temperatureProfile[None,1:-1,None,None]
thermalMassesSq = temperature**2*(np.array(thermalMassesSq)[:,None,None,None])
E = np.sqrt(grid.pzValues[None,None,:,None]**2+grid.ppValues[None,None,None,:]**2+thermalMassesSq)
EMassless = np.sqrt(grid.pzValues[None,None,:,None]**2+grid.ppValues[None,None,None,:]**2)
statistics = np.array(
    [-1 if particle.statistics == "Fermion" else 1 for particle in particleList]
)[:, None, None, None]
dfeqMsq = yukawa[:,None,None,None]**2*BoltzmannSolver._dfeq(E/temperature, statistics)

# delta f computed with the full collision operator and with the RTA
deltaFFull = np.sum(collisionInv*dfeqMsq[None,None,None,None,:,:,:,:], axis=(4,5,6,7))
deltaFRTA = dfeqMsq*EMassless/E**2/temperature

# Integrating delta f to get the pressure
PFull = boltzmann.getDeltas(EMassless*deltaFFull/E).Deltas.Delta00.coefficients[:,0]
PRTA = boltzmann.getDeltas(deltaFRTA).Deltas.Delta00.coefficients[:,0]

# Take the ratio of the pressures to get tau
taus = PFull/PRTA

print('Relaxation times (in units of 1/T):', taus)