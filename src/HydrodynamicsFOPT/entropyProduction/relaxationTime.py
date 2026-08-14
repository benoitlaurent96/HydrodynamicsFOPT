import numpy as np
from pathlib import Path

from WallGo import Grid, Particle, BoltzmannBackground, BoltzmannSolver

def estimateTau(file: Path, grid: Grid, particleList: list[Particle]):
    """
    This function estimates the relaxation time tau needed to compute
    the entropy production. It uses the collision files computed by WallGo.
    If you use it, please cite the WallGo paper 2411.04970 and 2510.27691.

    Parameters
    ----------
    file : Path
        Path to the directory containing the collision files computed by
        WallGo.
    grid : Grid
        Grid object used to compute the collision files.
    particleList : list[Particle]
        List of Particle object containing the properties of all the
        out-of-equilibrium species.
    """
    boltzmann = BoltzmannSolver(grid, basisN='Cardinal')
    background = BoltzmannBackground(0, np.zeros(grid.M+1), np.zeros((grid.M+1,1)), grid.momentumFalloffT*np.ones(grid.M+1))
    boltzmann.setBackground(background)
    boltzmann.updateParticleList(particleList)
    boltzmann.loadCollisions(file)
    
    _,_,_,collision = boltzmann.buildLinearEquations()
    size = len(particleList)*(grid.M-1)*(grid.N-1)**2
    collisionFlat = collision.reshape(
        (len(particleList), grid.M-1, grid.N-1, grid.N-1, size)
        ).reshape((size, size))
    collisionFlatInv = np.linalg.inv(collisionFlat)
    collisionInv = collisionFlatInv.reshape(
        (len(particleList), grid.M-1, grid.N-1, grid.N-1, size)
        ).reshape((len(particleList), grid.M-1, grid.N-1, grid.N-1, len(particleList), grid.M-1, grid.N-1, grid.N-1))
    
    temperature = background.temperatureProfile[None,1:-1,None,None]
    msq = np.array(
        [
            particle.msqVacuum(background.fieldProfiles)
            for particle in particleList
        ]
    )[:,1:-1,None,None]
    E = np.sqrt(grid.pzValues[None,None,:,None]**2+grid.ppValues[None,None,None,:]**2)
    statistics = np.array(
        [-1 if particle.statistics == "Fermion" else 1 for particle in particleList]
    )[:, None, None, None]
    dfeqMsq = msq*BoltzmannSolver._dfeq(E/temperature, statistics)
    
    deltaFTrue = np.sum(collisionInv*dfeqMsq[None,None,None,None,:,:,:,:], axis=(4,5,6,7))
    deltaFRel = dfeqMsq/E/temperature
    Delta00True = boltzmann.getDeltas(deltaFTrue).Deltas.Delta00.coefficients[:,0]
    Delta00Rel = boltzmann.getDeltas(deltaFRel).Deltas.Delta00.coefficients[:,0]
    
    return Delta00True/Delta00Rel