# HydrodynamicsFOPT
Module for solving hydrodynamics equation during a first-order phase transition

## Features

- Finds the solution of the fluid and matching equations.
- Uses the template model to approximate the EOS.
- Finds the LTE wall velocity.
- Can also find the out-of-equilibrium wall velocity if a desidered entropy production fraction is provided.
- Can be used for direct and inverse FOPTs.


## Installation

```bash
pip install -e .
```

## Example

```python
# Example to compute the wall velocity in LTE
# alN and psiN are defined in Eq. (19) of 2303.10171.
# The code returns all the static solutions.
# Note that the condition alN > (1-psiN)/3 must be satisfied to have a consistent solution.

from HydrodynamicsFOPT import Hydrodynamics
hydro = Hydrodynamics(alN=0.0067, psiN=0.98)
print(hydro.findVwLTE())
```

## Citation

This package is free and open-source. If you use it in your research, please cite [2303.10171](https://arxiv.org/abs/2303.10171), on which the main part of this code is based.
If you use the functions to compute the entropy production in ```/src/HydrodynamicsFOPT/entropyProduction/```, please also cite [2411.13641](https://arxiv.org/abs/2411.13641) and 26xx.xxxxx.

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.22
- SciPy ≥ 1.10

## License

MIT License — © 2025 Benoit Laurent
