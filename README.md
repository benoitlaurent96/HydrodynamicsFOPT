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

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.22
- SciPy ≥ 1.10

## License

MIT License — © 2025 Benoit Laurent
