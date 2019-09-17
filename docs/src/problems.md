# Problems

A `Problem` encodes a many-body quantum problem into a cost function to be
minimized. Problems can encode the search for the Ground State energy of an
hamiltonian of for the Steady-State of an Open Quantum System.

## Steady States of Open Quantum Systems

A `SteadyStateProblem` encodes the steady-state of a liouvillian into the
global minimum of the following cost function

``\mathcal{C} = \langle\langle\mathcal{L}^\dagger\mathcal{L}\rangle\rangle``

When stochastically sampling this cost function on the hilbert space, we can
evaluate it in two possible ways:
```math
\mathcal{C} = \sum_\sigma p(\sigma) \langle\langle\sigma |\mathcal{L}^\dagger\mathcal{L}\rho\rangle\rangle
```
or (variance style)
```math
\mathcal{C} = \sum_\sigma p(\sigma) |\langle\langle\sigma |\mathcal{L}\rho\rangle\rangle|^2
```

The second version leads to smaller variance of sampled variables and also is
faster to evaluate because it holds only ``\mathcal{L}`` instead of ``\mathcal{L}^\dagger\mathcal{L}``, as such I reccomend to use this one.
To use it, set `variance=true` when costructing the problem.

```@docs
SteadyStateProblem
```

## Ground State Problems
A `GroundStateProblem` encodes the ground-state of an Hamiltonian into the
global minimum of the total energy

``\mathcal{C} = \langle\psi|\hat{H}|\psi\rangle``

To construct a `GroundStateProblem` you must supply it a valid hamiltonian.

```@docs

GroundStateProblem
```
