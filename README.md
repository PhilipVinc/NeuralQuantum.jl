# NeuralQuantumBase

**NeuralQuantum** is a numerical framework written in [Julia] to investigate
Neural-Network representations of mixed quantum states and to find the Steady-
State of such NonEquilibrium Quantum Systems by montecarlo sampling.

## Installation
To Install `NeuralQuantum.jl`, please run the following commands to install all
dependencies:
```
] add QuantumOptics#master
] add Zygote#master
] add https://github.com/PhilipVinc/QuantumLattices.jl
] add https://github.com/PhilipVinc/ValueHistoriesLogger.jl
] add https://github.com/PhilipVinc/TensorBoardLogger.jl
] add IterativeSolvers
] add https://github.com/PhilipVinc/Optimisers.jl
] add https://github.com/PhilipVinc/NeuralQuantumBase.jl
```
If you also want `MPI` support then you should also install the following
repository. Check it's documentation to run simulations on an `MPI` cluster.
```
] add https://github.com/PhilipVinc/NeuralQuantumMPI.jl
```
If you are wondering what all those packages are for, here's an explanation:
 - `QuantumOptics#master` is needed because some tensor product operations have not yet been released
 - `Zygote#master` is needed for complex-valued AD, which is still experimental
 - `QuantumLattices` is a custom package that allows defining new types of operators on a lattice. It's not needed natively but it is usefull to define hamiltonians on a lattice.
 - `IterativeSolvers`, my branch because it includes the `minres-qlp` solver, which is needed for ill-defined highly singular linear algebra problems found in Natural gradient descent.
 - `Optimisers`, a custom version of the still unreleased `FluxML/Optimisers.jl`, with features that are not yet released in the original branch.
 - `ValueHistoriesLogger` custom logger for logging arbitrary values
 - `TensorBoardLogger` tensorboard support.

## Example
```
using NeuralQuantumBase, Random

# Create the lattice for a 4 site PBC chain
lattice = SquareLattice([4], PBC=true)

# Create the Lindbladian for Quantum ising
lind = quantum_ising_lind(lattice, g=1.0, V=2.0, γ=1.0)

# Select the numerical precision
T = Float64

# Create the problem object associated to finding the minimum of the Frobenius
# norm of L||rho⟩⟩.
prob = LdagLProblem(T, sys)

# Create a Neural Density Matrix representation of the Density Matrix
# for 4 sites, α_hidden = 1 and α_additional = 2
net  = rNDM(T, 4, 1, 2)
net = cached(net)
# Note: using the cached version of a network yield a 100% speed increase, but is not available for all networks.

# Choose a sampling algorithm


# Solve for the steady state
...
```

[Julia]: http://julialang.org
[Filippo Vicentini]: mailto:filippo.vicentini@univ-paris-diderot.fr
