# NeuralQuantum
[![Build Status](https://travis-ci.org/PhilipVinc/NeuralQuantum.jl.svg?branch=master)](https://travis-ci.org/PhilipVinc/NeuralQuantum.jl) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://philipvinc.github.io/NeuralQuantum.jl/dev) [![DOI](https://zenodo.org/badge/186389926.svg)](https://zenodo.org/badge/latestdoi/186389926)

**NeuralQuantum** is a numerical framework written in [Julia] to investigate
Neural-Network representations of mixed quantum states and to find the Steady-
State of dissipative Quantum Systems with variational Montecarlo schemes.

This code has been developed while working on [Variational neural network ansatz for steady states in open quantum systems](https://arxiv.org/abs/1902.10104), by [F. Vicentini] et al. Phys Rev Lett in press (2019).

## Installation
To Install `NeuralQuantum.jl`, run the following commands to install all
dependencies:
```
using Pkg
pkg"add https://github.com/PhilipVinc/QuantumLattices.jl"
pkg"add https://github.com/PhilipVinc/ValueHistoriesLogger.jl"
pkg"add https://github.com/PhilipVinc/NeuralQuantum.jl"
```
Those packages are needed for the following reasons:
 - `QuantumLattices` is a custom package that allows defining new types of operators on a lattice. It's not needed natively but it is usefull to define hamiltonians on a lattice.
 - `ValueHistoriesLogger` custom logger for logging arbitrary values

## Example
```
# Load dependencies
using NeuralQuantum, QuantumLattices
using Printf, ValueHistoriesLogger, Logging, ValueHistories

# Select the numerical precision
T      = Float64
# Select how many sites you want
Nsites = 6

# Create the lattice as [Nx, Ny, Nz]
lattice = SquareLattice([Nsites],PBC=true)
# Create the lindbladian for the QI model
lind = quantum_ising_lind(lattice, g=1.0, V=2.0, γ=1.0)
# Create the Problem (cost function) for the given lindbladian
# alternative is LdagL_L_prob. It works for NDM, not for RBM
prob = LdagL_spmat_prob(T, lind);

#-- Observables
# Define the local observables to look at.
Sx  = QuantumLattices.LocalObservable(lind, sigmax, Nsites)
Sy  = QuantumLattices.LocalObservable(lind, sigmay, Nsites)
Sz  = QuantumLattices.LocalObservable(lind, sigmaz, Nsites)
# Create the problem object with all the observables to be computed.
oprob = ObservablesProblem(Sx, Sy, Sz)


# Define the Neural Network. A NDM with N visible spins and αa=2 and αh=1
#alternative vectorized rbm: net  = RBMSplit(Complex{T}, Nsites, 6)
net  = rNDM(T, Nsites, 1, 2)
# Create a cached version of the neural network for improved performance.
cnet = cached(net)
# Chose a sampler. Options are FullSumSampler() which sums over the whole space
# ExactSampler() which does exact sampling or MCMCSamler which does a markov
# chain.
# This is a markov chain of length 1000 where the first 50 samples are trashed.
sampl = MCMCSampler(Metropolis(), 1000, burn=50)
# Chose a sampler for the observables.
osampl = FullSumSampler()

# Chose the gradient descent algorithm (alternative: Gradient())
# for more information on options type ?SR
algo  = SR(ϵ=T(0.01), use_iterative=true)
# Optimizer: how big the steps of the descent should be
optimizer = Optimisers.Descent(0.02)

# Create a multithreaded Iterative Sampler.
is = MTIterativeSampler(cnet, sampl, prob, algo)
ois = MTIterativeSampler(cnet, osampl, oprob, oprob)

# Create the logger to store all output data
log = MVLogger()

# Solve iteratively the problem
with_logger(log) do
    for i=1:50
        # Sample the gradient
        grad_data  = sample!(is)
        obs_data = sample!(ois)

        # Logging
        @printf "%4i -> %+2.8f %+2.2fi --\t \t-- %+2.5f\n" i real(grad_data.L) imag(grad_data.L) real(obs_data.ObsAve[1])
        @info "" optim=grad_data obs=obs_data

        succ = precondition!(cnet.der.tuple_all_weights, algo , grad_data, i)
        !succ && break
        Optimisers.update!(optimizer, cnet, cnet.der)
    end
end

# Optional: compute the exact solution
ρ   = last(steadystate.master(lind)[2])
ESx = real(expect(SparseOperator(Sx), ρ))
ESy = real(expect(SparseOperator(Sy), ρ))
ESz = real(expect(SparseOperator(Sz), ρ))
exacts = Dict("Sx"=>ESx, "Sy"=>ESy, "Sz"=>ESz)
## - end Optional

using Plots
data = log.hist

iter_cost, cost = get(data["optim/Loss"])
pl1 = plot(iter_cost, real(cost), yscale=:log10);

iter_mx, mx = get(data["obs/obs_1"])
pl2 = plot(iter_mx, mx);
hline!(pl2, [ESx,ESx]);

plot(pl1, pl2, layout=(2,1))
...
```

[Julia]: http://julialang.org
[F. Vicentini]: mailto:filippo.vicentini@univ-paris-diderot.fr
[Filippo Vicentini]: mailto:filippo.vicentini@univ-paris-diderot.fr
