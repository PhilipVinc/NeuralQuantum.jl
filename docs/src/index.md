
# NeuralQuantum.jl

*A Neural-Network steady-state solver*

**NeuralQuantum.jl** is a numerical framework written in [Julia](http://julialang.org)
to investigate Neural-Network representations of mixed quantum states and to
find the Steady-State of such NonEquilibrium Quantum Systems by MonteCarlo
sampling.

!!! note
    Please note that the code is research code and is not production ready, yet.

## Installation

To install `NeuralQuantum`, run in a Julia prompt the following
command.
```
] add QuantumOptics#master
] add https://github.com/PhilipVinc/QuantumLattices.jl
] add https://github.com/PhilipVinc/ValueHistoriesLogger.jl
] add https://github.com/PhilipVinc/NeuralQuantum.jl
```

## Basic Usage
```@meta
CurrentModule = NeuralQuantum
```

Once one has a Liouvillian problem of which one wants to compute the steady state, the underlaying idea of the package is the following:
 - One selects a Neural-Network based ansatz to approximate the density matrix (see Sec. [Networks](@ref));
 - One choses the algorithm to compute the gradient with which to minimise the objective function (see Sec. [Algorithms](@ref));
 - One choses an optimizer to perform the optimization, such as steepest gradient, accelerated gradient or others (see Sec. [Optimizers](@ref));

Here you can find a very short usage example. For a more in-depth walkthrough of
`NeuralQuantum.jl` please refer to Sec. [Basics](@ref).

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

## Table Of Contents
```@contents
```



## Main Functions

```@docs
solve
```

```@docs
ExactSamplerCache
```


## Index
```@index
```
