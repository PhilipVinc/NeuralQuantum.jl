
# NeuralQuantum.jl

*A Neural-Network steady-state solver*

**NeuralQuantum.jl** is a numerical framework written in [Julia](http://julialang.org)
to investigate Neural-Network representations of pure and mixed quantum states,
and to find the Steady-State of such Open Quantum Systems through MonteCarlo
procedures.
The package can also be used to compute the ground state of a many-body hamiltonian.

!!! note
    Please note that the code is research code and is not production ready, yet.

## Installation

To install `NeuralQuantum`, run in a Julia prompt the following
command.
```
] add QuantumOptics#master
] add https://github.com/PhilipVinc/QuantumLattices.jl
] add https://github.com/PhilipVinc/NeuralQuantum.jl
```

## Basic Usage
```@meta
CurrentModule = NeuralQuantum
```

When using NeuralQuantum, to determine the Ground State or Steady State of a
many-body problem, one needs to perform the following choices:
 - Chose a Neural-Network based ansatz to approximate the quantum state (see Sec. [Networks](@ref));
 - Chose the algorithm to compute the gradient with which to minimise the objective function, such as simple gradient or natural gradient/Stochastic Reconfiguration (see Sec. [Algorithms](@ref));
 - Chose the optimizer to perform the optimization, such as steepest gradient, accelerated gradient or others (see Sec. [Optimizers](@ref));

Here you can find a very short, commented example. For a more in-depth walkthrough of
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
# targeting the Steady State, using a memory efficient encoding and
# minimizing |Lρ|^2 as a variance, which is more efficient.
prob = SteadyStateProblem(T, lind);

#-- Observables
# Define the local observables to look at.
Sx  = QuantumLattices.LocalObservable(lind, sigmax, Nsites)
Sy  = QuantumLattices.LocalObservable(lind, sigmay, Nsites)
Sz  = QuantumLattices.LocalObservable(lind, sigmaz, Nsites)
# Create the problem object with all the observables to be computed.
oprob = ObservablesProblem(Sx, Sy, Sz)


# Define the Neural Network. A NDM with N visible spins and αa=2 and αh=1
#alternative vectorized rbm: net  = RBMSplit(Complex{T}, Nsites, 6)
net  = NDM(T, Nsites, 1, 2)
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

# Create the structure to store all output data
minimization_data = MVHistory()
Δw = grad_cache(cnet)

# Solve iteratively the problem
for i=1:110
    # Sample the gradient
    grad_data  = sample!(is)
    obs_data = sample!(ois)

    # Logging
    @printf "%4i -> %+2.8f %+2.2fi --\t \t-- %+2.5f\n" i real(grad_data.L) imag(grad_data.L) real(obs_data.ObsAve[1])
    push!(minimization_data, :loss, grad_data.L)
    for (name,val)=zip(obs_data.ObsNames, obs_data.ObsAve)
        push!(minimization_data, Symbol(name), val)
    end

    succ = precondition!(Δw.tuple_all_weights, algo , grad_data, i)
    !succ && break
    Optimisers.update!(optimizer, cnet, Δw)
end

# Optional: compute the exact solution
ρ   = last(steadystate.master(lind)[2])
ESx = real(expect(SparseOperator(Sx), ρ))
ESy = real(expect(SparseOperator(Sy), ρ))
ESz = real(expect(SparseOperator(Sz), ρ))
exacts = Dict("Sx"=>ESx, "Sy"=>ESy, "Sz"=>ESz)
## - end Optional

using Plots
data = minimization_data 

iter_cost, cost = get(minimization_data[:loss])
pl1 = plot(iter_cost, real(cost), yscale=:log10)

iter_mx, mx = get(minimization_data[:obs_1])
pl2 = plot(iter_mx, real(mx))
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
