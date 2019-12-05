
# NeuralQuantum.jl

*A Neural-Network steady-state solver*

**NeuralQuantum.jl** is a numerical framework written in [Julia](http://julialang.org)
to investigate Neural-Network representations of pure and mixed quantum states, and to find the Steady-State of such (Open) Quantum Systems through MonteCarlo procedures.
The package can also compute the ground state of a many-body hamiltonian.

!!! note
    This code is currently heavily in the making. v0.2 should mark a somewhat more stable interface, but it's very different from older versions.
    If you find this code interesting, I'd be glad if you could let me know and give me some feedback.

## Installation

To install `NeuralQuantum`, run in a Julia prompt the following command.
```
] add https://github.com/PhilipVinc/NeuralQuantum.jl
```

## Basic Usage
```@meta
CurrentModule = NeuralQuantum
```

When using NeuralQuantum, to determine the Ground State or Steady State of a many-body problem, one needs to perform the following choices:
 - Chose a Neural-Network based ansatz to approximate the quantum state (see Sec. [Networks](@ref));
 - Chose whever you want to perform a standard (stochastic) gradient descent, or if you want to use Natural Gradient Descent (also known as Stochastic Reconfiguration) (see Sec. [Algorithms](@ref));
 - Chose the optimizer to perform the optimization, such as steepest gradient, accelerated gradient or others (see Sec. [Optimizers](@ref));

Here you can find a very short, commented example. For a more in-depth walkthrough of `NeuralQuantum.jl` please refer to Sec. [Basics](@ref).

```
# Load dependencies
using NeuralQuantum, Random, Plots

# Parameters of the transverse field ising model
N = 20
h = 1.0
J = 1.0

# Constructs the Hilbert space for N 1//2 spins.
hilb = HomogeneousSpin(N, 1//2)

# Builds the hamiltonian
H = LocalOperator(hilb)
for i=1:N
    global H  -= h * sigmax(hilb, i)
    global H  += J * sigmaz(hilb, i) * sigmaz(hilb, mod(i, N)+1)
end

# Constructs the Neural Network
net  = RBM(Float32, N, 1)
# Initializes the parameters to a random gaussian of variance 0.01
init_random_pars!(net, sigma=0.01)

# Chose a Metropolis-Hastings sampler with Local Transition rule.
# Each chain has length 125 and every step involves applying the transition rule N times.
# Trash (burn) the first 100 elements of the chain.
sampl = MetropolisSampler(LocalRule(), 125, N, burn=100)

# Use Stochastic Reconfiguration with 0.1 diagonal offset and use a cholesky solver for inverting the matrix.
algo  = SR(ϵ=(0.1), algorithm=sr_cholesky)

# Run 8 chains in parallel
is = BatchedSampler(net, sampl, H, algo; batch_sz=8)

# SGD optimizer with step size 0.1
optimizer = Optimisers.Descent(0.1)

Evalues = Float64[];
Eerr = Float64[];
for i=1:300
    ldata, prec = sample!(is)
    ob = compute_observables(is)

    println(ldata)

    push!(Evalues, real(ldata.mean))
    push!(Eerr, ldata.error)
    grad = precondition!(prec, algo, i)
    Optimisers.update!(optimizer, net, grad)
end

plot(Evalues, yerr=Eerr)

# N=20 (thanks netket)
exact = -1.274549484318e00 * 20
hline!([exact, exact])
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
