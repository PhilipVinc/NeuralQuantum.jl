
# NeuralQuantum.jl : Neural Network states for Quantum Systems

**NeuralQuantum.jl** is a numerical framework written in [Julia](http://julialang.org) to investigate Neural-Network representations of pure and mixed quantum states, and to find the Steady-State of such (Open) Quantum Systems through MonteCarlo procedures.
The package can also compute the ground state of a many-body hamiltonian.

!!! note
    This code is currently heavily in the making. v0.2 should mark a somewhat more stable interface, but it's very different from older versions.
    If you find this code interesting, I'd be glad if you could let me know and give me some feedback.

## Installation

Download [Julia 1.3](https://julialang.org) or a more recent version (we do not support older versions of Julia). To install `NeuralQuantum`, run in a Julia prompt the following command.
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


## Table Of Contents
```@contents
```



## Index
```@index
```
