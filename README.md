# NeuralQuantum
[![Build Status](https://travis-ci.org/PhilipVinc/NeuralQuantum.jl.svg?branch=master)](https://travis-ci.org/PhilipVinc/NeuralQuantum.jl) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://philipvinc.github.io/NeuralQuantum.jl/dev) [![DOI](https://zenodo.org/badge/186389926.svg)](https://zenodo.org/badge/latestdoi/186389926)

**NeuralQuantum** is a numerical framework written in [Julia] to investigate
Neural-Network representations of mixed quantum states and to find the Steady-
State of dissipative Quantum Systems with variational Montecarlo schemes.
It can also compute the ground state of hermitian hamiltonians.

This code has been developed while working on [Variational neural network ansatz for steady states in open quantum systems](https://arxiv.org/abs/1902.10104), by [F. Vicentini] et al. [Phys Rev Lett 122, 250503 (2019)](https://link.aps.org/doi/10.1103/PhysRevLett.122.250503).

## Installation
To Install `NeuralQuantum.jl`, run the following commands to install it's dependcy.
```
using Pkg
pkg"add https://github.com/PhilipVinc/NeuralQuantum.jl"
```

Alternatively you may activate the project included in the manifest that comes with NeuralQuantum.


## Example
The basic idea of the package is the following: you create an hamiltonian/lindbladian made of LocalOperators (which is a custom format, similar to Sparse Matrices, that allows for more efficient operations).
You should create in the same way the operators encoding the observables that you want to compute.


Then, you will pick a network, a sampler, and create an iterative sampler to sample the network.

By default, if you don't provide the precision `Float32` is used.

You must write the training loop by yourself. Check the documentation and the examples in the folder `examples/` to better understand how to do this.

*IMPORTANT:* If you want to use multithreaded samplers (identified by a `MT` at the beginning of their name), you will launch one markov chain per julia thread. As such, you will get much better performance if you set `JULIA_NUM_THREADS` environment variable to the number of physical cores in your computer before launching julia.

```
using NeuralQuantum, QuantumOpticsBase, ProgressMeter
using NeuralQuantum: unsafe_get_el

N = 7
g = 0.4
V = 2.0

hilb = HomogeneousSpin(N,1//2)
hilb = HomogeneousHilbert(N,2)

ops = []
H = LocalOperator(hilb)

Sx = LocalOperator(hilb)
Sy = LocalOperator(hilb)
Sz = LocalOperator(hilb)

for i=1:N
    global H += g/2.0 * sigmax(hilb, i)
    global H += V/4.0 * sigmaz(hilb, i) * sigmaz(hilb, mod(i, N)+1)

    global Sx += sigmax(hilb, i)/N
    global Sy += sigmay(hilb, i)/N
    global Sz += sigmaz(hilb, i)/N

    push!(ops, sigmam(hilb, i))
end

liouv = liouvillian(H, ops)

sampl = MetropolisSampler(LocalRule(), 125, N, burn=100)
#sampl = ExactSampler(5000)
algo  = SR(ϵ=(0.001), algorithm=sr_cholesky)
#algo  = Gradient()

net  = NDM(Float64, N, 1, 1, NeuralQuantum.logℒ)
is = BatchedSampler(net, sampl, liouv, algo; batch_sz=16)
add_observable(is, "Sx", Sx)
add_observable(is, "Sy", Sy)
add_observable(is, "Sz", Sz)

optimizer = Optimisers.Descent(0.01)

Evalues = Float64[];
Eerr = Float64[];
for i=1:200
    ldata, prec = sample!(is)
    ob = compute_observables(is)

    println("$i - $ldata")

    push!(Evalues, ldata.mean)
    push!(Eerr, ldata.error)
    grad = precondition!(prec, algo, i)
    Optimisers.update!(optimizer, net, grad)
end

plot(Evalues, yerr=Eerr, yscale=:log10)
```

[Julia]: http://julialang.org
[F. Vicentini]: mailto:filippo.vicentini@univ-paris-diderot.fr
[Filippo Vicentini]: mailto:filippo.vicentini@univ-paris-diderot.fr
