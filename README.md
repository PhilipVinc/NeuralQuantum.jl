# NeuralQuantum
[![Build Status](https://travis-ci.org/PhilipVinc/NeuralQuantum.jl.svg?branch=master)](https://travis-ci.org/PhilipVinc/NeuralQuantum.jl) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://philipvinc.github.io/NeuralQuantum.jl/dev) [![DOI](https://zenodo.org/badge/186389926.svg)](https://zenodo.org/badge/latestdoi/186389926)

**NeuralQuantum** is a numerical framework written in [Julia] to investigate
Neural-Network representations of mixed quantum states and to find the Steady-
State of dissipative Quantum Systems with variational Montecarlo schemes.
It can also compute the ground state of hermitian hamiltonians.

This code has been developed while working on [Variational neural network ansatz for steady states in open quantum systems](https://arxiv.org/abs/1902.10104), by [F. Vicentini] et al. [Phys Rev Lett 122, 250503 (2019)](https://link.aps.org/doi/10.1103/PhysRevLett.122.250503).

## Installation
To Install `NeuralQuantum.jl`, run the following commands to install it's dependcy. Please note that we require julia >= 1.3, and relatively recent versions of several packages.
```
using Pkg
pkg"add https://github.com/PhilipVinc/NeuralQuantum.jl"
```

Alternatively you may activate the project included in the manifest that comes with NeuralQuantum.


## Examples

Check the folder `Examples/` for a few interesting examples.

[Julia]: http://julialang.org
[F. Vicentini]: mailto:filippo.vicentini@univ-paris-diderot.fr
[Filippo Vicentini]: mailto:filippo.vicentini@univ-paris-diderot.fr
