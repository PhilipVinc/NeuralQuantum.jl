# Basics

## Defining the problem

NeuralQuantum's aim is to compute the steady state of an Open Quantum System or the ground state of an Hamiltonian system. As such, the first step must be defining the quantum system you are interested in.

### Hilbert space

First, you should pick an hilbert space. As of now, only homogeneous Hilbert spaces are supported, [`HomogeneousFock`](@ref) or [`HomogeneousSpin`](@ref) If you wish, for example, to model 5 spin-1/2 particles you can create the Hilbert space as follows:

```julia
julia> using NeuralQuantum

julia> N = 5;
julia> hilb = HomogeneousSpin(N, 1//2)
Hilbert Space with 5 identical spins 1/2 of dimension 2

julia> shape(hilb)
5-element Array{Int64,1}:
 2 2 2 2 2

julia> state(hilb)
5-element Array{Float32,1}:
 -1.0 -1.0 -1.0 -1.0 -1.0
```

!!! Spin vs Fock Space
   You could also model the space as a Fock space with local dimension 2. This choice is formally equivalent, but in this case the states don't have values [-1.0, 1.0] but will take on the values [0.0, 1.0]. This can be useful sometimes when working with some networks. In general, the spin-space works better with `logcosh` activation function, while fock space works better with `softplus` activation function.


### Building an Hamiltonian

To build an hamiltonian, you cannot use simple matrices. Instead, you should use our custom format that behaves similarly to a sparse matrix, but has a few additional tricks that allows us to be efficient in the kind of calculations that Variational Monte Carlo requires.
To build an Hamiltonian, the simplest way is to use the standard pauli-matrices and bosonic creation/destruction operators, and compose them:

```julia
julia> h = 1.0; J=1.0;
julia> H = LocalOperator(hilb)
empty KLocalOperator on space:  HomogeneousSpin(5, 2)

julia> for i=1:N
         global H  -= h * sigmax(hilb, i)
       end
KLocalOperatorSum:
   -sites: Array{Int64,1}[[1], [2], [3], [4], [5]]

julia> for i=1:N
         global H  += J * sigmaz(hilb, i) * sigmaz(hilb, mod(i, N)+1)
       end
KLocalOperatorSum:
  -sites: Array{Int64,1}[[1], [2], [3], [4], [5], [1, 2], [2, 3], [3, 4], [4, 5], [1, 5]]
```

The built-in operators are [`sigmax`](@ref), [`sigmay`](@ref), [`sigmaz`](@ref), [`sigmap`](@ref),  [`sigmam`](@ref) and [`create`](@ref), [`destroy`](@ref). They support all the standard operations (transpose, conjugate, conjugate transpose).

You can also create your custom N-body operators, by specifying an hilbert space, the list of sites upon which it acts, and the matrix in the reduced space of the sites where it acts.
For example, to build by yourself the `sigmaz_1 * sigmaz_2` operator you can do the following:

```julia
julia> sites = [1,2]
julia> mat = diagm(0=>[1.0, -1.0, -1.0, 1.0])
julia> KLocalOperatorRow(hilb, [1,2],  complex.(mat))
KLocalOperator(Complex{Float64})
  Hilb: HomogeneousSpin(5, 2)
  sites: [1, 2]  (size: [2, 2])
 1.0 + 0.0im   0.0 + 0.0im   0.0 + 0.0im  0.0 + 0.0im
 0.0 + 0.0im  -1.0 + 0.0im   0.0 + 0.0im  0.0 + 0.0im
 0.0 + 0.0im   0.0 + 0.0im  -1.0 + 0.0im  0.0 + 0.0im
 0.0 + 0.0im   0.0 + 0.0im   0.0 + 0.0im  1.0 + 0.0im
```

### Choosing a Neural Network State

You should pick a state from those listed in [Networks](@ref). In the following we will pick a simple restricted Boltmann Machine (RBM).

```julia
net  = RBM(Float32, N, 1, af_logcosh)
```

In general, when working with neural networks, the first argument is an optional type for the parameters, the second is the number of sites in the system, and the others depend on the network. In the case of the RBM, the third argument is the density of the hidden layer, and the last argument is the activation function.
In general we have seen that best performance is found by combining logcosh activation with spin-hilbert spaces.

When you create a network, it has all it's weights distributed according to a gaussian with standard deviation 0.005.
You can also reinitialize all weights with a gaussian distribution by using the command [`init_random_pars!`](@ref).
```julia
julia> init_random_pars!(net, sigma=0.01)
```

## Chosing a sampler
The sampler is the algorithm that selects the states in the hilbert space to be
summed over. Options are [`Exact`](@ref), which computes the whole probability distribution and samples exactly, but is very expensive and only works for relatively small (N<10) systems.

In general, you will be using a [`MetropolisSampler`](@ref), which uses a Metropolis-Hastings Markov Chain with a specific transition rule.
Currently only a simple switching rule and an exchange rule are implemented.
```julia
sampler = MetropolisSampler(LocalRule(), 125, N, burn=100)
```

The first argument is the rule, the second argument is the length of each chain, the third argument is the number of times the LocalRule should be applied at every iteration (and should be of the order N). `burn` is an optional keyword argument with the number of unused iterations after the chain is resetted.

## Stochastic Reconfiguration
While it is possible to find the ground state with simple gradient descent, much better efficiency is achieved when the networks have less than 5000 parameters by using Natural Gradient Descent, or Stochastic Reconfiguration, which is somewhat equivalent to a second order newton method.

The stochastic reconfiguration essentially builds a local approximation of the metric, called S-matrix, and solves the equation $\delta x = S^{-1} \nabla C$ where $C$ is the cost function (the energy). This equation can be solved either by inversion or by using an iterative solver, which is much more efficient. Type `?SR` in julia to see it's documentation.
```julia
algo  = SR(ϵ=(0.1), algorithm=sr_cg, precision=1e-3)
```

Notable arguments are `ϵ`, which is the diagonal shift of the S matrix when inverting, the precision of the iterative solver and the algorithm used.

## Solving the problem
All is set. You now only need to construct a `BatchedSampler` (which is a weird name for the object actually effecting the sampling) and optimise the weights

```julia
is = BatchedSampler(net, sampl, H, algo; batch_sz=8)
optimizer = Optimisers.Descent(0.1)

Evalues = Float64[];
Eerr = Float64[];
for i=1:300
    ldata, prec = sample!(is)
    ob = compute_observables(is)

    push!(Evalues, real(ldata.mean))
    push!(Eerr, ldata.error)
    grad = precondition!(prec, algo, i)
    Optimisers.update!(optimizer, net, grad)
end
```

That's it. At every iteration `sample!(is)` will return two elements: the value of the cost function (with it's error) and an object containing data to compute the gradient, which is computed by `precondition!`.

## Computing Observables
If you wish to compute observables, you simply need to compose the operator that represent the observable, and then add it to the `BatchedSampler` by doing
```julia
julia> Sx = LocalOperator(hilb)
julia> for i=1:N
          global Sx += sigmax(hilb, i)/N
       end

julia> add_observable!(is, "Sx", Sx)
```

From now on, if you call `compute_observables(is)` you will obtain a dictionary with all the observables computed.
Observables are computed by using the same Markov Chain used to estimate the energy (cost function) to be minimised for hamiltonian systems. In the case of Open Quantum Systems a different markov chain is used.
