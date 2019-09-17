# Basics

## Setting up the problem

NeuralQuantum's aim is to compute the steady state of an Open Quantum System. As
such, the first step must be defining the elements that make up the lindbladaian,
namely the Hilbert space, the Hamiltonian and the Loss operators.

While it is possible to specify an arbitrary quantum system, the easiest way is
to use one of the alredy-implemented systems.
Alternatively, it is possible to define an Hamiltonian and jump operators by
using [QuantumOptics.jl](http://github.com/bastikr/QuantumOptics.jl).

```
using NeuralQuantum

Nspins = 5 # The number of spins in the system

# Create the lattice as [Nx, Ny, Nz]
lattice = SquareLattice([Nspins],PBC=true)

# Compute the liouvillian.
liouv = quantum_ising_system(lattice, V=0.2, g=1.0, gamma=0.1, PBC=true)
```

Next, we need to define the quantity that we wish to minimize variationally to
find the steady state. This will be ``\langle\rho|\mathcal{L}^\dagger\mathcal{L
}|\rho\rangle``, sampled by computing ``\mathcal{L}|\rho``.
I call this quantity the *problem*.
```
prob = SteadyStateProblem(lind);
```

## Choosing the Ansatz
The next step consists in creating the network-based ansatz for the density
matrix.
In this example we will use a 64-bit floating point precision translational
invariant Neural Density Matrix, with ``N_\text{spins}`` spins in the visible layer,
1 features in the hidden layer (~ ``3N_\text{spins}`` spins) and 2 features in the
ancilla.
To increase the expressive power of the network, one may increase freely the
number of features.
For a complete list of all possible ansatzes refer to section [Networks](@ref).

```
net_initial = NDM(Nspins, 1, 2)
```

When you create a network, it has all it's weights distributed according to a gaussian
with standard deviation 0.005.

## Solving for the steady state
Having specified the quantity to minimize and the ansatz, we only need to choose
the sampler and the optimization scheme.

The sampler is the algorithm that selects the states in the hilbert space to be
summed over. Options are `Exact`, which performs an exact sum over all possible
vectors in the hilbert space, `ExactDistrib`, which samples a certain number of
elements according to the exact distribution, and `Metropolis`, which performs a
Markov Chain according to the Metropolis rule. For this example we will chose an
exact sum over all the hilbert space.

The Optimizer is instead the algorithm that, given the gradient, updates the
variational weights of the ansatz. In this example we will use a simple gradient
descent scheme with step size `lr=0.01`. Many more optimizers are described in
Sec. [Optimizers](@ref).

```
# Initialize an Exact Sum Sampler
sampler = Exact()

# Initialize a GD optimizer with learning rate (step size) 0.01
optim = GD(lr=0.1)

# Optimize the weights of the neural network for 100 iterations
sol = solve(net_initial, prob, sampling_alg=sampler, optimizer=optim, max_iter=100)
```

After running the `solve` command you should see in the temrinal a list of the loss function
updating over time, and how much it varies across iterations, such as this:
```
  iter -> E= <rho| LdagL |rho>/<rho|rho>     → ΔE = variation since last iter
     3 -> E=(+7.199784e-01 +im+8.558472e-18) → ΔE = -1.20e-01
     4 -> E=(+6.173014e-01 +im-5.918733e-18) → ΔE = -1.03e-01
     5 -> E=(+4.952037e-01 +im-5.480910e-18) → ΔE = -1.22e-01
```
If the optimization goes well, `ΔE` should almost always be negative as to converge
towards the minimum.

## The solution object
The `solve` command returns a structure holding several important data:
   - `sol.net_initial` stores the initial configuration of the network
   - `sol.net_end` stores the configuration of the network at the end of the optimization
   - `sol.data` stores the data of the observables and other quantities of interest

To access, for example, the value of ``\langle\rho|\mathcal{L}^\dagger\mathcal{L
}|\rho\rangle`` along the optimization, one can simply do
```
sol[:Energy].iterations
sol[:Energy].values
```
where `iterations` is a vector containing the information at which iterations
the corresponding `value` was logged.


## Logging observables during the evolution
It can be usefull to store also some other observables during the evolution. To
do so, one can pass additional keyword-arguments to the solve command:

   - `observables`: A list of tuples, containing the observables and a symbol to name it, such as
   `[(:obs1, obs1), (:obs2, obs2)]`
   - `log_skip_steps`, an integer specifing every how many iterations the observable should be computed
   - `log_weights` saves the weights.
   - `log_fidelity`, logs the fidelity of the state w.r.t the target state. WARNING:
   computing the fidelity is a very computationally demanding task, and as such you
   should not usually use this feature

For example, to store the observables ``m_x``, ``m_y`` and ``m_z``  every 2
optimization step, we can do the following:
```
# Compute the operator for the average magnetization
mx, my, mz = magnetization([:x, :y, :z], sys)./Nspins

# Create the list of observables and symbols
obs = [(:mx, mx), (:my, my), (:mz, mz)];

# Create the logger
sol = solve(net_initial, prob, max_iter=100, observables=obs, save_at=2)
```

The `magnetization([::Symbols], ::System)` function returns the total
magnetization operator along the specified axis for the given system. For more
information on this function refer to [Systems](@ref).

#### Using the Logger
Once you have solution, you can plot it with:
```
using Plots

pl_E = plot(sol[:Energy])
pl_x = plot(sol[:mx])
pl_y = plot(sol[:my])
pl_z = plot(sol[:mz])
pl(pl_E, pl_x, pl_y, pl_z)
```

Quantities inside the logger are stored as :
```
sol[:OBSERVABLE].iterations ->
sol[:OBSERVABLE].values     ->
```

You can also concatenate several different solutions. This will return a MVHistory
object holding the evolution of all quantities of interest and observables, but it
won't hold anymore information on the weights.

## Summary
```

```
