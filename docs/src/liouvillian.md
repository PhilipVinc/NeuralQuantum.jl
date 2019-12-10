# Liouvillian Master Equation

The [Lindblad master equation](https://en.wikipedia.org/wiki/Lindbladian)

```math
\require{physics}
\frac{t \hat{\rho}}{dt} = \mathcal{L}\hat{\rho} = - i \left[\hat{H}, \hat{\rho}\right] + \sum_{i}\left( \hat{L}_i\hat{\rho}\hat{L}^\dagger_i - \frac{1}{2}\left\{\hat{L}^\dagger_i\hat{L}_i,\hat{\rho}\right\}\right)
```

is completely encoded within the Liouvillian ``\mathcal{L}``. To build it, you must first define the Hamiltonian and a vector containing the jump/Loss operators, and then use the function [`liouvillian`](@ref), as shown below.

```julia
julia> N = 7
julia> hilb = HomogeneousSpin(N,1//2)

julia> ops = []
0-element Array{Any,1}

julia> H = LocalOperator(hilb)
empty KLocalOperator on space:  HomogeneousFock(7, 2)

julia> for i=1:N
           global H += g/2.0 * sigmax(hilb, i)
           global H += V/4.0 * sigmaz(hilb, i) * sigmaz(hilb, mod(i, N)+1)

           push!(ops, sigmam(hilb, i))
       end

julia> liouv = liouvillian(H, ops)
KLocalLiouvillian(ComplexF64)
  Hilb: SuperOpSpace(HomogeneousFock(7, 2)))
```

The liouvillian behaves as most other operators, even though you cannot (at the moment) do further algebraic operations on it.

Like any other operator, you can convert it to a Matrix or SparseMatrix by calling the standard Julia conversion methods.
```julia
julia> Matrix(liouv)

julia> sparse(liouv)
```

Those can be used to _convert_ the lindbladian to formats used by other packages, such as QuantumOptics.jl.

## Steady States of Open Quantum Systems

The steady state of an Open Quantum System is computed by minimising the expectation value of

```math
\mathcal{C} = \frac{\text{Tr}\left[\hat{\rho}^\dagger\mathcal{L}^\dagger\mathcal{L}\hat{\rho}\right]}{\text{Tr}\left[\hat{\rho}^\dagger\hat{\rho}\right]}
```

When stochastically sampling this cost function on the hilbert space, it is computed as
```math
\mathcal{C} = \sum_\sigma p(\sigma, \tilde{\sigma}) \left|\left(\mathcal{L}\hat{\rho}\right)(\sigma, \tilde{\sigma})\right|^2
```
which has the 0-variance property. The Markov chain is performed according to the probability ``p(\sigma, \tilde{\sigma}) = |\rho(\sigma, \tilde{\sigma})|^2/Z`` with ``Z=\sum|\rho(\sigma, \tilde{\sigma})|^2``.
