# Problems

A `Problem` is the structure representing the problem that must be solved, such
as the minimization of the cost function to determine the steady state. Currently
only problems representing the computation of the steady state by minimization
of the cost function ``\mathcal{L}^\dagger\mathcal{L}`` are supported.

## Minimization of ``\mathcal{C} = \langle\langle\mathcal{L}^\dagger\mathcal{L}\rangle\rangle``

This cost function can be computed in two ways:
```math
\mathcal{C} = \sum_\sigma p(\sigma) \langle\langle\sigma |\mathcal{L}^\dagger\mathcal{L}\rho\rangle\rangle
```
or
```math
\mathcal{C} = \sum_\sigma p(\sigma) |\langle\langle\sigma |\mathcal{L}\rho\rangle\rangle||^2
```

The second version leads to smaller variance of sampled variables and also is
faster to evaluate because it holds only ``\mathcal{L}`` instead of ``\mathcal{L}^\dagger\mathcal{L}``, as such I reccomend to use this one.


```@docs
LdagL_Lmat_prob

LdagL_spmat_prob
```
