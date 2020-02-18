# Samplers
In v0.2 there are two types of monte carlo samplers implemented:

## Exact Sampler
[`ExactSampler`](@ref), to be used only for relatively small systems, constructs the full probability distribution and samples it directly.
This has exponential memory cost.

The only parameter for this sampler is the number of desired samples and an optional starting seed.


## Metropolis sampler
[`MetropolisSampler`](@ref), can be used on arbitrarily big system. It samples from a Markov Chain where states are updated according to the [Metropolis-Hastings rule](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm). In a nutshell, the Metropolis sampler starts from a state sampled from the system's hilbert space, then proposes a new state according to some rule, and accepts or rejects this new state depending on some probability.

MetropolisSampler takes 3 parameters: `rule`, `chain_length` and `passes`. The second, `chain_length` specifies how many samples should the chain have, while `passes` specifies how many times `rule` must be applied between two returned samples.
The effective chain length will actually be `chain_length * passes`, but only a fraction will be returned, in order to reduce correlation among different samples.

For ergodicity, passes should always be even.

### Metropolis Rules

#### Local
[`LocalRule`](@ref) is a transition rule for Metropolis-Hastings sampling where at every step a random site is switched to another random state.

This rule does not preserve any particular property or symmetry of the hilbert space or of the operator.

#### Exchange
[`ExchangeRule`](@ref) is a transition rule for Metropolis-Hastings sampling where at every step a random couple of sites i,j is selected, and their states switched.
The couples of sites i,j considered are those coupled by the hamiltonian, for example from tight binding terms.

This rule does preserve total magnetization or particle number, and related symmetries of the hamiltonian.

#### Operator
[`OperatorRule`](@ref) is a transition rule for Metropolis-Hastings sampling where at every step the state is changed to any other state to which it is coupled by an Operator (usually the hamiltonian).

This rule preserves symmetries of the operator.


## Reference

### Samplers

```@docs
ExactSampler
MetropolisSampler
```

### Metropolis Rules

```@docs
LocalRule
ExchangeRule
OperatorRule
```
