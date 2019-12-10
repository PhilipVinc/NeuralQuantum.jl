# Reference

This is the reference to the public interface of NeuralQuantum.

## Hilbert spaces

Methods relative to Hilbert spaces and their states

### General methods

```@docs
nsites
NeuralQuantum.shape
NeuralQuantum.local_dim
NeuralQuantum.spacedimension
NeuralQuantum.indexable
NeuralQuantum.is_homogeneous
```

### Constructors

```@docs
HomogeneousFock
HomogeneousSpin
```

### Working with states

```@docs
index
state
states
Random.rand!
Random.rand
apply!
NeuralQuantum.flipat!
NeuralQuantum.setat!
NeuralQuantum.local_index
NeuralQuantum.unsafe_get_el
```

## Operators

```@docs
sigmax
sigmay
sigmaz
sigmap
sigmam
create
destroy
KLocalOperator
NeuralQuantum.KLocalOperatorRow
NeuralQuantum.liouvillian
```

## Samplers

```@docs
ExactSampler
MetropolisSampler
```

### Transition Rules

```@docs
LocalRule
ExchangeRule
```
