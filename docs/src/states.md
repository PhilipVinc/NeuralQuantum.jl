# State abstract type

`State` is the base abstract type which must be subtyped for the elements that
span an hilbert space (or a Matrix space), and which also correspond to a
configuration of the visible layer of the neural network.

## Defining a new State concrete struct

It is possible that one wants to define a new type of state, which spans the
density matrix space in a different way and gives a correspondence between visible
layer configurations and elements in the density matrix in a novel way. To do so,
one must take care of defining the following ingredients:
    - The mutable struct holding the state itself;
    - Accessor methods that can be used by the sampler and the problem to compute
    observables;
    - Methods for modyfing the state;
    - (optionally) constructors.


a State derivating from DualValuedState must define the following:

```
mutable struct ExampleState <: DualValuedState
	...
	...
end
```

### Constructors
No default constructor must be created by itself, because it will
be called from specific code of every NeuralNetwork. Though, the standard interface would require to define
```
ExampleState{...}({number params}, {Ints })
```
where number params are all parameters listing the length of the various items in the ExampleState, and {Ints} are the ints that initialize them.

### Accessors
To define a new state, all those accessors must be defined:

 - `spacedimension(state::ExampleState)` returning the size of the space where this state lives. For example, for N spins this will be 2^N
 - `half_space_dimension(state::ExampleState)` returns the size of the physical hilbert space where the density matrix is defined. Usually 2^(N/2)
 - `toint(state::ExampleState)` returning the state expressed as an integer
 - `nsites(state::ExampleState)` returns the number of sites where this state is defined.
 - `vectorindex(state::ExampleState)` ????
 - `neurontype(state::ExampleState)` returning the type of the neuron for this state
 - `neurontype(::Type{ExampleState})` returning the type of the neuron for this state.

### Operations
 - `flipat!(state, i::Integer)`
 - `set!(state, i::Integer)`
 - `add!(State, i::Integer)`
 - `setzero!(state)
