#abstract type State end
#abstract type FiniteBasisState <: State end

add!(v::FiniteBasisState, i) = set!(v, toint(v)+i)
zero!(v::FiniteBasisState) = set!(v, 0)
@inline config(v) = v
Random.rand!(v::State) = rand!(GLOBAL_RNG, v)
flipat!(v::State, i) = flipat!(GLOBAL_RNG, v, i)
flipat_fast!(v::State, i) = flipat_fast!(GLOBAL_RNG, v, i)

export NAryState, DoubleState, BinaryState
export local_dimension, spacedimension
export nsites, toint, index, index_to_int, flipped, row, col, config
export add!, zero!, apply!, apply
export setat!, set!, set_index!, rand!

apply(σ::State, cngs) = apply!(deepcopy(σ), cngs)

"""
    apply!(state::State, changes)

Applies the changes `changes` to the `state`.

If `state isa DoubleState` then single-value changes
are applied to the columns of the state (in order to
compute matrix-operator products). Otherwise it should
be a tuple with changes of row and columns.

If changes is nothing, does nothing.
"""
apply!(σ::State, cngs::Nothing) = σ
