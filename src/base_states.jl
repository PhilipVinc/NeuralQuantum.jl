#abstract type State end
#abstract type FiniteBasisState <: State end

# state(hilb, ::MatrixNeuralNet) =
add!(v::FiniteBasisState, i) = set!(v, toint(v)+i)
zero!(v::FiniteBasisState) = set!(v, 0)
@inline config(v) = v
rand!(v::State) = rand!(GLOBAL_RNG, v)
flipat!(v::State, i) = flipat!(GLOBAL_RNG, v, i)

export NAryState, DoubleState, BinaryState
export local_dimension, spacedimension
export nsites, toint, index, index_to_int, flipped, row, col, config
export add!, zero!
export setat!, set!, set_index!, rand!
