#abstract type State end
#abstract type FiniteBasisState <: State end

# state(hilb, ::MatrixNeuralNet) =
add!(v::FiniteBasisState, i) = set!(v, toint(v)+i)
zero!(v::FiniteBasisState) = set!(v, 0)
@inline config(v) = v
rand!(v::State) = rand!(GLOBAL_RNG, v)

export add!, zero!, NAryState, DoubleState, BinaryState, spacedimension
export nsites, toint, index, index_to_int, flipped, row, col, config
export setat!, set!, set_index!, rand!
