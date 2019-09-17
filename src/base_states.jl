#abstract type State end
#abstract type FiniteBasisState <: State end

add!(v::FiniteBasisState, i) = set!(v, toint(v)+i)
zero!(v::FiniteBasisState) = set!(v, 0)
@inline config(v) = v
rand!(v::State) = rand!(GLOBAL_RNG, v)
flipat!(v::State, i) = flipat!(GLOBAL_RNG, v, i)
flipat_fast!(v::State, i) = flipat_fast!(GLOBAL_RNG, v, i)

init_lut!(s::State, net::NeuralNetwork) = nothing
update_lookup!(s::State, net::NeuralNetwork) = nothing

export NAryState, DoubleState, BinaryState
export local_dimension, spacedimension
export nsites, toint, index, index_to_int, flipped, row, col, config
export add!, zero!
export setat!, set!, set_index!, rand!
