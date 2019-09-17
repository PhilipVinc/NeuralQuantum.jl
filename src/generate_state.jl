state(prob::AbstractProblem, net, args...) = state(input_type(net), prob, net, args...)
state(T::Type{<:Number}, prob::AbstractProblem, net) = state(T, basis(prob), net)
#state(T::Type{<:Number}, prob::LRhoKLocalOpProblem, net) = state_lut(T, basis(prob), net)
state_lut(prob::AbstractProblem, net, args...) = state_lut(input_type(net), prob, net, args...)
state_lut(T::Type{<:Number}, prob::AbstractProblem, net) = state_lut(T, basis(prob), net)

function state(T::Type{<:Number}, hilb::Basis, net)
    !is_homogeneous(hilb) && error("Could not generate a state.")
    sys_state = _homogeneous_system_state(T, first(hilb.bases),
                                          length(hilb.bases), net)

    state = _network_state(sys_state, net)
    return state
end

function state_lut(T::Type{<:Number}, hilb::Basis, net)
    !is_homogeneous(hilb) && error("Could not generate a state.")
    sys_state = _homogeneous_system_state(T, first(hilb.bases),
                                          length(hilb.bases), net)

    state = _lut_state(sys_state, net)
    return state
end

function _lut_state(T::Type{<:Number}, hilb::Basis, net::MatrixNet)
    bare_state = state(T, basis(prob), net)
    state = DoubleState(ModifiedState(row(bare_state)))
    lut   = lookup(net)

    isnothing(lut) && return state
    return LUState(state, lut)
end

function _homogeneous_system_state(T::Type{<:Number}, hilb::Basis, nsites, net::Union{MatrixNet, KetNet})
    loc_size = first(hilb.shape)
    return NAryState(T, loc_size, nsites)
end

_network_state(sys_state::State, net::Union{MatrixNet, PureNet}) =
    DoubleState(sys_state)

_network_state(sys_state::State, net::KetNet) =
    return sys_state

function _lut_state(sys_state::State, net::Union{MatrixNet, PureNet})
    state = DoubleState(ModifiedState(sys_state))
    lut   = lookup(net)

    isnothing(lut) && return state
    return LUState(state, lut)
end
