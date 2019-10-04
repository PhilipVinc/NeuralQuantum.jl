state(prob::AbstractProblem, net, args...) = state(input_type(net), prob, net, args...)
state(T::Type{<:Number}, prob::AbstractProblem, net) = state(T, basis(prob), net)

function state(T::Type{<:Number}, hilb::Basis, net)
    !is_homogeneous(hilb) && error("Could not generate a state.")
    sys_state = _homogeneous_system_state(T, first(hilb.bases),
                                          length(hilb.bases), net)

    state = _network_state(sys_state, net)
    return state
end

function _homogeneous_system_state(T::Type{<:Number}, hilb::Basis, nsites, net::Union{MatrixNet, KetNet})
    loc_size = first(hilb.shape)
    return NAryState(T, loc_size, nsites)
end

_network_state(sys_state::State, net::Union{MatrixNet, PureNet}) =
    DoubleState(sys_state)

_network_state(sys_state::State, net::KetNet) =
    return sys_state
