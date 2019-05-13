state(prob, net, args...) = state(input_type(net), prob, net, args...)
state(T::Type{<:Number}, prob, net) = state(T, basis(prob), net)
function state(T::Type{<:Number}, hilb::Basis, net)
    is_homogeneous(hilb) && return _homogeneous_state(T, first(hilb.bases), length(hilb.bases), net)

    error("Could not generate a state.")
end

function _homogeneous_state(T::Type{<:Number}, hilb::Basis, nsites, net::MatrixNet)
    loc_size = first(hilb.shape)
    v = NAryState(T, loc_size, nsites)
    return DoubleState(v)
end

function _homogeneous_state(T::Type{<:Number}, hilb::Basis, nsites, net::PureNet)
    loc_size = first(hilb.shape)
    v = NAryState(T, loc_size, nsites)
    return DoubleState(v)
end

function _homogeneous_state(T::Type{<:Number}, hilb::Basis, nsites, net::KetNet)
    loc_size = first(hilb.shape)
    return NAryState(T, loc_size, nsites)
end
