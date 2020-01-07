struct RealDerivative{NT,V} <: AbstractDerivative
    fields::NT
    vectorised_data::V
end

RealDerivative(net::NeuralNetwork) = RealDerivative(out_type(net), net)
function RealDerivative(T::Type{<:Number}, net::NeuralNetwork)
    pars = trainable(net)

    vec    = similar(trainable_first(pars), T, _tlen(pars))
    i, fields = weight_tuple(net, vec)
    return RealDerivative(fields, (vec,))
end

# add :fields to the properties
@inline Base.propertynames(s::RealDerivative) = propertynames(getfield(s, :fields))
@inline Base.getindex(s::RealDerivative, val) = getproperty(s, val)
@inline Base.getproperty(s::RealDerivative, val::Symbol) = _getproperty(s, val)
@inline Base.getproperty(s::RealDerivative, val::Int) = _getproperty(s, val)
@inline function _getproperty(s::RealDerivative, val)
    val===:tuple_all_weights && return vec_data(s)
    return getproperty(getfield(s, :fields), val)
end
@inline vec_data(s::RealDerivative) = getfield(s, :vectorised_data)
@inline fields(s::RealDerivative) = getfield(s, :fields)

weights(der::RealDerivative) = der

Base.isapprox(x::RealDerivative, y::RealDerivative; kwargs...) =
    _isapprox(fields(x), fields(y); kwargs...)

Base.show(io::IO, der::RealDerivative) = begin
    pn = propertynames(der)
    str = "{"
    for fn=pn[1:end-1]
        str *= ":$fn, "
    end
    str *= ":" * string(last(pn))*" }"
    print(io,
    "RealDerivative with fields: ", str)
end
