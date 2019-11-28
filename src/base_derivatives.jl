export vec_data

abstract type AbstractDerivative end

struct RealDerivative{NT,V} <: AbstractDerivative
    fields::NT
    vectorised_data::V
end

@inline Base.propertynames(s::RealDerivative) = propertynames(getfield(s, :fields))
@inline Base.getindex(s::RealDerivative, val) =
    getproperty(s, val)
@inline Base.getproperty(s::RealDerivative, val::Symbol) = _getproperty(s, val)
@inline Base.getproperty(s::RealDerivative, val::Int) = _getproperty(s, val)
@inline function _getproperty(s::RealDerivative, val)
    val===:tuple_all_weights && return vec_data(s)
    return getproperty(getfield(s, :fields), val)
end
@inline vec_data(s::RealDerivative) = getfield(s, :vectorised_data)
@inline fields(s::RealDerivative) = getfield(s, :fields)

weights(der::RealDerivative) = der

function RealDerivative(net::NeuralNetwork)
    pars = trainable(net)

    vec    = similar(trainable_first(pars), out_type(net), _tlen(pars))
    i, fields = weight_tuple(net, vec)
    return RealDerivative(fields, [vec])
end


struct WirtingerDerivative{R,V} <: AbstractDerivative
    r_derivatives::R
    c_derivatives::R
    vectorised_data::V
end

vec_data(s::WirtingerDerivative)  = s.vectorised_data
Base.real(s::WirtingerDerivative) = s.r_derivatives
Base.imag(s::WirtingerDerivative) = s.c_derivatives

@inline Base.getproperty(s::WirtingerDerivative, val::Symbol) = _getproperty(s, val)
@inline Base.getproperty(s::WirtingerDerivative, val::Int) = _getproperty(s, val)
@inline function _getproperty(s::WirtingerDerivative, val)
    val===:tuple_all_weights && return vec_data(s)
    return getfield(s, val)
end

function WirtingerDerivative(net::NeuralNetwork)
    pars = trainable(net)

    vec         = similar(trainable_first(net), out_type(net), _tlen(pars)*2)
    i, fields_r = weight_tuple(net, vec)
    i, fields_c = weight_tuple(net, vec, i+1)
    return WirtingerDerivative(fields_r, fields_c, [vec])
end

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

Base.show(io::IO, ::MIME"text/plain", der::RealDerivative) = begin
    pn = propertynames(der)
    str = "{"
    for fn=pn[1:end-1]
        str *= ":$fn, "
    end
    str *= ":" * string(last(pn))*" }"
    print(io,
    "RealDerivative with fields: ", str)
end

Base.show(io::IO, der::WirtingerDerivative) = begin
    pn = propertynames(real(der))
    str = "{"
    for fn=pn[1:end-1]
        str *= ":$fn, "
    end
    str *= ":" * string(last(pn))*" }"
    print(io,
    "WirtingerDerivative with fields: ", str)
end

Base.show(io::IO, ::MIME"text/plain", der::WirtingerDerivative) = begin
    pn = propertynames(real(der))
    str = "{"
    for fn=pn[1:end-1]
        str *= ":$fn, "
    end
    str *= ":" * string(last(pn))*" }"
    print(io,
    "WirtingerDerivative with fields: ", str)
end

Base.isapprox(x::RealDerivative, y::RealDerivative; kwargs...) =
    _isapprox(fields(x), fields(y); kwargs...)

function _isapprox(x, y; kwargs...)
    kx = propertynames(x)
    ky = propertynames(y)
    all(kx .== ky) || return false
    for f=kx
        _isapprox(getproperty(x, f), getproperty(y, f); kwargs...) || return false
    end
    return true
end

_isapprox(x::AbstractArray, y::AbstractArray; kwargs...) =
    isapprox(x, y; kwargs...)
