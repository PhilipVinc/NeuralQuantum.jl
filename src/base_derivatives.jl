export vec_data

abstract type AbstractDerivative end

struct RealDerivative{NT,V} <: AbstractDerivative
    fields::NT
    vectorised_data::V
end

@inline Base.propertynames(s::RealDerivative) = propertynames(getfield(s, :fields))
@inline Base.getindex(s::RealDerivative, val::Symbol) =
    getproperty(s, val)
@inline function Base.getproperty(s::RealDerivative, val::Symbol)
    val===:tuple_all_weights && return vec_data(s)
    return getproperty(getfield(s, :fields), val)
end
@inline vec_data(s::RealDerivative) = getfield(s, :vectorised_data)

struct WirtingerDerivative{R,V} <: AbstractDerivative
    r_derivatives::R
    c_derivatives::R
    vectorised_data::V
end

vec_data(s::WirtingerDerivative)  = s.vectorised_data
Base.real(s::WirtingerDerivative) = s.r_derivatives
Base.imag(s::WirtingerDerivative) = s.c_derivatives

@inline function Base.getproperty(s::WirtingerDerivative, val::Symbol)
    val===:tuple_all_weights && return vec_data(s)
    return getfield(s, val)
end

function RealDerivative(net::NeuralNetwork)
    vec    = Vector{out_type(net)}()
    i, fields = weight_tuple(net, fieldnames(typeof(net)), vec)
    return RealDerivative(fields, [vec])
end

function WirtingerDerivative(net::NeuralNetwork)
    vec         = Vector{out_type(net)}()
    i, fields_r = weight_tuple(net, fieldnames(typeof(net)), vec)
    i, fields_c = weight_tuple(net, fieldnames(typeof(net)), vec, i+1)
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
