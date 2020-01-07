export vec_data

abstract type AbstractDerivative end

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
