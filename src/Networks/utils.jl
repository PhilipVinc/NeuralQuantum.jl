# Arrays

glorot_uniform(dims...) = glorot_uniform(Float32, dims...)
glorot_normal(dims...) = glorot_normal(Float32, dims...)
rescaled_normal(scale::Real, dims...) = rescaled_normal(Float32, scale, dims...)

glorot_uniform(T::Type, dims...) = (rand(T, dims...) .- T(0.5)) .* sqrt(T(24)/sum(dims))
glorot_normal(T::Type, dims...) = randn(T, dims...) .* sqrt(T(2)/sum(dims))
rescaled_normal(T::Type, scale::Real, dims::Integer...) = randn(T, dims...) .* T(scale) .* sqrt(T(24)/sum(dims))

# Utils

@inline ℒ2(value::T) where T= T(2.0)*cosh(value)
@inline ∂logℒ2(value::T) where T = tanh(value)
@inline logℒ2(value::T) where T<:Real =
    log(T(2.0))+ (abs(value)<= T(12.0) ? log(cosh(value)) : abs(value) - log(T(2.0)))
@inline logℒ2(value::Complex{T}) where T<:Real =
    log(T(2.0))+ (abs(value)<= T(12.0) ? log(cosh(value)) : abs(value) - log(T(2.0)))



@inline ℒ(value::T) where T= T(1.0) + exp(value)
@inline ∂logℒ(value::T) where T = T(1.0)/(T(1.0)+exp(-value))
@inline logℒ(value::T) where T = log(T(1.0) + exp(value))
