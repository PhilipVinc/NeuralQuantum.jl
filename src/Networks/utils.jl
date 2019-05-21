# Arrays

glorot_uniform(dims...) = glorot_uniform(Float32, dims...)
glorot_normal(dims...) = glorot_normal(Float32, dims...)
rescaled_normal(scale::Real, dims...) = rescaled_normal(Float32, scale, dims...)

glorot_uniform(T::Type, dims...) = (rand(T, dims...) .- T(0.5)) .* sqrt(T(24)/sum(dims))
glorot_normal(T::Type, dims...) = randn(T, dims...) .* sqrt(T(2)/sum(dims))
rescaled_normal(T::Type, scale::Real, dims::Integer...) = randn(T, dims...) .* T(scale) .* sqrt(T(24)/sum(dims))

# Utils

@inline ℒ2(x) = 2one(x)*cosh(x)
@inline ∂logℒ2(x) = tanh(x)
@inline logℒ2(x::T) where T<:Real =
    log(2one(x))+ (abs(x)<= T(12.0) ? log(cosh(x)) : abs(x) - log(2one(X)))
@inline logℒ2(value::Complex{T}) where T<:Real =
    log(T(2.0))+ (abs(x)<= T(12.0) ? log(cosh(x)) : abs(x) - log(2one(x)))



@inline ℒ(x) = one(x) + exp(x)
@inline ∂logℒ(x) = one(x)/(one(x)+exp(-x))
@inline logℒ(x) = log1p(exp(x))#log(one(x) + exp(x))
