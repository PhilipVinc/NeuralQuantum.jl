# define macro for statically AD-ing through activation function
#macro frule()

# Define activation functions
@inline ℒ2(x::T) where T = T(2)*cosh(x)
@inline ∂logℒ2(x) = tanh(x)
@inline logℒ2(x::T) where T<:Real =
    (abs(x)<= T(12.0) ? log(cosh(x)) : abs(x) - log(2one(x)))
@inline logℒ2(x::Complex{T}) where T<:Real =
    logℒ2(real(x)) + log(cos(imag(x)) + im * tanh(real(x)) * sin(imag(x)))



@inline ℒ(x) = one(x) + exp(x)
@inline ∂logℒ(x) = one(x)/(one(x)+exp(-x))
#const ∂logℒ = NNlib.σ
#NNlib.σ(x::Complex) = one(x)/(one(x)+exp(-x))
@inline logℒ(x) = log1p(exp(x))#log(one(x) + exp(x))
#const logℒ = NNlib.softplus
#NNlib.softplus(x::Complex) = log1p(exp(x))#log(one(x) + exp(x))

@inline fwd_der(f::typeof(logℒ)) = ∂logℒ
@inline fwd_der(f::typeof(logℒ), x) = ∂logℒ(x)

@inline fwd_der(f::typeof(logℒ2)) = ∂logℒ
@inline fwd_der(f::typeof(logℒ2), x) = ∂logℒ2(x)

@inline fwd_der(f::typeof(identity)) = identity
@inline fwd_der(f::typeof(identity), x::T) where T = one(T)

export af_softplus, af_logcosh
"""
    af_softplus(x) = log( 1 + exp(x) )

Classic softplus (https://en.wikipedia.org/wiki/Sigmoid_function) activation function.
"""
const af_softplus = logℒ

# TODO use NNlib implementation
"""
  logcosh(x)

  Return log(cosh(x)) which is computed in a (somewhat) numerically stable way.
"""
const af_logcosh = logℒ2
