export RBM

struct RBM{T} <: KetNeuralNetwork
    a::Vector{T}
    b::Vector{T}
    W::Matrix{T}
end

RBM(in, α, args...) = RBM(ComplexF32, in, α, args...)
RBM(T::Type, in, α,
    initW=(dims...)->rescaled_normal(T, 0.01, dims...),
    initb=(dims...)->rescaled_normal(T, 0.05, dims...),
    inita=(dims...)->rescaled_normal(T, 0.01, dims...)) =
    RBM(inita(in), initb(convert(Int,α*in)),
        initW(convert(Int,α*in), in))

input_type(net::RBM{T}) where T = real(T)
weight_type(net::RBM) = out_type(net)
out_type(net::RBM{T}) where T = T
input_shape(net::RBM) = length(net.a)
random_input_state(net::RBM{T}) where T = T.([rand(0:1) for i=1:length(net.a)])
is_analytic(net::RBM) = true

(net::RBM)(σ::State) = net(config(σ))
(net::RBM{T})(σ) where T = transpose(σ)*net.a .+ sum(logℒ.(net.b .+ net.W*σ))

function Base.show(io::IO, m::RBM{T}) where T
    print(io, "RBM{$T}, n=$(length(m.a)), n_hid=$(length(m.b)) => α=$(length(m.b)/length(m.a)))")
end

# Cached version
mutable struct RBMCache{T} <: NNCache{RBM{T}}
    θ::Vector{T}
    logℒθ::Vector{T}
    σ::Vector{T}
    valid::Bool # = false
end

cache(net::RBM{T}) where T =
    RBMCache(Vector{T}(undef,length(net.b)),
             Vector{T}(undef,length(net.b)),
             Vector{T}(undef,length(net.a)),
                  false)

function (net::RBM{T})(c::RBMCache, σ) where T
    θ = c.θ
    logℒθ = c.logℒθ

    #θ .= net.b .+ net.W * σ
    copy!(c.σ, σ)
    copy!(θ, net.b)
    BLAS.gemv!('N', T(1.0), net.W, c.σ, T(1.0), θ)

    logℒθ .= logℒ.(θ)
    logψ = dot(σ,net.a) + sum(logℒθ)
    return logψ
end

function logψ_and_∇logψ(∇logψ, c::RBMCache, net::RBM{T}, σ) where T
    θ = c.θ
    logℒθ = c.logℒθ

    #θ .= net.b .+ net.W * σ
    copy!(c.σ, σ)
    copy!(θ, net.b)
    BLAS.gemv!('N', T(1.0), net.W, c.σ, T(1.0), θ)

    logℒθ .= logℒ.(θ)
    logψ = dot(σ,net.a) + sum(logℒθ)

    ∇logψ.a .= σ
    ∇logψ.b .= ∂logℒ.(θ)
    ∇logψ.W .= ∇logψ.b  .* transpose(σ)
    return logψ, ∇logψ
end
