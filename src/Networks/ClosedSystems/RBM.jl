export RBM

struct RBM{VT,MT} <: KetNeuralNetwork
    a::VT
    b::VT
    W::MT
end
@functor RBM

"""
    RBMSplit([T=Complex{STD_REAL_PREC}], N, α, [initW, initb])

Constructs a Restricted Bolzmann Machine to encode a wavefunction,
with weights of type `T` (Defaults to ComplexF32), `N` input neurons,
N⋅α hidden neurons.
This is the Neural Quantum State (NQS) Ansatz.

`N` must match the size of the lattice.

The initial parameters of the neurons are initialized with a rescaled normal
distribution of width 0.01 for the coupling matrix and 0.05 for the local
biases. The default initializers can be overriden by specifying

initW=(dims...)->rescaled_normal(T, 0.01, dims...)
initb=(dims...)->rescaled_normal(T, 0.05, dims...)
inita=(dims...)->rescaled_normal(T, 0.01, dims...)

Refs:
    https://arxiv.org/abs/1606.02318
"""
RBM(in, α, args...) = RBM(ComplexF32, in, α, args...)
RBM(T::Type, in, α,
    initW=(dims...)->rescaled_normal(T, 0.01, dims...),
    initb=(dims...)->rescaled_normal(T, 0.05, dims...),
    inita=(dims...)->rescaled_normal(T, 0.01, dims...)) =
    RBM(inita(in), initb(convert(Int,α*in)),
        initW(convert(Int,α*in), in))

input_type(net::RBM{VT,MT}) where {VT,MT} = real(eltype(VT))
weight_type(net::RBM) = out_type(net)
out_type(net::RBM{VT,MT}) where {VT,MT} = eltype(VT)
input_shape(net::RBM) = length(net.a)
random_input_state(net::RBM{VT,MT}) where {VT,MT} = eltype(VT).([rand(0:1) for i=1:length(net.a)])
is_analytic(net::RBM) = true

(net::RBM)(σ::State) = net(config(σ))
(net::RBM)(σ::AbstractVector) = transpose(net.a)*σ .+ sum(logℒ.(net.b .+ net.W*σ))
(net::RBM)(σ::AbstractMatrix) = transpose(net.a)*σ .+ sum(logℒ.(net.b .+ net.W*σ), dims=1)

function Base.show(io::IO, m::RBM{T}) where T
    print(io, "RBM($(eltype(T)), n=$(length(m.a)), n_hid=$(length(m.b)) => α=$(length(m.b)/length(m.a)))")
end

# Cached version
mutable struct RBMCache{VT} <: NNCache{RBM}
    θ::VT
    logℒθ::VT
    σ::VT
    valid::Bool # = false
end

cache(net::RBM) =
    RBMCache(similar(net.b),
             similar(net.b),
             similar(net.a),
             false)

function (net::RBM)(c::RBMCache, σ_r)
    T=eltype(net.W)
    θ = c.θ
    logℒθ = c.logℒθ

    #θ .= net.b .+ net.W * σ
    σ = copyto!(c.σ, σ_r)
    copyto!(θ, net.b)
    BLAS.gemv!('N', T(1.0), net.W, σ, T(1.0), θ)

    logℒθ .= logℒ.(θ)
    logψ = dot(σ,net.a) + sum(logℒθ)
    return logψ
end

function logψ_and_∇logψ!(∇logψ, net::RBM, c::RBMCache, σ_r)
    T=eltype(net.W)
    θ = c.θ
    logℒθ = c.logℒθ

    #θ .= net.b .+ net.W * σ
    σ = copyto!(c.σ, σ_r)
    copyto!(θ, net.b)
    BLAS.gemv!('N', T(1.0), net.W, σ, T(1.0), θ)

    logℒθ .= logℒ.(θ)
    logψ = dot(σ,net.a) + sum(logℒθ)

    ∇logψ.a .= σ
    ∇logψ.b .= ∂logℒ.(θ)
    ∇logψ.W .= ∇logψ.b  .* transpose(σ)
    return logψ
end
