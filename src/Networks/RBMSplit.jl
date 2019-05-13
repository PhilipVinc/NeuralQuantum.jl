export RBMSplit

struct RBMSplit{T} <: MatrixNeuralNetwork
    ar::Vector{T}
    ac::Vector{T}
    b::Vector{T}
    Wr::Matrix{T}
    Wc::Matrix{T}
end

RBMSplit(in::Int, α::Number, args...) = RBMSplit(ComplexF32, in, α, args...)
RBMSplit(T::Type, in, α,
         initW=(dims...)->rescaled_normal(T, 0.01, dims...),
         initb=(dims...)->rescaled_normal(T, 0.05, dims...),
         inita=(dims...)->rescaled_normal(T, 0.01, dims...)) =
    RBMSplit(inita(in), inita(in),
             initb(convert(Int, α*in)),
             initW(convert(Int, α*in), in), initW(convert(Int, α*in), in))

input_type(net::RBMSplit{T}) where T = real(T)
weight_type(net::RBMSplit) = out_type(net)
out_type(net::RBMSplit{T}) where T = Complex{real(T)}
input_shape(net::RBMSplit) = (length(net.ar), length(net.ac))
random_input_state(net::RBMSplit{T}) where T =
    (T.([rand(0:1) for i=1:length(net.ar)]), T.([rand(0:1) for i=1:length(net.ar)]))
is_analytic(net::RBMSplit) = true


(net::RBMSplit)(σ::State) = net(config(σ)...)
(net::RBMSplit{T})(σr, σc) where T = transpose(net.ar)*σr .+ transpose(net.ac)*σc .+ sum(logℒ.(net.b .+
                                                net.Wr*σr .+ net.Wc*σc))


function Base.show(io::IO, m::RBMSplit)
    print(io, "RBMSplit($(eltype(m.ar)), n=$(length(m.ar)), α=$(length(m.b)/length(m.ar)))")
end
Base.show(io::IO, ::MIME"text/plain", m::RBMSplit) = print(
"RBMSplit($(eltype(m.ar)), n=$(length(m.ar)), α=$(length(m.b)/length(m.ar)))")

# Cached version
mutable struct RBMSplitCache{T} <: NNCache{RBMSplit{T}}
    θ::Vector{T}
    θ_tmp::Vector{T}
    logℒθ::Vector{T}
    valid::Bool # = false
end

cache(net::RBMSplit{T}) where T =
    RBMSplitCache(Vector{T}(undef,length(net.b)),
                  Vector{T}(undef,length(net.b)),
                  Vector{T}(undef,length(net.b)),
                  false)

(net::RBMSplit)(c::RBMSplitCache, σ) = net(c, config(σ)...)
function (net::RBMSplit)(c::RBMSplitCache, σr,σc)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    T = eltype(θ)

    #θ .= net.b .+
    #        net.Wr*σr .+
    #            net.Wc*σc
    mul!(θ, net.Wr, σr)
    mul!(θ_tmp, net.Wc, σc)
    θ .+= net.b .+ θ_tmp
    #LinearAlgebra.BLAS.blascopy!(length(net.b), net.b, 1, θ, 1)
    #LinearAlgebra.BLAS.gemv!('N', one(T), net.Wr, σr, one(T), θ)
    #LinearAlgebra.BLAS.gemv!('N', one(T), net.Wc, σc, one(T), θ)

    logℒθ .= logℒ.(θ)
    logψ = dot(σr,net.ar) + dot(σc,net.ac) + sum(logℒθ)
    return logψ
end

function logψ_and_∇logψ!(∇logψ, net::RBMSplit, c::RBMSplitCache, σr,σc)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    T = eltype(θ)

    #θ .= net.b .+
    #        net.Wr*σr .+
    #            net.Wc*σc
    mul!(θ, net.Wr, σr)
    mul!(θ_tmp, net.Wc, σc)
    θ .+= net.b .+ θ_tmp

    logℒθ .= logℒ.(θ)
    logψ = dot(σr,net.ar) + dot(σc,net.ac) + sum(logℒθ)

    ∇logψ.ar .= σr
    ∇logψ.ac .= σc
    ∇logψ.b  .= ∂logℒ.(θ)
    ∇logψ.Wr .= ∂logℒ.(θ) .* transpose(σr)
    ∇logψ.Wc .= ∂logℒ.(θ) .* transpose(σc)

    return logψ, ∇logψ
end
