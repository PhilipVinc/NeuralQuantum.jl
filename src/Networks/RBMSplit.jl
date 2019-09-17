export RBMSplit

struct RBMSplit{VT,MT} <: MatrixNeuralNetwork
    ar::VT
    ac::VT
    b::VT
    Wr::MT
    Wc::MT
end
@treelike RBMSplit

"""
    RBMSplit([T=Complex{STD_REAL_PREC}], N, α, [initW, initb, inita])

Constructs a Restricted Bolzmann Machine to encode a vectorised density matrix,
with weights of type `T` (Defaults to ComplexF32), `2N` input neurons,
2N⋅α hidden neurons.
This network does not ensure positive-definitness of the density matrix.

`N` must match the size of the lattice.

The initial parameters of the neurons are initialized with a rescaled normal
distribution of width 0.01 for the coupling matrix and 0.05 for the local
biases. The default initializers can be overriden by specifying

initW=(dims...)->rescaled_normal(T, 0.01, dims...),
initb=(dims...)->rescaled_normal(T, 0.05, dims...),
initb=(dims...)->rescaled_normal(T, 0.01, dims...),

Refs:
    https://arxiv.org/abs/1902.07006
"""
RBMSplit(in::Int, α::Number, args...) = RBMSplit(ComplexF32, in, α, args...)
RBMSplit(T::Type, in, α,
         initW=(dims...)->rescaled_normal(T, 0.01, dims...),
         initb=(dims...)->rescaled_normal(T, 0.05, dims...),
         inita=(dims...)->rescaled_normal(T, 0.01, dims...)) =
    RBMSplit(inita(in), inita(in),
             initb(convert(Int, α*in)),
             initW(convert(Int, α*in), in), initW(convert(Int, α*in), in))

input_type(net::RBMSplit)  = real(eltype(net.ar))
weight_type(net::RBMSplit) = out_type(net)
out_type(net::RBMSplit)    = eltype(net.Wr)
input_shape(net::RBMSplit) = (length(net.ar), length(net.ac))
random_input_state(net::RBMSplit) =
    (eltype(net.ar).([rand(0:1) for i=1:length(net.ar)]), eltype(net.ar).([rand(0:1) for i=1:length(net.ar)]))
is_analytic(net::RBMSplit) = true


(net::RBMSplit)(σ::State) = net(config(σ)...)
(net::RBMSplit)(σr, σc)   = transpose(net.ar)*σr .+ transpose(net.ac)*σc .+ sum_autobatch(logℒ.(net.b .+
                                                        net.Wr*σr .+ net.Wc*σc))


function Base.show(io::IO, m::RBMSplit)
    print(io, "RBMSplit($(eltype(m.ar)), n=$(length(m.ar)), α=$(length(m.b)/length(m.ar)))")
end
Base.show(io::IO, ::MIME"text/plain", m::RBMSplit) = print(
"RBMSplit($(eltype(m.ar)), n=$(length(m.ar)), α=$(length(m.b)/length(m.ar)))")

# Cached version
mutable struct RBMSplitCache{VT,VS,VST} <: NNCache{RBMSplit}
    θ::VT
    θ_tmp::VT
    logℒθ::VT
    ∂logℒθ::VT

    # complex sigmas
    res::VS #batch
    res_tmp::VST #batch

    # states
    σr::VT
    σc::VT

    valid::Bool # = false
end

cache(net::RBMSplit) =
    RBMSplitCache(similar(net.b),
                  similar(net.b),
                  similar(net.b),
                  similar(net.b),
                  similar(net.b),
                  similar(net.b),
                  similar(net.b, length(net.ar)),
                  similar(net.b, length(net.ar)),
                  false)

(net::RBMSplit)(c::RBMSplitCache, σ::State) = net(c, config(σ))
(net::RBMSplit)(c::RBMSplitCache, (σr, σc)::Tuple{AbstractArray,AbstractArray}) = net(c, σr, σc)
function (net::RBMSplit)(c::RBMSplitCache, σr_r, σc_r)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    T = eltype(θ)

    # copy the states to complex valued states for the computations.
    σr = c.σr; copyto!(σr, σr_r)
    σc = c.σc; copyto!(σc, σc_r)

    #θ .= net.b .+
    #        net.Wr*σr .+
    #            net.Wc*σc
    mul!(θ, net.Wr, σr)
    mul!(θ_tmp, net.Wc, σc)
    θ .+= net.b .+ θ_tmp

    logℒθ .= logℒ.(θ)
    lnψ = dot(σr,net.ar) + dot(σc,net.ac) + sum(logℒθ)
    return lnψ
end

function logψ_and_∇logψ!(∇logψ, net::RBMSplit, c::RBMSplitCache, σr_r, σc_r)
    θ      = c.θ
    θ_tmp  = c.θ_tmp
    logℒθ  = c.logℒθ
    ∂logℒθ = c.∂logℒθ
    T      = eltype(θ)

    # copy the states to complex valued states for the computations.
    σr = c.σr; copyto!(σr, σr_r)
    σc = c.σc; copyto!(σc, σc_r)

    #θ .= net.b .+
    #        net.Wr*σr .+
    #            net.Wc*σc
    mul!(θ, net.Wr, σr)
    mul!(θ_tmp, net.Wc, σc)
    θ .+= net.b .+ θ_tmp

    logℒθ  .= logℒ.(θ)
    ∂logℒθ .= ∂logℒ.(θ)

    ∇logψ.ar .= σr
    ∇logψ.ac .= σc
    ∇logψ.b  .= ∂logℒθ
    ∇logψ.Wr .= ∂logℒθ .* transpose(σr)
    ∇logψ.Wc .= ∂logℒθ .* transpose(σc)

    lnψ = dot(σr,net.ar) + dot(σc,net.ac) + sum(logℒθ)
    return lnψ
end
