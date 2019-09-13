export RBMSplit

struct RBMSplit{VT,MT} <: MatrixNeuralNetwork
    ar::VT
    ac::VT
    b::VT
    Wr::MT
    Wc::MT
end
@treelike RBMSplit

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
(net::RBMSplit)(σr, σc)   = transpose(net.ar)*σr .+ transpose(net.ac)*σc .+ sum(logℒ.(net.b .+
                                                        net.Wr*σr .+ net.Wc*σc))


function Base.show(io::IO, m::RBMSplit)
    print(io, "RBMSplit($(eltype(m.ar)), n=$(length(m.ar)), α=$(length(m.b)/length(m.ar)))")
end
Base.show(io::IO, ::MIME"text/plain", m::RBMSplit) = print(
"RBMSplit($(eltype(m.ar)), n=$(length(m.ar)), α=$(length(m.b)/length(m.ar)))")

# Cached version
mutable struct RBMSplitCache{VT} <: NNCache{RBMSplit}
    θ::VT
    θ_tmp::VT
    logℒθ::VT
    ∂logℒθ::VT

    # complex sigmas
    σr::VT
    σc::VT

    # states

    valid::Bool # = false
end

cache(net::RBMSplit) =
    RBMSplitCache(similar(net.b),
                  similar(net.b),
                  similar(net.b),
                  similar(net.b),
                  similar(net.b, length(net.ar)),
                  similar(net.b, length(net.ar)),
                  false)

(net::RBMSplit)(c::RBMSplitCache, σ) = net(c, config(σ)...)
function (net::RBMSplit)(c::RBMSplitCache, σr_r, σc_r)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    T = eltype(θ)

    # copy the states to complex valued states for the computations.
    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)

    #θ .= net.b .+
    #        net.Wr*σr .+
    #            net.Wc*σc
    mul!(θ, net.Wr, σr)
    mul!(θ_tmp, net.Wc, σc)
    θ .+= net.b .+ θ_tmp

    logℒθ .= logℒ.(θ)
    logψ = dot(σr,net.ar) + dot(σc,net.ac) + sum(logℒθ)
    return logψ
end

function logψ_and_∇logψ!(∇logψ, net::RBMSplit, c::RBMSplitCache, σr_r, σc_r)
    θ      = c.θ
    θ_tmp  = c.θ_tmp
    logℒθ  = c.logℒθ
    ∂logℒθ = c.∂logℒθ
    T      = eltype(θ)

    # copy the states to complex valued states for the computations.
    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)

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

    logψ = dot(σr,net.ar) + dot(σc,net.ac) + sum(logℒθ)
    return logψ, ∇logψ
end
