export NDM

struct NDM{VT,MT} <: MatrixNeuralNetwork
    b_μ::VT
    h_μ::VT
    w_μ::MT
    u_μ::MT

    b_λ::VT
    h_λ::VT
    d_λ::VT
    w_λ::MT
    u_λ::MT
end
@treelike NDM

"""
    NDM([T=STD_REAL_PREC], N, αₕ, αₐ, [initW, initb, inita])

Constructs a Neural Density Matrix with numerical precision `T` (Defaults to
Float32), `N` input neurons, N⋅αₕ hidden neurons and N⋅αₐ ancillary neurons.
This network ensure that the density matrix is always positive definite.

The number of input neurons `N` must match the size of the lattice.

The initial parameters of the neurons are initialized with a rescaled normal
distribution of width 0.01 for the coupling matrix and 0.005 for the local
biases. The default initializers can be overriden by specifying

initW=(dims...)->rescaled_normal(T, 0.01, dims...),
initb=(dims...)->rescaled_normal(T, 0.005, dims...),
inita=(dims...)->rescaled_normal(T, 0.005, dims...))

Refs:
    https://arxiv.org/abs/1801.09684
    https://arxiv.org/abs/1902.10104
"""
NDM(args...) = NDM(STD_REAL_PREC, args...)
NDM(T::Type{<:Real}, in, αh, αa,
    initW=(dims...)->rescaled_normal(T, 0.01, dims...),
    initb=(dims...)->rescaled_normal(T, 0.005, dims...),
    inita=(dims...)->rescaled_normal(T, 0.005, dims...)) =
    NDM(inita(in),
        initb(convert(Int,αh*in)),
        initW(convert(Int,αh*in), in),
        initW(convert(Int,αa*in), in),
        inita(in),
        initb(convert(Int,αh*in)),
        initb(convert(Int,αa*in)),
        initW(convert(Int,αh*in), in),
        initW(convert(Int,αa*in), in))

input_type(net::NDM{VT,MT}) where {VT,MT} = eltype(VT)
weight_type(net::NDM) = input_type(net)
out_type(net::NDM{VT,MT}) where {VT,MT} = Complex{eltype(VT)}
input_shape(net::NDM) = (length(net.b_μ), length(net.b_μ))
random_input_state(net::NDM{VT,MT}) where {VT,MT} =
    (eltype(VT).([rand(0:1) for i=1:length(net.b_μ)]), eltype(VT).([rand(0:1) for i=1:length(net.b_μ)]))
is_analytic(net::NDM) = true

Base.show(io::IO, m::NDM) = print(io,
    "NDM($(eltype(m.b_μ)), n=$(length(m.b_μ)), αₕ=$(length(m.h_μ)/length(m.b_μ)), αₐ=$(length(m.d_λ)/length(m.b_μ)))")

Base.show(io::IO, ::MIME"text/plain", m::NDM) = print(io,
    "NDM($(eltype(m.b_μ)), n=$(length(m.b_μ)), α=$(length(m.h_μ)/length(m.b_μ)), αₐ=$(length(m.d_λ)/length(m.b_μ)))")


@inline (net::NDM)(σ::State) = net(config(σ)...)
@inline (net::NDM)(σ::Tuple) = net(σ...)
function (W::NDM)(σr, σc)
    T=eltype(W.u_λ)
    ∑logℒ_λ_σ = sum_autobatch(logℒ.(W.h_λ .+ W.w_λ*σr))
    ∑logℒ_μ_σ = sum_autobatch(logℒ.(W.h_μ .+ W.w_μ*σr))

    ∑logℒ_λ_σp = sum_autobatch(logℒ.(W.h_λ .+ W.w_λ*σc))
    ∑logℒ_μ_σp = sum_autobatch(logℒ.(W.h_μ .+ W.w_μ*σc))

    ∑σ = σr .+ σc
    Δσ = σr .- σc

    _Π = T(0.5)  .* W.u_λ*∑σ .+
           T(0.5)im .* W.u_μ*Δσ .+ W.d_λ

    Γ_λ = T(0.5)   * (∑logℒ_λ_σ + ∑logℒ_λ_σp + transpose(W.b_λ)*∑σ )
    Γ_μ = T(0.5)im * (∑logℒ_μ_σ - ∑logℒ_μ_σp + transpose(W.b_μ)*Δσ )
    Π  = sum_autobatch(logℒ.(_Π))

    lnψ = Γ_λ + Γ_μ + Π
    return lnψ
end

# Cached version
mutable struct NDMCache{T,VT,VCT} <: NNCache{NDM}
    θλ_σ::VT
    θμ_σ::VT
    θλ_σp::VT
    θμ_σp::VT

    θλ_σ_tmp::VT
    θμ_σ_tmp::VT
    θλ_σp_tmp::VT
    θμ_σp_tmp::VT

    σr::VT
    σc::VT
    ∑σ::VT
    Δσ::VT

    ∑logℒ_λ_σ::T
    ∑logℒ_μ_σ::T
    ∂logℒ_λ_σ::VT
    ∂logℒ_μ_σ::VT
    ∂logℒ_λ_σp::VT
    ∂logℒ_μ_σp::VT
    _Π::VCT
    _Π2::VCT
    _Π_tmp::VT
    ∂logℒ_Π::VCT

    σ_row_cache::VT
    i_σ_row_cache::Int

    valid::Bool # = false
end

cache(net::NDM) =
    NDMCache(similar(net.h_μ),
              similar(net.h_μ),
              similar(net.h_μ),
              similar(net.h_μ),

              similar(net.h_μ),
              similar(net.h_μ),
              similar(net.h_μ),
              similar(net.h_μ),

              similar(net.b_μ),
              similar(net.b_μ),
              similar(net.b_μ),
              similar(net.b_μ),

              zero(eltype(net.b_μ)), zero(eltype(net.b_μ)),
              similar(net.h_μ), similar(net.h_μ),
              similar(net.h_μ), similar(net.h_μ),
              similar(net.d_λ, complex(eltype(net.d_λ))),
              similar(net.d_λ, complex(eltype(net.d_λ))),
              similar(net.d_λ, eltype(net.d_λ)),
              similar(net.d_λ, complex(eltype(net.d_λ))),
                  #VT(T, length(net.h_μ)), zeros(T, length(net.h_μ)),
                  #VT(T, length(net.h_μ)), zeros(T, length(net.h_μ)),
                  #zeros(Complex{T}, length(net.d_λ)), zeros(T, length(net.d_λ)), zeros(Complex{T}, length(net.d_λ)),

              similar(net.b_μ),
              -1,

              false)


(net::NDM)(c::NDMCache, σ::State) = net(c, config(σ))
(net::NDM)(c::NDMCache, (σr, σc)::Tuple{AbstractArray,AbstractArray}) = net(c, σr, σc)
function (W::NDM)(c::NDMCache, σr, σc)
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    θλ_σ    = c.θλ_σ
    θμ_σ    = c.θμ_σ
    θλ_σp   = c.θλ_σp
    θμ_σp   = c.θμ_σp
    _Π      = c._Π
    _Π_tmp  = c._Π_tmp
    T       = eltype(c.θλ_σ)

    if !c.valid || c.σ_row_cache ≠ σr
        c.σ_row_cache .= σr
        c.valid = true

        # θλ_σ .= W.h_λ + W.w_λ*σr
        LinearAlgebra.BLAS.blascopy!(length(W.h_λ), W.h_λ, 1, θλ_σ, 1)
        LinearAlgebra.BLAS.gemv!('N', one(T), W.w_λ, σr, one(T), θλ_σ)

        # θμ_σ .= W.h_μ + W.w_μ*σr
        LinearAlgebra.BLAS.blascopy!(length(W.h_μ), W.h_μ, 1, θμ_σ, 1)
        LinearAlgebra.BLAS.gemv!('N', one(T), W.w_μ, σr, one(T), θμ_σ)

        # cache.∑logℒ_λ_σ = sum(logℒ.(θλ_σ, (NT(),)))
        c.θλ_σ_tmp .= logℒ.(θλ_σ)
        c.∑logℒ_λ_σ = sum(c.θλ_σ_tmp)

        # cache.∑logℒ_μ_σ = sum(logℒ.(θμ_σ, (NT(),)))
        c.θμ_σ_tmp .= logℒ.(θμ_σ)
        c.∑logℒ_μ_σ = sum(c.θμ_σ_tmp)

        c.∂logℒ_λ_σ .= ∂logℒ.(θλ_σ)
        c.∂logℒ_μ_σ .= ∂logℒ.(θμ_σ)
    end

    ∑logℒ_λ_σ = c.∑logℒ_λ_σ
    ∑logℒ_μ_σ = c.∑logℒ_μ_σ
    ∂logℒ_λ_σ = c.∂logℒ_λ_σ
    ∂logℒ_μ_σ = c.∂logℒ_μ_σ

    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    #θλ_σp .= W.h_λ + W.w_λ*σc
    #θμ_σp .= W.h_μ + W.w_μ*σc
    LinearAlgebra.BLAS.blascopy!(length(W.h_λ), W.h_λ, 1, θλ_σp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(1.0), W.w_λ, σc, T(1.0), θλ_σp)
    LinearAlgebra.BLAS.blascopy!(length(W.h_μ), W.h_μ, 1, θμ_σp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(1.0), W.w_μ, σc, T(1.0), θμ_σp)

    #∑logℒ_λ_σp = sum(logℒ.(θλ_σp, (NT(),)))
    c.θλ_σp_tmp .= logℒ.(θλ_σp)
    ∑logℒ_λ_σp = sum(c.θλ_σp_tmp)

    #∑logℒ_μ_σp = sum(logℒ.(θμ_σp, (NT(),)))
    c.θμ_σp_tmp .= logℒ.(θμ_σp)
    ∑logℒ_μ_σp = sum(c.θμ_σp_tmp)

    #_Π         = (T(0.5)  * W.u_λ*∑σ
    #               + T(0.5)im * W.u_μ*Δσ .+ W.d_λ)
    LinearAlgebra.BLAS.blascopy!(length(W.d_λ), W.d_λ, 1, _Π_tmp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(0.5), W.u_λ, ∑σ, one(T), _Π_tmp)
    _Π .= _Π_tmp
    LinearAlgebra.BLAS.gemv!('N', T(0.5), W.u_μ, Δσ, T(0.0), _Π_tmp)
    _Π .+= T(1.0)im.* _Π_tmp

    Γ_λ = T(0.5) * (∑logℒ_λ_σ + ∑logℒ_λ_σp + ∑σ ⋅ W.b_λ)
    Γ_μ = T(0.5) * (∑logℒ_μ_σ - ∑logℒ_μ_σp + Δσ ⋅ W.b_μ)

    # Π = sum(logℒ.(_Π, (NT(),)))
    _Π .= logℒ.(_Π)
    Π   = sum(_Π)

    logψ = Γ_λ + T(1.0)im * Γ_μ + Π
    return logψ
end

function logψ_and_∇logψ!(∇logψ, W::NDM, c::NDMCache, σr,σc)
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    θλ_σ    = c.θλ_σ
    θμ_σ    = c.θμ_σ
    θλ_σp   = c.θλ_σp
    θμ_σp   = c.θμ_σp
    _Π      = c._Π
    _Π_tmp  = c._Π_tmp
    T       = eltype(θλ_σ)

    if !c.valid || c.σ_row_cache ≠ σr
        c.σ_row_cache .= σr
        c.valid = true

        # θλ_σ .= W.h_λ + W.w_λ*σr
        LinearAlgebra.BLAS.blascopy!(length(W.h_λ), W.h_λ, 1, θλ_σ, 1)
        LinearAlgebra.BLAS.gemv!('N', one(T), W.w_λ, σr, one(T), θλ_σ)

        # θμ_σ .= W.h_μ + W.w_μ*σr
        LinearAlgebra.BLAS.blascopy!(length(W.h_μ), W.h_μ, 1, θμ_σ, 1)
        LinearAlgebra.BLAS.gemv!('N', one(T), W.w_μ, σr, one(T), θμ_σ)

        # cache.∑logℒ_λ_σ = sum(logℒ.(θλ_σ, (NT(),)))
        c.θλ_σ_tmp .= logℒ.(θλ_σ)
        c.∑logℒ_λ_σ = sum(c.θλ_σ_tmp)

        # cache.∑logℒ_μ_σ = sum(logℒ.(θμ_σ, (NT(),)))
        c.θμ_σ_tmp .= logℒ.(θμ_σ)
        c.∑logℒ_μ_σ = sum(c.θμ_σ_tmp)

        c.∂logℒ_λ_σ .= ∂logℒ.(θλ_σ)
        c.∂logℒ_μ_σ .= ∂logℒ.(θμ_σ)
    end

    ∑logℒ_λ_σ = c.∑logℒ_λ_σ
    ∑logℒ_μ_σ = c.∑logℒ_μ_σ
    ∂logℒ_λ_σ = c.∂logℒ_λ_σ
    ∂logℒ_μ_σ = c.∂logℒ_μ_σ

    # --- Common terms with computation of ψ --- #
    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    #θλ_σp .= W.h_λ + W.w_λ*σc
    #θμ_σp .= W.h_μ + W.w_μ*σc
    LinearAlgebra.BLAS.blascopy!(length(W.h_λ), W.h_λ, 1, θλ_σp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(1.0), W.w_λ, σc, T(1.0), θλ_σp)
    LinearAlgebra.BLAS.blascopy!(length(W.h_μ), W.h_μ, 1, θμ_σp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(1.0), W.w_μ, σc, T(1.0), θμ_σp);

    #∑logℒ_λ_σp = sum(logℒ.(θλ_σp, (NT(),)))
    c.θλ_σp_tmp .= logℒ.(θλ_σp)
    ∑logℒ_λ_σp = sum(c.θλ_σp_tmp)

    #∑logℒ_μ_σp = sum(logℒ.(θμ_σp, (NT(),)))
    c.θμ_σp_tmp .= logℒ.(θμ_σp)
    ∑logℒ_μ_σp = sum(c.θμ_σp_tmp)

    #_Π         = (T(0.5)  * W.u_λ*∑σ
    #               + T(0.5)im * W.u_μ*Δσ .+ W.d_λ)
    LinearAlgebra.BLAS.blascopy!(length(W.d_λ), W.d_λ, 1, _Π_tmp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(0.5), W.u_λ, ∑σ, T(1.0), _Π_tmp)
    _Π .= _Π_tmp
    LinearAlgebra.BLAS.gemv!('N', T(0.5), W.u_μ, Δσ, T(0.0), _Π_tmp)
    _Π .+= T(1.0)im.* _Π_tmp
    #@info "_Π diff " maximum(abs.(_Π - (T(0.5)  * transpose(transpose(∑σ)*W.u_λ) + T(0.5)im* transpose(transpose(Δσ)*W.u_μ) .+ W.d_λ)))

    # --- End common terms with computation of ψ --- #

    # Compute additional terms for derivatives
    ∂logℒ_λ_σp = c.∂logℒ_λ_σp; ∂logℒ_λ_σp .= ∂logℒ.(θλ_σp)
    ∂logℒ_μ_σp = c.∂logℒ_μ_σp; ∂logℒ_μ_σp .= ∂logℒ.(θμ_σp)
    ∂logℒ_Π    = c.∂logℒ_Π;    ∂logℒ_Π    .= ∂logℒ.(_Π)

    # Store the derivatives
    ∇logψ.b_λ .= T(0.5)   .* ∑σ
    ∇logψ.b_μ .= T(0.5)im .* Δσ

    ∇logψ.h_λ .= T(0.5)   .* (∂logℒ_λ_σ .+ ∂logℒ_λ_σp)
    ∇logψ.h_μ .= T(0.5)im .* (∂logℒ_μ_σ .- ∂logℒ_μ_σp)

    ∇logψ.w_λ .= T(0.5)   .* (∂logℒ_λ_σ.*transpose(σr) .+ ∂logℒ_λ_σp.*transpose(σc))
    ∇logψ.w_μ .= T(0.5)im .* (∂logℒ_μ_σ.*transpose(σr) .- ∂logℒ_μ_σp.*transpose(σc))

    ∇logψ.d_λ .= ∂logℒ_Π
    ∇logψ.u_λ .= T(0.5) .* ∂logℒ_Π .* transpose(∑σ)
    ∇logψ.u_μ .= T(0.5)im .*  ∂logℒ_Π .* transpose(Δσ)

    Γ_λ = T(0.5) * (∑logℒ_λ_σ + ∑logℒ_λ_σp + ∑σ ⋅ W.b_λ)
    Γ_μ = T(0.5) * (∑logℒ_μ_σ - ∑logℒ_μ_σp + Δσ ⋅ W.b_μ)
    _Π .= logℒ.(_Π)
    Π   = sum(_Π)
    logψ = Γ_λ + T(1.0)im * Γ_μ + Π

    return logψ
end

function vectorize_gradient(net::NDM{T}, gradient::NamedTuple) where T
    fnames_μ = [:b_μ, :h_μ, :w_μ, :u_μ]
    fnames_λ = [:b_λ, :h_λ, :d_λ, :w_λ, :u_λ]
    fnames_sets = [fnames_μ, fnames_λ]
    kvpairs    =Dict{Symbol, Any}()
    params_vecs = []

    for fnames=fnames_sets
        lens = Int[1]
        for f=fnames
            push!(lens, length(gradient[f]))
        end
        n_params   = sum(lens)-1
        indices    = cumsum(lens)

        params_vec = Vector{Complex{T}}(undef, n_params)
        push!(params_vecs, params_vec)
        i=1;
        for f=fnames
            @views datavec = params_vec[indices[i]:indices[i+1]-1]
            reshpd_params  = reshape(datavec, size(gradient[f]))
            reshpd_params .= gradient[f]
            push!(kvpairs, f=>reshpd_params)
            i = i+1
        end
    end
    newgrad = (;kvpairs...)
    return (newgrad, params_vecs)
end
