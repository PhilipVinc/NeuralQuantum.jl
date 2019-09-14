export NDMComplex

struct NDMComplex{VT,MT} <: MatrixNeuralNetwork
    a::VT
    b::VT
    c::VT
    W::MT
    X::MT
end

NDMComplex(args...) = NDMComplex(ComplexF32, args...)
NDMComplex(T::Type{<:Complex}, in, αh, αa,
    initW=(dims...)->rescaled_normal(T, 0.01, dims...),
    initb=(dims...)->rescaled_normal(T, 0.005, dims...),
    inita=(dims...)->rescaled_normal(T, 0.005, dims...)) =
    NDMComplex(inita(in),
               initb(convert(Int,αh*in)),
               initb(convert(Int,αa*in)),
               initW(convert(Int,αh*in), in),
               initW(convert(Int,αa*in), in))

input_type(net::NDMComplex{VT,MT}) where {VT,MT} = real(eltype(VT))
weight_type(net::NDMComplex) = eltype(net.a)
out_type(net::NDMComplex{VT,MT}) where {VT,MT} = eltype(VT)
input_shape(net::NDMComplex) = (length(net.b_μ), length(net.b_μ))
random_input_state(net::NDMComplex{VT,MT}) where {VT,MT} =
    (eltype(VT).([rand(0:1) for i=1:length(net.a)]), eltype(VT).([rand(0:1) for i=1:length(net.a)]))
is_analytic(net::NDMComplex) = true

@inline (net::NDMComplex)(σ::State) = net(config(σ)...)
@inline (net::NDMComplex)(σ::Tuple) = net(σ...)
function (W::NDMComplex)(σr, σc)
    T=eltype(W.a)

    θ_l = W.c +
            W.W * σr +
                conj(W.W) * σc

    θ_r   = W.b +
              W.X*σr

    θ_c   = W.b +
              W.X*σc

    logψ = dot(σr, W.a) + dot(W.a, σc) + #dot(σc, conj(W.a)) +
               sum(logℒ2.(θ_l))      +
                   sum(logℒ2.(θ_r))     +
                        sum(conj.(logℒ2.(θ_c)))
    return logψ
end

# Cached version
mutable struct NDMComplexCache{VT,VTR} <: NNCache{NDMComplex}
    θ_l_r::VT
    θ_l::VT
    θ_r::VT
    θ_c::VT

    logℒθ_l::VT
    logℒθ_r::VT
    logℒθ_c::VT

    ∂logℒθ_l::VT
    ∂logℒθ_r::VT
    ∂logℒθ_c::VT

    ∑σ::VT
    Δσ::VT

    σ_row_cache::VTR
    i_σ_row_cache::Int

    valid::Bool # = false
end

cache(net::NDMComplex) =
    NDMComplexCache(similar(net.c),
              similar(net.c),
              similar(net.b),
              similar(net.b),

              similar(net.c),
              similar(net.b),
              similar(net.b),

              similar(net.c),
              similar(net.b),
              similar(net.b),

              similar(real.(net.a)),
              similar(real.(net.a)),

              similar(real.(net.a)),
              -1,

              false)

(net::NDMComplex)(c::NDMComplexCache, σ) = net(c, config(σ)...)
function (W::NDMComplex)(c::NDMComplexCache, σr, σc)
    T       = eltype(W.W)
    θ_l     = c.θ_l
    θ_l_r   = c.θ_l_r
    θ_c     = c.θ_c
    θ_r     = c.θ_r
    logℒθ_l = c.logℒθ_l
    logℒθ_r = c.logℒθ_r
    logℒθ_c = c.logℒθ_c

    if !c.valid || c.σ_row_cache ≠ σr
        c.σ_row_cache .= σr
        c.valid = true

        #θ_l_r = W.c + W.W*σr
        mul!(θ_l_r, W.W, σr)
        θ_l_r .+= W.c

        #θ_r   = W.b +  W.X*σr
        mul!(θ_r, W.X, σr)
        θ_r .+= W.b

        logℒθ_r .= logℒ2.(θ_r)
    end

    #θ_l .+= conj(W.W) * σc
    mul!(θ_l, W.W, σc)
    conj!(θ_l)
    θ_l .+= θ_l_r

    #θ_c   = W.b + W.X*σc
    mul!(θ_c, W.X, σc)
    θ_c .+= W.b

    logℒθ_l .= logℒ2.(θ_l)
    logℒθ_c .= conj.(logℒ2.(θ_c))

    logψ = dot(W.a, σr) + dot(σc,W.a') +
           sum(logℒθ_l) + sum(logℒθ_r) + sum(logℒθ_c)
    return logψ
end

function logψ_and_∇logψ!(∇logψ, W::NDMComplex, c::NDMComplexCache, σr,σc)
    T       = eltype(W.W)
    θ_l     = c.θ_l
    θ_l_r   = c.θ_l_r
    θ_c     = c.θ_c
    θ_r     = c.θ_r
    logℒθ_l = c.logℒθ_l
    logℒθ_r = c.logℒθ_r
    logℒθ_c = c.logℒθ_c

    if !c.valid || c.σ_row_cache ≠ σr
        c.σ_row_cache .= σr
        c.valid = true

        #θ_l_r = W.c + W.W*σr
        mul!(θ_l_r, W.W, σr)
        θ_l_r .+= W.c

        #θ_r   = W.b +  W.X*σr
        mul!(θ_r, W.X, σr)
        θ_r .+= W.b

        logℒθ_r .= logℒ2.(θ_r)
    end

    #θ_l .+= conj(W.W) * σc
    mul!(θ_l, W.W, σc)
    conj!(θ_l)
    θ_l .+= θ_l_r

    #θ_c   = W.b + W.X*σc
    mul!(θ_c, W.X, σc)
    θ_c .+= W.b

    logℒθ_l .= logℒ2.(θ_l)
    logℒθ_c .= conj.(logℒ2.(θ_c))

    logψ = dot(W.a, σr) + dot(σc,W.a') +
           logℒθ_l + logℒθ_r + logℒθ_c
    # --- End common terms with computation of ψ --- #
    ∑σ    = c.∑σ
    Δσ    = c.Δσ
    c.∑σ .= σr .+ σc
    c.Δσ .= σr .- σc

    # Compute additional terms for derivatives
    ∂logℒθ_l = c.∂logℒθ_l; ∂logℒθ_l .= ∂logℒ2.(θ_l)
    ∂logℒθ_r = c.∂logℒθ_r; ∂logℒθ_r .= ∂logℒ2.(θ_r)
    ∂logℒθ_c = c.∂logℒθ_c; ∂logℒθ_c .= ∂logℒ2.(θ_c)
    p = one(T) + im*one(T)
    m = one(T) - im*one(T)

    # Store the derivatives
    #∇logψ.b .= ∑σ .- im .* Δσ
    #∇logψ.h .= p.* ∂logℒθ_r .+ m.* conj.(∂logℒθ_c)
    #∇logψ.c .= p.* ∂logℒθ_r .+ m.* conj.(∂logℒθ_c)
    ∇logψ.a .= ∑σ
    ∇logψ.b .= ∂logℒθ_r
    ∇logψ.c .= ∂logℒθ_l

    ∇logψ.W .= ∂logℒθ_r

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
