export NDMComplex

struct NDMComplex{VT,MT} <: MatrixNeuralNetwork
    a::VT
    b::VT
    c::VT
    W::MT
    X::MT
end
@treelike NDMComplex

"""
    NDMComplex([T=STD_REAL_PREC], N, αₕ, αₐ, [initW, initb, inita])

Constructs a Neural Density Matrix with numerical precision `T` (Defaults to
Float32), `N` input neurons, N⋅αₕ hidden neurons and N⋅αₐ ancillary neurons.
This network ensure that the density matrix is always positive definite.

The number of input neurons `N` must match the size of the lattice.

This network is very similar to `NDM`, but instead of splitting the modulus and
phase of the density matrix in two disjoint networks, it uses a single network
with complex weights.

The initial parameters of the neurons are initialized with a rescaled normal
distribution of width 0.01 for the coupling matrix and 0.005 for the local
biases. The default initializers can be overriden by specifying

initW=(dims...)->rescaled_normal(T, 0.01, dims...),
initb=(dims...)->rescaled_normal(T, 0.005, dims...),
inita=(dims...)->rescaled_normal(T, 0.005, dims...))

Refs:
"""
NDMComplex(args...) = NDMComplex(Complex{STD_REAL_PREC}, args...)
NDMComplex(::Real, ::Int) = throw("NDMComplex needs complex type")
NDMComplex(T::Type{<:Complex}, in, αh, αa,
    initW=(dims...)->rescaled_normal(T, 0.01, dims...),
    initb=(dims...)->rescaled_normal(T, 0.005, dims...),
    inita=(dims...)->rescaled_normal(T, 0.005, dims...)) =
    NDMComplex(inita(in),
               initb(convert(Int,αh*in)),
               initb(convert(Int,αa*in)),
               initW(convert(Int,αa*in), in),
               initW(convert(Int,αh*in), in))

input_type(net::NDMComplex{VT,MT}) where {VT,MT} = real(eltype(VT))
weight_type(net::NDMComplex) = eltype(net.a)
out_type(net::NDMComplex{VT,MT}) where {VT,MT} = eltype(VT)
input_shape(net::NDMComplex) = (length(net.a), length(net.a))
random_input_state(net::NDMComplex{VT,MT}) where {VT,MT} =
    (input_type(net).([rand(0:1) for i=1:length(net.a)]), input_type(net).([rand(0:1) for i=1:length(net.a)]))
is_analytic(net::NDMComplex) = false

Base.show(io::IO, m::NDMComplex) = print(io,
    "NDMComplex($(eltype(m.a)), n=$(length(m.a)), αₕ=$(length(m.b)/length(m.a)), αₐ=$(length(m.c)/length(m.a)))")

Base.show(io::IO, ::MIME"text/plain", m::NDMComplex) = print(io,
    "NDMComplex($(eltype(m.a)), n=$(length(m.a)), α=$(length(m.b)/length(m.a)), αₐ=$(length(m.c)/length(m.a)))")

@inline (net::NDMComplex)(σ::State) = net(config(σ)...)
@inline (net::NDMComplex)(σ::Tuple) = net(σ...)
function (W::NDMComplex)(σ, σp)
    T=eltype(W.a)

    θ̃     = W.c +
              W.W * σ +
                conj(W.W) * σp

    θ_σ   = W.b +
              W.X*σ

    θ_σp   = W.b +
              W.X*σp

    logψ = dot(σ, W.a) + dot(W.a, σp) + #dot(σp, conj(W.a)) +
               sum(logℒ2.(θ̃))    +
                   sum(logℒ2.(θ_σ))  +
                        sum(logℒ2.(conj.(θ_σp))) + log(T(1/8))
    return logψ
end

# Cached version
mutable struct NDMComplexCache{VT,VTR} <: NNCache{NDMComplex}
    θ̃_r::VT
    θ̃::VT
    θ_σp::VT
    θ_σ::VT

    logℒθ̃::VT
    logℒθ_σp::VT
    logℒθ_σ::VT

    ∂logℒθ̃::VT
    ∂logℒθ_σp::VT
    ∂logℒθ_σ::VT

    σ::VT
    σp::VT
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

              similar(net.a),
              similar(net.a),
              similar(net.a),
              similar(net.a),

              similar(real.(net.a)),
              -1,

              false)

(net::NDMComplex)(c::NDMComplexCache, σ) = net(c, config(σ)...)
function (W::NDMComplex)(c::NDMComplexCache, σ_r, σp_r)
    T        = eltype(W.W)
    θ̃        = c.θ̃
    θ̃_r      = c.θ̃_r
    θ_σ      = c.θ_σ
    θ_σp     = c.θ_σp
    logℒθ̃    = c.logℒθ̃
    logℒθ_σ  = c.logℒθ_σ
    logℒθ_σp = c.logℒθ_σp

    # copy the states to complex valued states for the computations.
    σ  = c.σ;  copyto!(σ,  σ_r)
    σp = c.σp; copyto!(σp, σp_r)

    if !c.valid || c.σ_row_cache ≠ σ
        c.σ_row_cache .= σ
        c.valid = true

        #θ̃_r = W.c + W.W*σ
        mul!(θ̃_r, W.W, σ)
        θ̃_r .+= W.c

        #θ_σp   = W.b +  W.X*σ
        mul!(θ_σ, W.X, σ)
        θ_σ .+= W.b

        logℒθ_σ .= logℒ2.(θ_σ)
    end

    #θ̃ .+= conj(W.W) * σp
    mul!(θ̃, W.W, σp)
    conj!(θ̃)
    θ̃ .+= θ̃_r

    #θ_σ   = W.b + W.X*σp
    mul!(θ_σp, W.X, σp)
    θ_σp .+= W.b

    logℒθ̃    .= logℒ2.(θ̃)
    logℒθ_σp .= logℒ2.(conj.(θ_σp))

    dotσ  = dot(σ, W.a)
    dotσp = dot(W.a, σp)
    logψ  = log(T(1/8)) +  sum(logℒθ_σp) + sum(logℒθ_σ) + sum(logℒθ̃) +
              dotσ + dotσp

    return logψ
end

function logψ_and_∇logψ!(∇logψ, W::NDMComplex, c::NDMComplexCache, σ_r, σp_r)
    T        = eltype(W.W)
    θ̃        = c.θ̃
    θ̃_r      = c.θ̃_r
    θ_σ      = c.θ_σ
    θ_σp     = c.θ_σp
    logℒθ̃    = c.logℒθ̃
    logℒθ_σ  = c.logℒθ_σ
    logℒθ_σp = c.logℒθ_σp

    # copy the states to complex valued states for the computations.
    σ  = c.σ;  copyto!(σ,  σ_r)
    σp = c.σp; copyto!(σp, σp_r)

    if !c.valid || c.σ_row_cache ≠ σ
        c.σ_row_cache .= σ
        c.valid = true

        #θ̃_r = W.c + W.W*σ
        mul!(θ̃_r, W.W, σ)
        θ̃_r .+= W.c

        #θ_σp   = W.b +  W.X*σ
        mul!(θ_σ, W.X, σ)
        θ_σ .+= W.b

        logℒθ_σ .= logℒ2.(θ_σ)
    end

    #θ̃ .+= conj(W.W) * σp
    mul!(θ̃, W.W, σp)
    conj!(θ̃)
    θ̃ .+= θ̃_r

    #θ_σ   = W.b + W.X*σp
    mul!(θ_σp, W.X, σp)
    θ_σp .+= W.b

    logℒθ̃    .= logℒ2.(θ̃)
    logℒθ_σp .= logℒ2.(conj.(θ_σp))

    dotσ  = dot(σ, W.a)
    dotσp = dot(W.a, σp)
    logψ  = log(T(1/8)) +  sum(logℒθ_σp) + sum(logℒθ_σ) + sum(logℒθ̃) +
              dotσ + dotσp
    # --- End common terms with computation of ψ --- #
    ∑σ    = c.∑σ
    Δσ    = c.Δσ
    c.∑σ .= σ .+ σp
    c.Δσ .= σ .- σp

    # Compute additional terms for derivatives
    ∂logℒθ̃    = c.∂logℒθ̃;    ∂logℒθ̃    .= ∂logℒ2.(θ̃)
    ∂logℒθ_σ  = c.∂logℒθ_σ;  ∂logℒθ_σ  .= ∂logℒ2.(θ_σ)
    ∂logℒθ_σp = c.∂logℒθ_σp; ∂logℒθ_σp .= ∂logℒ2.(conj.(θ_σp))

    # Store the derivatives
    ∇logψ_r = real(∇logψ)
    ∇logψ_i = imag(∇logψ)

    ∇logψ_r.a .=     ∑σ
    ∇logψ_i.a .= im.*Δσ

    ∇logψ_r.b .=      ∂logℒθ_σ .+ ∂logℒθ_σp
    ∇logψ_i.b .= im.*(∂logℒθ_σ .- ∂logℒθ_σp)

    ∇logψ_r.X .=      ∂logℒθ_σ.*transpose(σ) .+ ∂logℒθ_σp.*transpose(σp)
    ∇logψ_i.X .= im.*(∂logℒθ_σ.*transpose(σ) .- ∂logℒθ_σp.*transpose(σp))

    ∇logψ_r.c .=      ∂logℒθ̃
    ∇logψ_i.c .=      ∂logℒθ̃

    ∇logψ_r.W .=      ∂logℒθ̃.*transpose(∑σ)
    ∇logψ_i.W .=  im.*∂logℒθ̃.*transpose(Δσ)

    return logψ
end
