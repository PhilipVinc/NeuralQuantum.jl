mutable struct AccumulatorObsGrad{A,B,C,D,E,F} <: AbstractObservableAccumulator
    res::A
    ∇res::B

    ψ_σ::C       # Buffers alls the <σ|ψ> of the denominator
    ∇ψ_σ::D
    mel_buf::E      # ⟨σ|Ô|σ'⟩ in the buffer

    ∇logψ_acc::F
end

function AccumulatorObsGrad(net::NeuralNetwork, hilb, batch_sz)
    bnet      = cached(net, batch_sz)
    CT        = Complex{real(out_type(net))}
    v         = state(hilb, bnet)

    acc       = AccumulatorLogGradPsi(bnet, v, batch_sz)

    w         = trainable_first(net)
    RT        = real(eltype(w))

    res       = zero(eltype(out_similar(bnet)))
    ∇res      = grad_cache(CT, net)

    ∇0_buf    = vec_data(grad_cache(net))[1]
    mel_buf   = similar(w, CT, 1, batch_sz)

    return AccumulatorObsGrad(
        res, ∇res,
        res, ∇0_buf,
        mel_buf, acc)
end

@forward AccumulatorObsGrad.∇logψ_acc Base.length, Base.count,
    isfull

accum(a::AccumulatorObsGrad) = a.∇logψ_acc

function init!(c::AccumulatorObsGrad, σ, ψ_σ, ∇ψ_σ)
    c.ψ_σ     = ψ_σ
    c.∇ψ_σ    = ∇ψ_σ
    c.mel_buf .= 0
    c.res     = 0

    init!(accum(c), σ)
    return nothing
end


function (c::AccumulatorObsGrad)(mel::Number, cngs_l, cngs_r, v)
    isfull(c) && process_accumulator!(c)

    # If the matrix element is zero, don't do anything
    mel == 0.0 && return acc

    n_cngs_l = isnothing(cngs_l) ? 0 : length(cngs_l)
    n_cngs_r = isnothing(cngs_r) ? 0 : length(cngs_r)

    # If there are no changes, just sum it
    if n_cngs_l == 0 && n_cngs_r == 0
        c.res += mel
    else
        accum(c)(cngs_l, cngs_r)
        c.mel_buf[count(c)] = mel
    end

    return c
end

function (c::AccumulatorObsGrad)(mel::Number, cngs, v)
    isfull(c) && process_accumulator!(c)

    # If the matrix element is zero, don't do anything
    mel == 0.0 && return acc

    n_cngs = isnothing(cngs) ? 0 : length(cngs)

    # If there are no changes, just sum it
    if cngs == 0
        c.res += mel
    else
        accum(c)(cngs)
        c.mel_buf[count(c)] = mel
    end

    return c
end


function process_accumulator!(c::AccumulatorObsGrad)
    count(c) == 0 && return nothing

    ∇ψ_σp, ψ_σp = process_accumulator!(accum(c))

    ∇res  = vec_data(c.∇res)[1]
    ∇ψ_σp = vec_data(∇ψ_σp)[1]
    ∇ψ_σ  = c.∇ψ_σ

    c.mel_buf .*= exp.(ψ_σp .- c.ψ_σ)
    ∇ψ_σp     .= c.mel_buf .* (∇ψ_σp .- ∇ψ_σ)

    sum!(∇res, ∇ψ_σp)
    c.res += sum(c.mel_buf)

    return nothing
end

function finalize!(c::AccumulatorObsGrad)
    process_accumulator!(c)
    return c.res, c.∇res
end
