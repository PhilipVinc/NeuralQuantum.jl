
mutable struct AccumulatorObsScalarGPU{A,B,C,D} <: AbstractObservableAccumulator
    res::A

    ψ_σ::B          # Buffers alls the <σ|ψ> of the denominator
    Δ_σ::C
    mel_buf::C      # ⟨σ|Ô|σ'⟩ in the buffer

    logψ_acc::D
end

function AccumulatorObsScalar(net::NeuralNetwork, hilb, v::Union{gpuAStateBatch, gpuADoubleStateBatch},
                              batch_sz)
    bnet      = cached(net, batch_sz)
    CT        = Complex{real(out_type(net))}
    v         = state(hilb, bnet)

    acc       = AccumulatorLogPsi(bnet, v, batch_sz)

    w         = trainable_first(net)
    RT        = real(eltype(w))

    res       = zero(eltype(out_similar(bnet)))
    mel_buf   = collect(similar(w, CT, 1, batch_sz))
    Δ_σ       = similar(mel_buf)

    return AccumulatorObsScalarGPU(
        res,
        res,
        Δ_σ, mel_buf,
        acc)
end

@forward AccumulatorObsScalarGPU.logψ_acc Base.length, Base.count,
    isfull

accum(a::AccumulatorObsScalarGPU) = a.logψ_acc

function init!(c::AccumulatorObsScalarGPU, σ, ψ_σ)
    c.ψ_σ     = ψ_σ
    c.mel_buf .= 0
    c.res     = 0

    init!(accum(c), σ)
    return nothing
end


function (c::AccumulatorObsScalarGPU)(mel::Number, cngs_l, cngs_r, v)
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

function (c::AccumulatorObsScalarGPU)(mel::Number, cngs, v)
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

function process_accumulator!(c::AccumulatorObsScalarGPU)
    ψ_σp = process_accumulator!(accum(c))

    ψ_σp .= exp.(ψ_σp .- c.ψ_σ)
    copy!(c.Δ_σ, ψ_σp)

    c.mel_buf .*= Δ_σ
    c.res += sum(c.mel_buf)

    return nothing
end

function finalize!(c::AccumulatorObsScalarGPU)
    process_accumulator!(c)
    return c.res
end
