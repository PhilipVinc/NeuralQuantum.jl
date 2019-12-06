"""
    ScalarBatchAccumulator(net, state, batch_size)

A ScalarBatchAccumulator is used to evaluate the contribution to
local observables ⟨σ|Ô|σ'⟩ψ(σ')/ψ(σ) , but by computing the neural
network ψ(σ) in batches of size `batch_size`. This is essential
to extrract a speedup when using GPUs.

This is an internal implementation detail of NeuralQuantum, and should
not be relied upon.

Once constructed, a ScalarBatchAccumulator is supposed to be used as follows:
- if `isfull(sba) == true` you should not push new elements to it (an error
will be throw otherwise)
- data is pushed as `sba(⟨σ|Ô|σ'⟩, σ', ψ(σ))`.
The configuration should be passed as a vector (if ket) or as a tuple
of two vectors (if density matrix).
"""
mutable struct AccumulatorObsScalar{A,B,C,D} <: AbstractObservableAccumulator
    res::A

    ψ_σ::B       # Buffers alls the <σ|ψ> of the denominator
    mel_buf::C      # ⟨σ|Ô|σ'⟩ in the buffer

    logψ_acc::D
end

function AccumulatorObsScalar(net::NeuralNetwork, hilb, v, batch_sz)
    bnet      = cached(net, batch_sz)
    CT        = Complex{real(out_type(net))}
    v         = state(hilb, bnet)

    acc       = AccumulatorLogPsi(bnet, v, batch_sz)

    w         = trainable_first(net)
    RT        = real(eltype(w))

    res       = zero(eltype(out_similar(bnet)))
    mel_buf   = similar(w, CT, 1, batch_sz)

    return AccumulatorObsScalar(
        res,
        res,
        mel_buf, acc)
end

@forward AccumulatorObsScalar.logψ_acc Base.length, Base.count,
    isfull

accum(a::AccumulatorObsScalar) = a.logψ_acc

function init!(c::AccumulatorObsScalar, σ, ψ_σ)
    c.ψ_σ     = ψ_σ
    c.mel_buf .= 0
    c.res     = 0

    init!(accum(c), σ)
    return nothing
end

# This resets the value at the end of a process_accumulatr
function reset!(c::AccumulatorObsScalar, σ)
    c.mel_buf .= 0

    init!(accum(c), σ)
    return nothing
end


function (c::AccumulatorObsScalar)(mel::Number, cngs_l, cngs_r, v)
    isfull(c) && (process_accumulator!(c); reset!(c, v))

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

function (c::AccumulatorObsScalar)(mel::Number, cngs, v)
    isfull(c) && (process_accumulator!(c); reset!(c, v))

    # If the matrix element is zero, don't do anything
    mel == 0.0 && return c

    n_cngs = isnothing(cngs) ? 0 : length(cngs)

    # If there are no changes, just sum it
    if n_cngs == 0
        c.res += mel
    else
        accum(c)(cngs)
        c.mel_buf[count(c)] = mel
    end

    return c
end

"""
    process_accumulator!(c)

Processes all states stored in the accumulator, by computing their
relative local contribution.

It is safe to call this even if the accumulator is not full. In this
case all data beyond the count should be disregarded as it was
not initialized.

The output will be returned. You should not assume ownership of
the output, as it is preallocated and will be used for further
computations of the accumulator.
"""
function process_accumulator!(c::AccumulatorObsScalar)
    count(c) == 0 && return nothing

    ψ_σp = process_accumulator!(accum(c))

    c.mel_buf .*= exp.(ψ_σp .- c.ψ_σ)
    c.res += sum(c.mel_buf)

    return nothing
end

function finalize!(c::AccumulatorObsScalar)
    process_accumulator!(c)
    return c.res
end
