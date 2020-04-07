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
mutable struct AccumulatorLogPsi{N,A,B} <: AbstractMachineValAccumulator
    bnet::N         # A batched version of the cached neural network

    in_buf::A       # the matrix of Nsites x batchsz used as input
    out_buf::B      # The row vector of outputs

    buf_n::Int      # Counter for elements in buffer
    batch_sz::Int   # batch size
end

function AccumulatorLogPsi(net::NeuralNetwork, v, batch_sz)
    #bnet    = cached(net, batch_sz)
    bnet    = net

    w       = trainable_first(net)
    RT      = real(eltype(w))
    in_buf  = deepcopy(v)
    out_buf = out_similar(net)

    return AccumulatorLogPsi(
        bnet,
        in_buf, out_buf,
        0, batch_sz)
end


Base.length(a::AccumulatorLogPsi) = a.batch_sz
Base.count(a::AccumulatorLogPsi)  = a.buf_n
isfull(a::AccumulatorLogPsi) = count(a) == length(a)

"""
    init!(c::ScalarBatchAccumulator)

Resets the internal counter of the accumulator, deleting
all previously accumulated (but not computed) values.
"""
init!(c::AccumulatorLogPsi) = c.buf_n = 0
function init!(c::AccumulatorLogPsi, σ)
    init!(c)

    # Reset the state so that later is faster to apply the changes
    state_copy!(c.in_buf, σ)
end

function (c::AccumulatorLogPsi)(v::Union{AState, ADoubleState})
    @assert !isfull(c) "Pushed data to a full accumulator."

    # Increase the step in our internal buffer
    # this should be guaranteed to always be < max_capacity
    c.buf_n = c.buf_n + 1

    if v isa AState
        buf_i = unsafe_get_batch(c.in_buf, c.buf_n)
        buf_i .= v
    elseif v isa ADoubleState
        buf_r_i = unsafe_get_batch(row(c.in_buf), c.buf_n)
        buf_c_i = unsafe_get_batch(col(c.in_buf), c.buf_n)
        buf_r_i .= row(v)
        buf_c_i .= col(v)
    else
        throw("Not implemented")
    end

    return nothing
end

function (c::AccumulatorLogPsi)(cngs)
    @assert !isfull(c) "Pushed data to a full accumulator."
    c.buf_n = c.buf_n + 1

    σp = unsafe_get_batch(c.in_buf, c.buf_n)
    @inbounds apply!(σp, cngs)
end

function (c::AccumulatorLogPsi)(cngs_l, cngs_r)
    @assert !isfull(c) "Pushed data to a full accumulator."
    c.buf_n = c.buf_n + 1

    σp = unsafe_get_batch(c.in_buf, c.buf_n)
    @inbounds apply!(σp, cngs_l, cngs_r)
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
function process_accumulator!(c::AccumulatorLogPsi)
    out = c.out_buf

    # Compute the batch of logψ neural networks
    logψ!(out, c.bnet, c.in_buf)

    # Reset the counter of the batch accumulator
    init!(c)

    return out
end
