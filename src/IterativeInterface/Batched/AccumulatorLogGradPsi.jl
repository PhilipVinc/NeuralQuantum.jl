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
mutable struct AccumulatorLogGradPsi{N,A,B,C} <: AbstractMachineGradAccumulator
    bnet::N         # A batched version of the cached neural network

    in_buf::A       # the matrix of Nsites x batchsz used as input
    out_buf::B      # The row vector of outputs
    ∇out_buf::C

    buf_n::Int      # Counter for elements in buffer
    batch_sz::Int   # batch size
end

function AccumulatorLogGradPsi(net::NeuralNetwork, v, batch_sz)
    #bnet     = cached(net, batch_sz)
    bnet     = net

    w        = trainable_first(net)
    RT       = real(eltype(w))
    in_buf   = deepcopy(v)
    out_buf  = out_similar(net)
    ∇out_buf = grad_cache(net, batch_sz)

    return AccumulatorLogGradPsi(
        bnet,
        in_buf, out_buf, ∇out_buf,
        0, batch_sz)
end

Base.length(a::AccumulatorLogGradPsi) = a.batch_sz
Base.count(a::AccumulatorLogGradPsi)  = a.buf_n
isfull(a::AccumulatorLogGradPsi) = count(a) == length(a)

"""
    init!(c::ScalarBatchAccumulator)

Resets the internal counter of the accumulator, deleting
all previously accumulated (but not computed) values.
"""
init!(c::AccumulatorLogGradPsi) = c.buf_n = 0
function init!(c::AccumulatorLogGradPsi, σ)
    init!(c)

    # Reset the state so that later is faster to apply the changes
    statecopy!(c.in_buf, σ)
    return nothing
end

function (c::AccumulatorLogGradPsi)(v::Union{AState, ADoubleState})
    @assert !isfull(c) "Pushed data to a full accumulator."

    # Increase the step in our internal buffer
    # this should be guaranteed to always be < max_capacity
    c.buf_n = c.buf_n + 1

    buf_i = NeuralQuantum.unsafe_get_batch(c.in_buf, c.buf_n)
    statecopy!(buf_i, v)

    return nothing
end

function (c::AccumulatorLogGradPsi)(cngs)
    @assert !isfull(c) "Pushed data to a full accumulator."
    c.buf_n = c.buf_n + 1

    σp = unsafe_get_batch(c.in_buf, c.buf_n)
    apply!(σp, cngs)
end

function (c::AccumulatorLogGradPsi)(cngs_l, cngs_r)
    @assert !isfull(c) "Pushed data to a full accumulator."
    c.buf_n = c.buf_n + 1

    σp = unsafe_get_batch(c.in_buf, c.buf_n)
    apply!(σp, cngs_l, cngs_r)
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
function process_accumulator!(c::AccumulatorLogGradPsi)
    out  = c.out_buf
    ∇out = c.∇out_buf

    # Compute the batch of logψ neural networks
    logψ_and_∇logψ!(∇out, out, c.bnet, c.in_buf)

    # Reset the counter of the batch accumulator
    init!(c)

    return ∇out, out
end
