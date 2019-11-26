mutable struct AccumulatorLogPsiGPU{N,Ac,Ag,B} <: AbstractMachineValAccumulator
    bnet::N         # A batched version of the cached neural network

    in_buf_cpu::Ac  # the matrix of Nsites x batchsz used as input on cpu
    in_buf_gpu::Ag  # the matrix of Nsites x batchsz used as input on gpu
    out_buf::B      # The row vector of outputs

    buf_n::Int      # Counter for elements in buffer
    batch_sz::Int   # batch size
end

function AccumulatorLogPsi(net::NeuralNetwork, v::Union{gpuAStateBatch, gpuADoubleStateBatch},
                           batch_sz)
    #bnet    = cached(net, batch_sz)
    bnet    = net

    in_buf_cpu  = statecollect(v)
    in_buf_gpu  = deepcopy(v)
    out_buf     = out_similar(net)

    return AccumulatorLogPsiGPU(
        bnet,
        in_buf_cpu, in_buf_gpu, out_buf,
        0, batch_sz)
end


Base.length(a::AccumulatorLogPsiGPU) = a.batch_sz
Base.count(a::AccumulatorLogPsiGPU)  = a.buf_n
isfull(a::AccumulatorLogPsiGPU) = count(a) == length(a)

"""
    init!(c::ScalarBatchAccumulator)

Resets the internal counter of the accumulator, deleting
all previously accumulated (but not computed) values.
"""
init!(c::AccumulatorLogPsiGPU) = c.buf_n = 0
function init!(c::AccumulatorLogPsiGPU, σ)
    init!(c)

    # Reset the state so that later is faster to apply the changes
    statecopy!(c.in_buf_cpu, σ)
end

function (c::AccumulatorLogPsiGPU)(v::Union{AState, ADoubleState})
    @assert !isfull(c) "Pushed data to a full accumulator."

    # Increase the step in our internal buffer
    # this should be guaranteed to always be < max_capacity
    c.buf_n = c.buf_n + 1

    if v isa AState
        buf_i = unsafe_get_batch(c.in_buf_cpu, c.buf_n)
        buf_i .= v
    elseif v isa ADoubleState
        buf_r_i = unsafe_get_batch(row(c.in_buf_cpu), c.buf_n)
        buf_c_i = unsafe_get_batch(col(c.in_buf_cpu), c.buf_n)
        buf_r_i .= row(v)
        buf_c_i .= col(v)
    else
        throw("Not implemented")
    end

    return nothing
end

function (c::AccumulatorLogPsiGPU)(cngs)
    @assert !isfull(c) "Pushed data to a full accumulator."
    c.buf_n = c.buf_n + 1

    σp = unsafe_get_batch(c.in_buf_cpu, c.buf_n)
    apply!(σp, cngs)
end

function (c::AccumulatorLogPsiGPU)(cngs_l, cngs_r)
    @assert !isfull(c) "Pushed data to a full accumulator."
    c.buf_n = c.buf_n + 1

    σp = unsafe_get_batch(c.in_buf_cpu, c.buf_n)
    apply!(σp, cngs_l, cngs_r)
end

function process_accumulator!(c::AccumulatorLogPsiGPU)
    out = c.out_buf

    # copuy to gpu
    copy!(c.in_buf_gpu, c.in_buf_cpu)
    # Compute the batch of logψ neural networks
    logψ!(out, c.bnet, c.in_buf_gpu)

    # Reset the counter of the batch accumulator
    init!(c)

    return out
end
